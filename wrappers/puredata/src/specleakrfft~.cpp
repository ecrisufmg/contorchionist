#include "m_pd.h"
#include "core_ap_specleakrfft.hpp"
#include "../utils/include/pd_arg_parser.h"

#include <vector>
#include <memory>
#include <exception>

// Define the C-style struct for the Pd tilde object
typedef struct _specleakrfft_tilde {
    t_object x_obj;
    t_float x_f; // Dummy float for signal inlet
    t_float x_f_phase; // Dummy float for phase inlet

    std::unique_ptr<SpectralLeakRFFTProcessor<double>> processor;

    t_outlet* p_phase_outlet;

    // Parameters stored in the Pd object
    double p_threshold;
    double p_attack_alpha;
    double p_decay_time_s;
    double p_sample_rate; // Sample rate of the main patch
    double p_hop_size;
    size_t p_fft_size; // Configured FFT size

} t_specleakrfft_tilde;

// Declare the class
static t_class *specleakrfft_tilde_class;

// --- Method Prototypes ---
static void *specleakrfft_tilde_new(t_symbol *s, int argc, t_atom *argv);
static void specleakrfft_tilde_free(t_specleakrfft_tilde *x);
static void specleakrfft_tilde_dsp(t_specleakrfft_tilde *x, t_signal **sp);
static t_int *specleakrfft_tilde_perform(t_int *w);

// --- Setter Methods ---
static void specleakrfft_tilde_threshold(t_specleakrfft_tilde *x, t_floatarg f);
static void specleakrfft_tilde_attack(t_specleakrfft_tilde *x, t_floatarg f);
static void specleakrfft_tilde_decay(t_specleakrfft_tilde *x, t_symbol *s, int argc, t_atom *argv);


// --- DSP Routine ---
static t_int *specleakrfft_tilde_perform(t_int *w) {
    t_specleakrfft_tilde *x = (t_specleakrfft_tilde *)(w[1]);
    t_sample *in_mag = (t_sample *)(w[2]);
    t_sample *in_phase = (t_sample *)(w[3]);
    t_sample *out_mag = (t_sample *)(w[4]);
    t_sample *out_phase = (t_sample *)(w[5]);
    int n = (int)(w[6]); // Block size

    if (x->processor) {
        size_t num_bins = n / 2 + 1;

        // 1. Convert incoming POWER to MAGNITUDE
        std::vector<double> magnitude_frame(num_bins);
        for (size_t i = 0; i < num_bins; ++i) {
            magnitude_frame[i] = std::sqrt(std::max(0.0f, in_mag[i]));
        }

        // Create vector for phase frame
        std::vector<double> phase_frame(in_phase, in_phase + num_bins);

        // 2. Process the magnitude and phase frames
        x->processor->process_frame(magnitude_frame, phase_frame);

        // 3. Convert processed MAGNITUDE back to POWER
        const auto& processed_magnitude_frame = x->processor->get_processed_frame();
        for (size_t i = 0; i < num_bins; ++i) {
            out_mag[i] = processed_magnitude_frame[i] * processed_magnitude_frame[i];
        }

        // 4. Output the processed phase
        const auto& processed_phase_frame = x->processor->get_processed_phase_frame();
        for (size_t i = 0; i < num_bins; ++i) {
            out_phase[i] = processed_phase_frame[i];
        }

        // Zero out the rest of the output buffers
        for (size_t i = num_bins; i < n; ++i) {
            out_mag[i] = 0.0;
            out_phase[i] = 0.0;
        }
    } else {
        // If processor isn't ready, just zero the outputs
        for (int i = 0; i < n; ++i) {
            out_mag[i] = 0.0;
            out_phase[i] = 0.0;
        }
    }

    return (w + 7);
}


// --- DSP Setup ---
static void specleakrfft_tilde_dsp(t_specleakrfft_tilde *x, t_signal **sp) {
    // The block size inside pfft~ corresponds to the FFT size.
    size_t pfft_size = sp[0]->s_n;

    // Check if the FFT size from pfft~ matches the configured size
    if (x->p_fft_size != pfft_size) {
        post("specleakrfft~: FFT size mismatch! Object configured with @fftsize %zu, but pfft~ is running at %zu.", x->p_fft_size, pfft_size);
        post("specleakrfft~: Resizing internal processor to match pfft~.");
        x->p_fft_size = pfft_size;
        if(x->processor) {
            x->processor->resize(x->p_fft_size);
        }
    }

    dsp_add(specleakrfft_tilde_perform, 6, x, sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, sp[3]->s_vec, sp[0]->s_n);
}

// --- Object Lifecycle ---

static void specleakrfft_tilde_free(t_specleakrfft_tilde *x) {
    // unique_ptr handles deletion
}

static void *specleakrfft_tilde_new(t_symbol *s, int argc, t_atom *argv) {
    t_specleakrfft_tilde *x = (t_specleakrfft_tilde *)pd_new(specleakrfft_tilde_class);
    if (!x) {
        return nullptr;
    }

    // Argument Parsing
    pd_utils::ArgParser parser(argc, argv, (t_object*)x);

    x->p_fft_size = static_cast<size_t>(parser.get_float("fftsize", 1024)); // Default to 1024

    // Instantiate the C++ processor
    try {
        x->processor = std::make_unique<SpectralLeakRFFTProcessor<double>>(x->p_fft_size);
    } catch (const std::exception& e) {
        pd_error(x, "specleakrfft~: EXCEPTION during processor creation: %s", e.what());
        return nullptr;
    }

    // Configure processor with remaining arguments
    x->p_sample_rate = sys_getsr(); // Get main patch sample rate
    x->p_hop_size = parser.get_float("hopsize", x->p_fft_size / 2.0f);

    specleakrfft_tilde_threshold(x, parser.get_float("threshold", 0.01f));
    specleakrfft_tilde_attack(x, parser.get_float("attack", 0.8f));

    x->p_decay_time_s = parser.get_float("decay", 4.0f);
    if (x->processor) {
        x->processor->set_decay_time_s(x->p_decay_time_s, x->p_sample_rate, x->p_hop_size);
    }

    // Create inlets and outlets
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
    outlet_new(&x->x_obj, &s_signal);
    x->p_phase_outlet = outlet_new(&x->x_obj, &s_signal);

    post("specleakrfft~: Spectral Leak RFFT tilde object created.");
    return (void *)x;
}

// --- Setters ---

static void specleakrfft_tilde_threshold(t_specleakrfft_tilde *x, t_floatarg f) {
    x->p_threshold = f;
    if (x->processor) x->processor->set_threshold(x->p_threshold);
}

static void specleakrfft_tilde_attack(t_specleakrfft_tilde *x, t_floatarg f) {
    x->p_attack_alpha = f;
    if (x->processor) x->processor->set_attack_alpha(x->p_attack_alpha);
}

static void specleakrfft_tilde_decay(t_specleakrfft_tilde *x, t_symbol *s, int argc, t_atom *argv) {
    if (argc < 1 || argv[0].a_type != A_FLOAT) {
        pd_error(x, "specleakrfft~: decay method requires a float decay time");
        return;
    }
    x->p_decay_time_s = atom_getfloat(&argv[0]);
    if (argc > 1 && argv[1].a_type == A_FLOAT) x->p_sample_rate = atom_getfloat(&argv[1]);
    if (argc > 2 && argv[2].a_type == A_FLOAT) x->p_hop_size = atom_getfloat(&argv[2]);

    if (x->processor) {
        x->processor->set_decay_time_s(x->p_decay_time_s, x->p_sample_rate, x->p_hop_size);
    }
}

// --- PD Class Setup ---
extern "C" {
    void specleakrfft_tilde_setup(void) {
        specleakrfft_tilde_class = class_new(gensym("specleakrfft~"),
                                             (t_newmethod)specleakrfft_tilde_new,
                                             (t_method)specleakrfft_tilde_free,
                                             sizeof(t_specleakrfft_tilde),
                                             CLASS_DEFAULT,
                                             A_GIMME, 0);

        CLASS_MAINSIGNALIN(specleakrfft_tilde_class, t_specleakrfft_tilde, x_f);
        class_addmethod(specleakrfft_tilde_class, (t_method)specleakrfft_tilde_dsp, gensym("dsp"), A_CANT, 0);

        // Setter methods
        class_addmethod(specleakrfft_tilde_class, (t_method)specleakrfft_tilde_threshold, gensym("threshold"), A_FLOAT, 0);
        class_addmethod(specleakrfft_tilde_class, (t_method)specleakrfft_tilde_attack, gensym("attack"), A_FLOAT, 0);
        class_addmethod(specleakrfft_tilde_class, (t_method)specleakrfft_tilde_decay, gensym("decay"), A_GIMME, 0);

        post("specleakrfft~: C++ Spectral Leak RFFT v1.1 (Signal-rate)");
    }
}

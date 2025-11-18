#include "m_pd.h"
#include "core_ap_spectrails.h"
#include "../utils/include/pd_arg_parser.h"

#include <memory>
#include <vector>
#include <exception>

// Define the C-style struct for the Pd tilde object
typedef struct _torch_spectrails_tilde {
    t_object x_obj;
    t_float x_f; // Dummy float for signal inlet (magnitude)
    t_float x_f_phase; // Dummy float for phase inlet

    std::unique_ptr<contorchionist::core::ap_spectrails::SpectralTrailsProcessor<float>> processor;

    t_outlet* p_phase_outlet;

    // Parameters stored in the Pd object
    float p_threshold;
    float p_attack;
    float p_decay;
    bool p_limiter_enabled;
    float p_max_value;
    size_t p_num_bins; // N/2 + 1
    size_t p_fft_size; // Full FFT size (N)

} t_torch_spectrails_tilde;

// Declare the class
static t_class *torch_spectrails_tilde_class;

// --- Method Prototypes ---
static void *torch_spectrails_tilde_new(t_symbol *s, int argc, t_atom *argv);
static void torch_spectrails_tilde_free(t_torch_spectrails_tilde *x);
static void torch_spectrails_tilde_dsp(t_torch_spectrails_tilde *x, t_signal **sp);
static t_int *torch_spectrails_tilde_perform(t_int *w);

// --- Setter Methods ---
static void torch_spectrails_tilde_threshold(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_attack(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_decay(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_limiter(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_maxvalue(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_reset(t_torch_spectrails_tilde *x);


// --- DSP Routine ---
static t_int *torch_spectrails_tilde_perform(t_int *w) {
    t_torch_spectrails_tilde *x = (t_torch_spectrails_tilde *)(w[1]);
    t_sample *in_mag = (t_sample *)(w[2]);
    t_sample *in_phase = (t_sample *)(w[3]);
    t_sample *out_mag = (t_sample *)(w[4]);
    t_sample *out_phase = (t_sample *)(w[5]);
    int n = (int)(w[6]); // Block size (should be FFT size)

    if (!x->processor) {
        // If processor isn't ready, zero the outputs
        for (int i = 0; i < n; ++i) {
            out_mag[i] = 0.0f;
            out_phase[i] = 0.0f;
        }
        return (w + 7);
    }

    try {
        // Calculate number of useful bins (N/2 + 1)
        size_t num_bins = n / 2 + 1;

        // Create torch tensors from input buffers (CPU)
        auto mag_tensor = torch::from_blob(
            in_mag, 
            {static_cast<long>(num_bins)}, 
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone(); // Clone to avoid aliasing issues

        auto phase_tensor = torch::from_blob(
            in_phase, 
            {static_cast<long>(num_bins)}, 
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone();

        // Process through the spectral trails processor
        auto output_tensors = x->processor->process_frame(mag_tensor, phase_tensor);
        
        auto& processed_mag = output_tensors[0];
        auto& processed_phase = output_tensors[1];

        // Ensure outputs are on CPU
        processed_mag = processed_mag.to(torch::kCPU);
        processed_phase = processed_phase.to(torch::kCPU);

        // Copy processed bins to output buffers
        auto mag_accessor = processed_mag.accessor<float, 1>();
        auto phase_accessor = processed_phase.accessor<float, 1>();

        for (size_t i = 0; i < num_bins; ++i) {
            out_mag[i] = mag_accessor[i];
            out_phase[i] = phase_accessor[i];
        }

        // Zero out the rest of the output buffers (redundant bins)
        for (size_t i = num_bins; i < static_cast<size_t>(n); ++i) {
            out_mag[i] = 0.0f;
            out_phase[i] = 0.0f;
        }

    } catch (const std::exception& e) {
        pd_error(x, "torch.spectrails~: Exception in perform: %s", e.what());
        
        // On error, zero outputs
        for (int i = 0; i < n; ++i) {
            out_mag[i] = 0.0f;
            out_phase[i] = 0.0f;
        }
    }

    return (w + 7);
}


// --- DSP Setup ---
static void torch_spectrails_tilde_dsp(t_torch_spectrails_tilde *x, t_signal **sp) {
    // The block size inside pfft~ corresponds to the FFT size
    size_t pfft_size = sp[0]->s_n;

    // Check if the FFT size from pfft~ matches the configured size
    if (x->p_fft_size != pfft_size) {
        post("torch.spectrails~: FFT size mismatch! Object configured with fftsize %zu, but pfft~ is running at %zu.", 
             x->p_fft_size, pfft_size);
        post("torch.spectrails~: Resizing internal processor to match pfft~.");
        
        x->p_fft_size = pfft_size;
        x->p_num_bins = x->p_fft_size / 2 + 1;
        
        if (x->processor) {
            x->processor->resize(x->p_num_bins);
        }
    }

    dsp_add(torch_spectrails_tilde_perform, 6, x, 
            sp[0]->s_vec,  // in_mag
            sp[1]->s_vec,  // in_phase
            sp[2]->s_vec,  // out_mag
            sp[3]->s_vec,  // out_phase
            sp[0]->s_n);
}

// --- Object Lifecycle ---

static void torch_spectrails_tilde_free(t_torch_spectrails_tilde *x) {
    // unique_ptr handles deletion automatically
}

static void *torch_spectrails_tilde_new(t_symbol *s, int argc, t_atom *argv) {
    t_torch_spectrails_tilde *x = (t_torch_spectrails_tilde *)pd_new(torch_spectrails_tilde_class);
    if (!x) {
        return nullptr;
    }

    // Argument Parsing
    pd_utils::ArgParser parser(argc, argv, (t_object*)x);

    // Get FFT size (default 1024)
    x->p_fft_size = static_cast<size_t>(parser.get_float("fftsize", 1024.0f));
    x->p_num_bins = x->p_fft_size / 2 + 1;

    // Parse parameters with defaults
    x->p_threshold = parser.get_float("threshold", 0.01f);
    x->p_attack = parser.get_float("attack", 0.8f);
    x->p_decay = parser.get_float("decay", 0.999f);
    x->p_limiter_enabled = static_cast<bool>(parser.get_float("limiter", 0.0f));
    x->p_max_value = parser.get_float("maxvalue", 1.0f);

    // Instantiate the C++ processor
    try {
        x->processor = std::make_unique<contorchionist::core::ap_spectrails::SpectralTrailsProcessor<float>>(
            x->p_num_bins,
            torch::kCPU,
            false // verbose
        );

        // Configure processor with parsed parameters
        x->processor->set_threshold(x->p_threshold);
        x->processor->set_attack(x->p_attack);
        x->processor->set_decay(x->p_decay);
        x->processor->set_limiter_enabled(x->p_limiter_enabled);
        x->processor->set_max_value(x->p_max_value);

    } catch (const std::exception& e) {
        pd_error(x, "torch.spectrails~: Exception during processor creation: %s", e.what());
        return nullptr;
    }

    // Create inlets: signal inlet for phase (magnitude is default)
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
    
    // Create outlets: magnitude and phase
    outlet_new(&x->x_obj, &s_signal);
    x->p_phase_outlet = outlet_new(&x->x_obj, &s_signal);

    // Log creation with all parameters
    post("torch.spectrails~: ===== Object Created =====");
    post("  FFT Size: %zu", x->p_fft_size);
    post("  Num Bins: %zu", x->p_num_bins);
    post("  Threshold: %.6f", x->p_threshold);
    post("  Attack: %.6f", x->p_attack);
    post("  Decay: %.6f", x->p_decay);
    post("  Limiter: %s", x->p_limiter_enabled ? "ON" : "OFF");
    post("  Max Value: %.6f", x->p_max_value);
    post("=====================================");

    return (void *)x;
}

// --- Setters ---

static void torch_spectrails_tilde_threshold(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->p_threshold = f;
    if (x->processor) {
        x->processor->set_threshold(x->p_threshold);
        post("torch.spectrails~: threshold set to %.6f", x->p_threshold);
    }
}

static void torch_spectrails_tilde_attack(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->p_attack = f;
    if (x->processor) {
        x->processor->set_attack(x->p_attack);
        post("torch.spectrails~: attack set to %.6f", x->p_attack);
    }
}

static void torch_spectrails_tilde_decay(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->p_decay = f;
    if (x->processor) {
        x->processor->set_decay(x->p_decay);
        post("torch.spectrails~: decay set to %.6f", x->p_decay);
    }
}

static void torch_spectrails_tilde_limiter(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->p_limiter_enabled = (f != 0.0f);
    if (x->processor) {
        x->processor->set_limiter_enabled(x->p_limiter_enabled);
        post("torch.spectrails~: limiter %s", x->p_limiter_enabled ? "enabled" : "disabled");
    }
}

static void torch_spectrails_tilde_maxvalue(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->p_max_value = f;
    if (x->processor) {
        x->processor->set_max_value(x->p_max_value);
        post("torch.spectrails~: max value set to %.6f", x->p_max_value);
    }
}

static void torch_spectrails_tilde_reset(t_torch_spectrails_tilde *x) {
    if (x->processor) {
        x->processor->reset_memory();
        post("torch.spectrails~: memory reset");
    }
}

// --- PD Class Setup --
extern "C" void setup_torch0x2espectrails_tilde(void) {
        torch_spectrails_tilde_class = class_new(gensym("torch.spectrails~"),
                                                 (t_newmethod)torch_spectrails_tilde_new,
                                                 (t_method)torch_spectrails_tilde_free,
                                                 sizeof(t_torch_spectrails_tilde),
                                                 CLASS_DEFAULT,
                                                 A_GIMME, 0);

        CLASS_MAINSIGNALIN(torch_spectrails_tilde_class, t_torch_spectrails_tilde, x_f);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_dsp, 
                       gensym("dsp"), A_CANT, 0);

        // Setter methods
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_threshold, 
                       gensym("threshold"), A_FLOAT, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_attack, 
                       gensym("attack"), A_FLOAT, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_decay, 
                       gensym("decay"), A_FLOAT, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_limiter, 
                       gensym("limiter"), A_FLOAT, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_maxvalue, 
                       gensym("maxvalue"), A_FLOAT, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_reset, 
                       gensym("reset"), A_NULL, 0);

        post("torch.spectrails~: Spectral Trails Processor v1.0");
        post("  Use with torch.rfft~ and torch.irfft~ inside pfft~");
        post("  Parameters: @threshold @attack @decay @limiter @maxvalue");
    
}

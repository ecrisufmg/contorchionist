/**
 * @file torch.spectrogram~.cpp
 * @brief Max/MSP wrapper for the libtorch-based Spectrogram processor.
 *
 */

#include "ext.h"
#include "ext_obex.h"
#include "z_dsp.h"
#include "ext_buffer.h"

#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include "../../../core/include/core_ap_spectrogram.h"
#include "../../../core/include/core_util_windowing.h"
#include "../../../core/include/core_util_conversions.h"
#include "../../../core/include/core_util_normalizations.h"
#include "../../../core/include/core_util_devices.h"

// Use a using declaration for convenience
using SpectrogramProcessor = contorchionist::core::ap_spectrogram::SpectrogramProcessor<float>;
using WindowType = contorchionist::core::util_windowing::Type;
using NormalizationType = contorchionist::core::util_normalizations::NormalizationType;
using SpectrumDataFormat = contorchionist::core::util_conversions::SpectrumDataFormat;

// Define the Max C-style struct
typedef struct _torch_spectrogram_max {
    t_pxobject ob; // The object itself (t_pxobject in MSP)
    std::unique_ptr<SpectrogramProcessor> processor;

    // Parameters stored in the Max object
    long x_n_fft;
    long x_hop_size;
    WindowType x_win_type;
    t_symbol *s_win_type_attr;
    NormalizationType x_norm_type;
    t_symbol *s_norm_type_attr;
    SpectrumDataFormat x_unit_type;
    t_symbol *s_unit_type_attr;
    torch::Device x_device;
    t_symbol *s_device_attr;
    long verbose;

    // DSP state
    double x_samplerate;
    std::vector<float> m_input_buffer_float;

    // Outlets
    void *m_outlet1; // Component 1 (e.g., Magnitude)
    void *m_outlet2; // Component 2 (e.g., Phase)

    // Clock for deferring output
    void *m_clock;
    std::vector<float> m_output_buf1;
    std::vector<float> m_output_buf2;

} t_torch_spectrogram_max;

// --- Method Prototypes ---
void *torch_spectrogram_max_new(t_symbol *s, long argc, t_atom *argv);
void torch_spectrogram_max_free(t_torch_spectrogram_max *x);
void torch_spectrogram_max_assist(t_torch_spectrogram_max *x, void *b, long m, long a, char *s);
void torch_spectrogram_max_dsp64(t_torch_spectrogram_max *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags);
void torch_spectrogram_max_perform64(t_torch_spectrogram_max *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam);
void torch_spectrogram_max_configure_processor(t_torch_spectrogram_max *x);
void torch_spectrogram_max_clock_task(t_torch_spectrogram_max *x);
void torch_spectrogram_max_output_frame(t_torch_spectrogram_max *x);

// Attribute Setters
t_max_err torch_spectrogram_max_attr_set_nfft(t_torch_spectrogram_max *x, void *attr, long argc, t_atom *argv);
t_max_err torch_spectrogram_max_attr_set_hopsize(t_torch_spectrogram_max *x, void *attr, long argc, t_atom *argv);
t_max_err torch_spectrogram_max_attr_set_wintype(t_torch_spectrogram_max *x, void *attr, long argc, t_atom *argv);
t_max_err torch_spectrogram_max_attr_set_norm(t_torch_spectrogram_max *x, void *attr, long argc, t_atom *argv);
t_max_err torch_spectrogram_max_attr_set_unit(t_torch_spectrogram_max *x, void *attr, long argc, t_atom *argv);
t_max_err torch_spectrogram_max_attr_set_device(t_torch_spectrogram_max *x, void *attr, long argc, t_atom *argv);

// Global class pointer
static t_class *s_torch_spectrogram_max_class = nullptr;

// --- Helper Functions ---
// (Will be added later)

// --- Max Object Lifecycle & DSP ---
void ext_main(void *r) {
    t_class *c = class_new("torch.spectrogram~", (method)torch_spectrogram_max_new, (method)torch_spectrogram_max_free,
                           (long)sizeof(t_torch_spectrogram_max), (method)NULL, A_GIMME, 0);

    class_addmethod(c, (method)torch_spectrogram_max_dsp64, "dsp64", A_CANT, 0);
    class_addmethod(c, (method)torch_spectrogram_max_assist, "assist", A_CANT, 0);

    // Attributes
    CLASS_ATTR_LONG(c, "n_fft", 0, t_torch_spectrogram_max, x_n_fft);
    CLASS_ATTR_ACCESSORS(c, "n_fft", NULL, (method)torch_spectrogram_max_attr_set_nfft);
    CLASS_ATTR_LABEL(c, "n_fft", 0, "FFT Size");

    CLASS_ATTR_LONG(c, "hop_size", 0, t_torch_spectrogram_max, x_hop_size);
    CLASS_ATTR_ACCESSORS(c, "hop_size", NULL, (method)torch_spectrogram_max_attr_set_hopsize);
    CLASS_ATTR_LABEL(c, "hop_size", 0, "Hop Size");

    CLASS_ATTR_LONG(c, "hop", 0, t_torch_spectrogram_max, x_hop_size);
    CLASS_ATTR_ACCESSORS(c, "hop", NULL, (method)torch_spectrogram_max_attr_set_hopsize);
    CLASS_ATTR_LABEL(c, "hop", 0, "Hop Size");

    CLASS_ATTR_SYM(c, "wintype", 0, t_torch_spectrogram_max, s_win_type_attr);
    CLASS_ATTR_ACCESSORS(c, "wintype", NULL, (method)torch_spectrogram_max_attr_set_wintype);
    CLASS_ATTR_LABEL(c, "wintype", 0, "Window Type");

    CLASS_ATTR_SYM(c, "norm", 0, t_torch_spectrogram_max, s_norm_type_attr);
    CLASS_ATTR_ACCESSORS(c, "norm", NULL, (method)torch_spectrogram_max_attr_set_norm);
    CLASS_ATTR_LABEL(c, "norm", 0, "Normalization Type");

    CLASS_ATTR_SYM(c, "unit", 0, t_torch_spectrogram_max, s_unit_type_attr);
    CLASS_ATTR_ACCESSORS(c, "unit", NULL, (method)torch_spectrogram_max_attr_set_unit);
    CLASS_ATTR_LABEL(c, "unit", 0, "Output Unit");

    CLASS_ATTR_SYM(c, "device", 0, t_torch_spectrogram_max, s_device_attr);
    CLASS_ATTR_ACCESSORS(c, "device", NULL, (method)torch_spectrogram_max_attr_set_device);
    CLASS_ATTR_LABEL(c, "device", 0, "Torch Device (cpu, cuda, mps)");

    CLASS_ATTR_LONG(c, "verbose", 0, t_torch_spectrogram_max, verbose);
    CLASS_ATTR_LABEL(c, "verbose", 0, "Enable Verbose Logging");

    class_dspinit(c);
    class_register(CLASS_BOX, c);
    s_torch_spectrogram_max_class = c;

    post("torch.spectrogram~ (libtorch) loaded");
}

// --- Helper Functions ---
static WindowType atom_to_window_type(t_atom *ap, t_object* owner) {
    if (atom_gettype(ap) == A_SYM) {
        std::string str = atom_getsym(ap)->s_name;
        try {
            return contorchionist::core::util_windowing::string_to_torch_window_type(str);
        } catch (const std::invalid_argument&) {
            object_error(owner, "unknown window type '%s', defaulting to hann", str.c_str());
        }
    }
    return WindowType::HANN;
}

static NormalizationType atom_to_normalization_type(t_atom *ap, t_object* owner) {
    if (atom_gettype(ap) == A_SYM) {
        std::string str = atom_getsym(ap)->s_name;
         try {
            return contorchionist::core::util_normalizations::string_to_normalization_type(str, false);
        } catch (const std::invalid_argument&) {
            object_error(owner, "unknown normalization type '%s', defaulting to window", str.c_str());
        }
    }
    return NormalizationType::WINDOW;
}

static SpectrumDataFormat atom_to_spectrum_data_format(t_atom *ap, t_object* owner) {
    if (atom_gettype(ap) == A_SYM) {
        std::string str = atom_getsym(ap)->s_name;
        try {
            return contorchionist::core::util_conversions::string_to_spectrum_data_format(str);
        } catch (const std::invalid_argument&) {
            object_error(owner, "unknown unit type '%s', defaulting to magphase", str.c_str());
        }
    }
    return SpectrumDataFormat::MAGPHASE;
}

static torch::Device atom_to_device(t_atom* ap, t_object* owner) {
    if (atom_gettype(ap) == A_SYM) {
        std::string str = atom_getsym(ap)->s_name;
        auto device_result = contorchionist::core::util_devices::parse_torch_device(str);
        if (device_result.second.empty()) {
            return device_result.first;
        } else {
            object_error(owner, "invalid device string '%s': %s. Defaulting to CPU", str.c_str(), device_result.second.c_str());
        }
    }
    return torch::kCPU;
}


// --- Max Object Lifecycle & DSP ---

void *torch_spectrogram_max_new(t_symbol *s, long argc, t_atom *argv) {
    t_torch_spectrogram_max *x = (t_torch_spectrogram_max *)object_alloc(s_torch_spectrogram_max_class);

    if (x) {
        dsp_setup((t_pxobject *)x, 1);

        // Initialize parameters to defaults
        x->x_n_fft = 1024;
        x->x_hop_size = 512;
        x->x_win_type = WindowType::HANN;
        x->s_win_type_attr = gensym("hann");
        x->x_norm_type = NormalizationType::WINDOW;
        x->s_norm_type_attr = gensym("window");
        x->x_unit_type = SpectrumDataFormat::MAGPHASE;
        x->s_unit_type_attr = gensym("magphase");
        x->x_device = torch::kCPU;
        x->s_device_attr = gensym("cpu");
        x->verbose = 0;
        x->processor = nullptr;
        x->x_samplerate = sys_getsr();
        if (x->x_samplerate <= 0) x->x_samplerate = 44100.0; // Default SR if not available

        // Setup outlets and clock
        x->m_outlet2 = listout((t_object *)x);
        x->m_outlet1 = listout((t_object *)x);
        x->m_clock = clock_new(x, (method)torch_spectrogram_max_output_frame);

        // Process attribute arguments
        attr_args_process(x, argc, argv);

        // Configure the processor for the first time
        torch_spectrogram_max_configure_processor(x);
    }
    return (x);
}

void torch_spectrogram_max_free(t_torch_spectrogram_max *x) {
    dsp_free((t_pxobject *)x);
    if (x->m_clock) {
        object_free(x->m_clock);
    }
    // unique_ptr will handle the processor memory
}

void torch_spectrogram_max_assist(t_torch_spectrogram_max *x, void *b, long m, long a, char *s) {
    if (m == ASSIST_INLET) {
        snprintf(s, 256, "Signal Input / Messages");
    } else { // ASSIST_OUTLET
        if (a == 0) {
            snprintf(s, 256, "Component 1 (list)");
        } else if (a == 1) {
            snprintf(s, 256, "Component 2 (list)");
        }
    }
}

void torch_spectrogram_max_configure_processor(t_torch_spectrogram_max *x) {
    if (x->x_n_fft <= 0 || x->x_hop_size <= 0) {
        object_error((t_object*)x, "n_fft and hop_size must be positive. Processor not configured.");
        x->processor.reset();
        return;
    }
     if (x->x_samplerate <= 0) {
        object_error((t_object*)x, "Sample rate must be positive. Processor not configured.");
        x->processor.reset();
        return;
    }

    try {
        if (x->verbose) {
            object_post((t_object*)x, "Configuring processor: n_fft=%ld, hop=%ld, sr=%.2f",
                        x->x_n_fft, x->x_hop_size, x->x_samplerate);
        }
        x->processor = std::make_unique<SpectrogramProcessor>(
            x->x_n_fft,
            x->x_hop_size,
            x->x_win_type,
            x->x_unit_type,
            x->x_norm_type,
            x->x_device,
            x->x_samplerate,
            (bool)x->verbose
        );
         if (x->verbose) {
            object_post((t_object*)x, "Processor configured successfully.");
        }
    } catch (const std::exception& e) {
        object_error((t_object *)x, "Error configuring processor: %s", e.what());
        x->processor.reset();
    }
}

void torch_spectrogram_max_dsp64(t_torch_spectrogram_max *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags) {
    if (samplerate != x->x_samplerate) {
        x->x_samplerate = samplerate;
        if (x->processor) {
            x->processor->set_sampling_rate(samplerate);
        } else {
            torch_spectrogram_max_configure_processor(x);
        }
    }

    try {
        x->m_input_buffer_float.resize(maxvectorsize);
    } catch (const std::bad_alloc& e) {
        object_error((t_object*)x, "Failed to allocate temporary buffer: %s", e.what());
        return;
    }

    object_method(dsp64, gensym("dsp_add64"), x, torch_spectrogram_max_perform64, 0, NULL);
}

void torch_spectrogram_max_perform64(t_torch_spectrogram_max *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam) {
    if (!x->processor || numins < 1) {
        return;
    }

    double *in_signal = ins[0];
    long n = sampleframes;

    // Convert input from double to float
    for (long i = 0; i < n; ++i) {
        x->m_input_buffer_float[i] = static_cast<float>(in_signal[i]);
    }

    // Process audio and check if a new frame is ready
    if (x->processor->process(x->m_input_buffer_float.data(), n, x->m_output_buf1, x->m_output_buf2)) {
        // A new frame is ready, schedule the clock to output it in the main thread
        clock_set(x->m_clock, 0);
    }
}

void torch_spectrogram_max_output_frame(t_torch_spectrogram_max *x) {
    // This function is executed by the clock in the main thread
    if (!x->m_output_buf1.empty()) {
        long num_elements = x->m_output_buf1.size();
        t_atom* atom_list = (t_atom*)sysmem_newptr(num_elements * sizeof(t_atom));
        if (atom_list) {
            for (long i = 0; i < num_elements; ++i) {
                atom_setfloat(&atom_list[i], x->m_output_buf1[i]);
            }
            outlet_list(x->m_outlet1, NULL, static_cast<short>(num_elements), atom_list);
            sysmem_freeptr(atom_list);
        }
    }

    if (!x->m_output_buf2.empty()) {
        long num_elements = x->m_output_buf2.size();
        t_atom* atom_list = (t_atom*)sysmem_newptr(num_elements * sizeof(t_atom));
        if (atom_list) {
            for (long i = 0; i < num_elements; ++i) {
                atom_setfloat(&atom_list[i], x->m_output_buf2[i]);
            }
            outlet_list(x->m_outlet2, NULL, static_cast<short>(num_elements), atom_list);
            sysmem_freeptr(atom_list);
        }
    }
}


// --- Attribute Setters ---

t_max_err torch_spectrogram_max_attr_set_nfft(t_torch_spectrogram_max *x, void *attr, long argc, t_atom *argv) {
    if (argc > 0) {
        long new_n_fft = atom_getlong(argv);
        if (new_n_fft <= 0) {
            object_error((t_object*)x, "n_fft must be positive");
            return MAX_ERR_GENERIC;
        }
        if (x->x_n_fft != new_n_fft) {
            x->x_n_fft = new_n_fft;
            torch_spectrogram_max_configure_processor(x);
        }
    }
    return MAX_ERR_NONE;
}

t_max_err torch_spectrogram_max_attr_set_hopsize(t_torch_spectrogram_max *x, void *attr, long argc, t_atom *argv) {
    if (argc > 0) {
        long new_hop_size = atom_getlong(argv);
        if (new_hop_size <= 0) {
            object_error((t_object*)x, "hop_size must be positive");
            return MAX_ERR_GENERIC;
        }
        if (x->x_hop_size != new_hop_size) {
            x->x_hop_size = new_hop_size;
            torch_spectrogram_max_configure_processor(x);
        }
    }
    return MAX_ERR_NONE;
}

t_max_err torch_spectrogram_max_attr_set_wintype(t_torch_spectrogram_max *x, void *attr, long argc, t_atom *argv) {
    if (argc > 0 && atom_gettype(argv) == A_SYM) {
        x->s_win_type_attr = atom_getsym(argv);
        WindowType new_type = atom_to_window_type(argv, (t_object*)x);
        if (x->x_win_type != new_type) {
            x->x_win_type = new_type;
            if (x->processor) x->processor->set_window_type(x->x_win_type);
            else torch_spectrogram_max_configure_processor(x);
        }
    }
    return MAX_ERR_NONE;
}

t_max_err torch_spectrogram_max_attr_set_norm(t_torch_spectrogram_max *x, void *attr, long argc, t_atom *argv) {
    if (argc > 0 && atom_gettype(argv) == A_SYM) {
        x->s_norm_type_attr = atom_getsym(argv);
        NormalizationType new_type = atom_to_normalization_type(argv, (t_object*)x);
        if (x->x_norm_type != new_type) {
            x->x_norm_type = new_type;
            if (x->processor) x->processor->set_normalization_type(x->x_norm_type);
            else torch_spectrogram_max_configure_processor(x);
        }
    }
    return MAX_ERR_NONE;
}

t_max_err torch_spectrogram_max_attr_set_unit(t_torch_spectrogram_max *x, void *attr, long argc, t_atom *argv) {
    if (argc > 0 && atom_gettype(argv) == A_SYM) {
        x->s_unit_type_attr = atom_getsym(argv);
        SpectrumDataFormat new_format = atom_to_spectrum_data_format(argv, (t_object*)x);
        if (x->x_unit_type != new_format) {
            x->x_unit_type = new_format;
            if (x->processor) x->processor->set_output_format(x->x_unit_type);
            else torch_spectrogram_max_configure_processor(x);
        }
    }
    return MAX_ERR_NONE;
}

t_max_err torch_spectrogram_max_attr_set_device(t_torch_spectrogram_max *x, void *attr, long argc, t_atom *argv) {
    if (argc > 0 && atom_gettype(argv) == A_SYM) {
        x->s_device_attr = atom_getsym(argv);
        torch::Device new_device = atom_to_device(argv, (t_object*)x);
        if (x->x_device.type() != new_device.type() || x->x_device.index() != new_device.index()) {
            x->x_device = new_device;
            torch_spectrogram_max_configure_processor(x); // Recreate processor for new device
        }
    }
    return MAX_ERR_NONE;
}

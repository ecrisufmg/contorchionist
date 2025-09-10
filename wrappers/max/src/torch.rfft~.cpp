/**
 * @file torch.rfft~.cpp
 * @brief Max/MSP wrapper for the libtorch-based RFFT processor.
 * @author (Your Name)
 */

#include "ext.h"
#include "ext_obex.h"
#include "z_dsp.h"

#include "../../../core/include/core_ap_rfft.h"
#include "../../../core/include/core_util_windowing.h"
#include "../../../core/include/core_util_conversions.h"
#include "../../../core/include/core_util_normalizations.h"

#include <string>
#include <vector>
#include <algorithm>

// Use a using declaration for convenience
using RFFTProcessor = contorchionist::core::ap_rfft::RFFTProcessor<float>;
using WindowType = contorchionist::core::util_windowing::Type;
using NormalizationType = contorchionist::core::util_normalizations::NormalizationType;
using SpectrumDataFormat = contorchionist::core::util_conversions::SpectrumDataFormat;

// Define the Max C-style struct
typedef struct _torch_rfft_max {
    t_pxobject ob; // The object itself (t_pxobject in MSP)
    RFFTProcessor* processor; // Pointer to the C++ RFFTProcessor object

    // Parameters stored in the Max object
    WindowType x_win_type;
    t_symbol *s_win_type_attr;
    NormalizationType x_norm_type;
    t_symbol *s_norm_type_attr;
    SpectrumDataFormat x_unit_type;
    t_symbol *s_unit_type_attr;
    long x_overlap_factor;
    long x_windowing_enabled;
    long verbose;
    
    // DSP state
    long x_vectorsize;
    double x_samplerate;

    // Outlets
    void *m_outlet1; // Real, Mag, Power, dB
    void *m_outlet2; // Imag, Phase

    // Buffers for float/double conversion
    std::vector<float> m_input_buffer_float;
    std::vector<float> m_output1_buffer_float;
    std::vector<float> m_output2_buffer_float;

} t_torch_rfft_max;

// --- Method Prototypes ---
void *torch_rfft_max_new(t_symbol *s, long argc, t_atom *argv);
void torch_rfft_max_free(t_torch_rfft_max *x);
void torch_rfft_max_assist(t_torch_rfft_max *x, void *b, long m, long a, char *s);
void torch_rfft_max_dsp64(t_torch_rfft_max *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags);
void torch_rfft_max_perform64(t_torch_rfft_max *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam);
void torch_rfft_tilde_update_processor_settings(t_torch_rfft_max *x, long n, double sr);

// Attribute Setters
t_max_err torch_rfft_max_attr_set_wintype(t_torch_rfft_max *x, void *attr, long argc, t_atom *argv);
t_max_err torch_rfft_max_attr_set_norm(t_torch_rfft_max *x, void *attr, long argc, t_atom *argv);
t_max_err torch_rfft_max_attr_set_unit(t_torch_rfft_max *x, void *attr, long argc, t_atom *argv);
t_max_err torch_rfft_max_attr_set_overlap(t_torch_rfft_max *x, void *attr, long argc, t_atom *argv);
t_max_err torch_rfft_max_attr_set_winenable(t_torch_rfft_max *x, void *attr, long argc, t_atom *argv);


// Global class pointer
static t_class *s_torch_rfft_max_class = nullptr;

// --- Helper Functions ---
static WindowType atom_to_window_type(t_atom *ap) {
    if (atom_gettype(ap) == A_SYM) {
        std::string str = atom_getsym(ap)->s_name;
        try {
            return contorchionist::core::util_windowing::string_to_torch_window_type(str);
        } catch (const std::invalid_argument&) {
            object_post(nullptr, "torch.rfft~: unknown window type '%s', defaulting to rectangular", str.c_str());
        }
    }
    return WindowType::RECTANGULAR;
}

static NormalizationType atom_to_normalization_type(t_atom *ap) {
    if (atom_gettype(ap) == A_SYM) {
        std::string str = atom_getsym(ap)->s_name;
         try {
            return contorchionist::core::util_normalizations::string_to_normalization_type(str, false);
        } catch (const std::invalid_argument&) {
            object_post(nullptr, "torch.rfft~: unknown normalization type '%s', defaulting to none", str.c_str());
        }
    }
    return NormalizationType::NONE;
}

static SpectrumDataFormat atom_to_spectrum_data_format(t_atom *ap) {
    if (atom_gettype(ap) == A_SYM) {
        std::string str = atom_getsym(ap)->s_name;
        try {
            return contorchionist::core::util_conversions::string_to_spectrum_data_format(str);
        } catch (const std::invalid_argument&) {
            object_post(nullptr, "torch.rfft~: unknown unit type '%s', defaulting to complex", str.c_str());
        }
    }
    return SpectrumDataFormat::COMPLEX;
}

// --- Max Object Lifecycle & DSP ---
void ext_main(void *r) {
    t_class *c = class_new("torch.rfft~", (method)torch_rfft_max_new, (method)torch_rfft_max_free,
                           (long)sizeof(t_torch_rfft_max), 0L, A_GIMME, 0);

    class_addmethod(c, (method)torch_rfft_max_dsp64, "dsp64", A_CANT, 0);
    class_addmethod(c, (method)torch_rfft_max_assist, "assist", A_CANT, 0);

    // Attributes
    CLASS_ATTR_SYM(c, "wintype", 0, t_torch_rfft_max, s_win_type_attr);
    CLASS_ATTR_ACCESSORS(c, "wintype", NULL, (method)torch_rfft_max_attr_set_wintype);
    CLASS_ATTR_LABEL(c, "wintype", 0, "Window Type (symbol: rect, hann, etc.)");

    CLASS_ATTR_SYM(c, "norm", 0, t_torch_rfft_max, s_norm_type_attr);
    CLASS_ATTR_ACCESSORS(c, "norm", NULL, (method)torch_rfft_max_attr_set_norm);
    CLASS_ATTR_LABEL(c, "norm", 0, "Normalization Type (symbol: none, window, etc.)");

    CLASS_ATTR_SYM(c, "unit", 0, t_torch_rfft_max, s_unit_type_attr);
    CLASS_ATTR_ACCESSORS(c, "unit", NULL, (method)torch_rfft_max_attr_set_unit);
    CLASS_ATTR_LABEL(c, "unit", 0, "Output Unit (symbol: complex, magphase, etc.)");

    CLASS_ATTR_LONG(c, "overlap", 0, t_torch_rfft_max, x_overlap_factor);
    CLASS_ATTR_ACCESSORS(c, "overlap", NULL, (method)torch_rfft_max_attr_set_overlap);
    CLASS_ATTR_LABEL(c, "overlap", 0, "Overlap Factor (power of 2)");

    CLASS_ATTR_LONG(c, "winenable", 0, t_torch_rfft_max, x_windowing_enabled);
    CLASS_ATTR_ACCESSORS(c, "winenable", NULL, (method)torch_rfft_max_attr_set_winenable);
    CLASS_ATTR_LABEL(c, "winenable", 0, "Windowing Enabled (0 or 1)");

    CLASS_ATTR_LONG(c, "verbose", 0, t_torch_rfft_max, verbose);
    CLASS_ATTR_LABEL(c, "verbose", 0, "Enable Verbose Logging (0 or 1)");

    class_dspinit(c);
    class_register(CLASS_BOX, c);
    s_torch_rfft_max_class = c;

    post("torch.rfft~ (libtorch) loaded");
}

void *torch_rfft_max_new(t_symbol *s, long argc, t_atom *argv) {
    t_torch_rfft_max *x = (t_torch_rfft_max *)object_alloc(s_torch_rfft_max_class);

    if (x) {
        dsp_setup((t_pxobject *)x, 1);

        // Initialize parameters to defaults
        x->x_win_type = WindowType::RECTANGULAR;
        x->s_win_type_attr = gensym("rectangular");
        x->x_norm_type = NormalizationType::NONE;
        x->s_norm_type_attr = gensym("none");
        x->x_unit_type = SpectrumDataFormat::COMPLEX;
        x->s_unit_type_attr = gensym("complex");
        x->x_overlap_factor = 1;
        x->x_windowing_enabled = 0;
        x->verbose = 0;
        x->processor = nullptr;
        
        // Initialize DSP state
        x->x_vectorsize = 0;
        x->x_samplerate = 0.0;

        // Process attribute arguments
        attr_args_process(x, argc, argv);

        // Create outlets
        x->m_outlet2 = outlet_new((t_object *)x, "signal");
        x->m_outlet1 = outlet_new((t_object *)x, "signal");

        // Create the C++ processor object
        try {
            x->processor = new RFFTProcessor(
                torch::kCPU,
                x->x_win_type,
                (bool)x->x_windowing_enabled,
                (bool)x->verbose,
                x->x_norm_type,
                x->x_unit_type
            );
        } catch (const std::exception& e) {
            object_error((t_object *)x, "Failed to create torch.rfft~ processor: %s", e.what());
        }
    }
    return (x);
}

void torch_rfft_max_free(t_torch_rfft_max *x) {
    dsp_free((t_pxobject *)x);
    if (x->processor) {
        delete x->processor;
        x->processor = nullptr;
    }
}

void torch_rfft_max_assist(t_torch_rfft_max *x, void *b, long m, long a, char *s) {
    if (m == ASSIST_INLET) {
        snprintf(s, 256, "Signal Input / Messages");
    } else { // ASSIST_OUTLET
        if (a == 0) {
            snprintf(s, 256, "Component 1 (Real, Mag, Power, dB)");
        } else if (a == 1) {
            snprintf(s, 256, "Component 2 (Imag, Phase)");
        }
    }
}

void torch_rfft_tilde_update_processor_settings(t_torch_rfft_max *x, long n, double sr) {
    if (!x->processor) return;

    try {
        x->processor->set_normalization(
            1,  // 1 = forward FFT
            2,  // 2 = RFFT mode
            n,
            x->x_norm_type,
            (float)sr,
            (float)x->x_overlap_factor
        );
    } catch (const std::exception& e) {
        object_error((t_object *)x, "Error updating rfft processor settings: %s", e.what());
    }
}


void torch_rfft_max_dsp64(t_torch_rfft_max *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags) {
    if (!x->processor) {
        object_error((t_object*)x, "torch.rfft~: processor not initialized. Cannot start DSP.");
        return;
    }
    
    // Store current DSP settings
    x->x_vectorsize = maxvectorsize;
    x->x_samplerate = samplerate;

    // Update processor with actual block size and sample rate
    torch_rfft_tilde_update_processor_settings(x, maxvectorsize, samplerate);

    // Resize temp buffers
    try {
        x->m_input_buffer_float.resize(maxvectorsize);
        x->m_output1_buffer_float.resize(maxvectorsize, 0.f);
        x->m_output2_buffer_float.resize(maxvectorsize, 0.f);
    } catch (const std::bad_alloc& e) {
        object_error((t_object*)x, "Failed to allocate temporary buffers: %s", e.what());
        return;
    }

    object_method(dsp64, gensym("dsp_add64"), x, torch_rfft_max_perform64, 0, NULL);
}

void torch_rfft_max_perform64(t_torch_rfft_max *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam) {
    if (!x->processor || numins < 1 || numouts < 2) {
        return;
    }

    double *in_signal = ins[0];
    double *out1_signal = outs[0];
    double *out2_signal = outs[1];
    long n = sampleframes;

    // Convert input from double to float
    for (long i = 0; i < n; ++i) {
        x->m_input_buffer_float[i] = static_cast<float>(in_signal[i]);
    }

    // Zero output buffers
    std::fill(x->m_output1_buffer_float.begin(), x->m_output1_buffer_float.end(), 0.f);
    std::fill(x->m_output2_buffer_float.begin(), x->m_output2_buffer_float.end(), 0.f);

    try {
        auto cpu_tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        torch::Tensor input_tensor_cpu = torch::from_blob(x->m_input_buffer_float.data(), {n}, cpu_tensor_options);

        std::vector<torch::Tensor> result_vec = x->processor->process_rfft(input_tensor_cpu);

        if (!result_vec.empty()) {
            SpectrumDataFormat current_format = x->processor->get_output_format();
            torch::Tensor first_component_tensor, second_component_tensor;

            if (current_format == SpectrumDataFormat::COMPLEX) {
                if (result_vec.size() == 1 && result_vec[0].is_complex()) {
                    torch::Tensor complex_spectrum = result_vec[0].cpu();
                    first_component_tensor = torch::real(complex_spectrum).contiguous();
                    second_component_tensor = torch::imag(complex_spectrum).contiguous();
                }
            } else {
                if (result_vec.size() == 2) {
                    first_component_tensor = result_vec[0].cpu().contiguous();
                    second_component_tensor = result_vec[1].cpu().contiguous();
                }
            }

            if (first_component_tensor.defined()) {
                long num_to_copy = std::min((long)x->m_output1_buffer_float.size(), (long)first_component_tensor.numel());
                memcpy(x->m_output1_buffer_float.data(), first_component_tensor.data_ptr<float>(), num_to_copy * sizeof(float));
            }
            if (second_component_tensor.defined()) {
                long num_to_copy = std::min((long)x->m_output2_buffer_float.size(), (long)second_component_tensor.numel());
                memcpy(x->m_output2_buffer_float.data(), second_component_tensor.data_ptr<float>(), num_to_copy * sizeof(float));
            }
        }

    } catch (const std::exception& e) {
        if (x->verbose) {
            object_error((t_object *)x, "torch.rfft~ error: %s", e.what());
        }
    }

    // Convert output from float to double
    for (long i = 0; i < n; ++i) {
        out1_signal[i] = static_cast<double>(x->m_output1_buffer_float[i]);
        out2_signal[i] = static_cast<double>(x->m_output2_buffer_float[i]);
    }
}


// --- Attribute Setters ---

t_max_err torch_rfft_max_attr_set_wintype(t_torch_rfft_max *x, void *attr, long argc, t_atom *argv) {
    if (argc > 0 && atom_gettype(argv) == A_SYM) {
        x->s_win_type_attr = atom_getsym(argv);
        x->x_win_type = atom_to_window_type(argv);
        if (x->processor) {
             x->processor->set_window_type(x->x_win_type);
             if (x->x_vectorsize > 0) {
                 torch_rfft_tilde_update_processor_settings(x, x->x_vectorsize, x->x_samplerate);
             }
        }
    }
    return MAX_ERR_NONE;
}

t_max_err torch_rfft_max_attr_set_norm(t_torch_rfft_max *x, void *attr, long argc, t_atom *argv) {
    if (argc > 0 && atom_gettype(argv) == A_SYM) {
        x->s_norm_type_attr = atom_getsym(argv);
        x->x_norm_type = atom_to_normalization_type(argv);
        if (x->processor) {
            x->processor->set_normalization_type(x->x_norm_type);
            if (x->x_vectorsize > 0) {
                torch_rfft_tilde_update_processor_settings(x, x->x_vectorsize, x->x_samplerate);
            }
        }
    }
    return MAX_ERR_NONE;
}

t_max_err torch_rfft_max_attr_set_unit(t_torch_rfft_max *x, void *attr, long argc, t_atom *argv) {
    if (argc > 0 && atom_gettype(argv) == A_SYM) {
        x->s_unit_type_attr = atom_getsym(argv);
        x->x_unit_type = atom_to_spectrum_data_format(argv);
        if (x->processor) x->processor->set_output_format(x->x_unit_type);
    }
    return MAX_ERR_NONE;
}

t_max_err torch_rfft_max_attr_set_overlap(t_torch_rfft_max *x, void *attr, long argc, t_atom *argv) {
    if (argc > 0 && atom_gettype(argv) == A_LONG) {
        long new_overlap = atom_getlong(argv);
        if (new_overlap > 0 && (new_overlap & (new_overlap - 1)) == 0) { // Check for power of 2
            x->x_overlap_factor = new_overlap;
            if (x->processor && x->x_vectorsize > 0) {
                torch_rfft_tilde_update_processor_settings(x, x->x_vectorsize, x->x_samplerate);
            }
        } else {
            object_error((t_object *)x, "torch.rfft~: overlap factor must be a power of 2.");
        }
    }
    return MAX_ERR_NONE;
}

t_max_err torch_rfft_max_attr_set_winenable(t_torch_rfft_max *x, void *attr, long argc, t_atom *argv) {
    if (argc > 0 && atom_gettype(argv) == A_LONG) {
        x->x_windowing_enabled = atom_getlong(argv);
        if (x->processor) {
            x->processor->enable_windowing((bool)x->x_windowing_enabled);
            if (x->x_vectorsize > 0) {
                torch_rfft_tilde_update_processor_settings(x, x->x_vectorsize, x->x_samplerate);
            }
        }
    }
    return MAX_ERR_NONE;
}

/**
 * @file torch.rmsoverlap~.cpp
 * @brief Max/MSP wrapper for the libtorch-based RMSOverlap class.
 * @author (Your Name)
 */

#include "ext.h"
#include "ext_obex.h"
#include "z_dsp.h"

// Core library includes
#include "../../../core/include/core_ap_rmsoverlap.h"
#include "../../../core/include/core_util_windowing.h"

#include <string>
#include <vector>
#include <algorithm> // For std::min, std::max

// Use a using declaration for convenience
using RMSOverlap = contorchionist::core::ap_rmsoverlap::RMSOverlap<float>;

// Define the Max C-style struct
typedef struct _torch_rmsoverlap_max {
    t_pxobject ob; // The object itself (t_pxobject in MSP)
    RMSOverlap* analyzer; // Pointer to the C++ RMSOverlap object

    // Parameters stored in the Max object
    long x_window_size;
    long x_hop_size;
    contorchionist::core::util_windowing::Type x_win_type;
    t_symbol *s_win_type_attr;
    double x_zero_padding_factor;
    contorchionist::core::util_windowing::Alignment x_win_align;
    t_symbol *s_winalign_attr;
    RMSOverlap::NormalizationType x_norm_type;
    t_symbol *s_normtype_attr;
    double x_fixed_norm_multiplier;
    long verbose; // Use long for attribute system

    // Outlets
    void *m_outlet_rms;
    void *m_outlet_list;

    // Buffers for float/double conversion
    std::vector<float> m_input_buffer_float;
    std::vector<float> m_output_buffer_float;

} t_torch_rmsoverlap_max;

// --- Method Prototypes ---
void *torch_rmsoverlap_max_new(t_symbol *s, long argc, t_atom *argv);
void torch_rmsoverlap_max_free(t_torch_rmsoverlap_max *x);
void torch_rmsoverlap_max_assist(t_torch_rmsoverlap_max *x, void *b, long m, long a, char *s);
void torch_rmsoverlap_max_dsp64(t_torch_rmsoverlap_max *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags);
void torch_rmsoverlap_max_perform64(t_torch_rmsoverlap_max *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam);

// Attribute Setters
t_max_err torch_rmsoverlap_max_attr_set_winsize(t_torch_rmsoverlap_max *x, void *attr, long argc, t_atom *argv);
t_max_err torch_rmsoverlap_max_attr_set_hopsize(t_torch_rmsoverlap_max *x, void *attr, long argc, t_atom *argv);
t_max_err torch_rmsoverlap_max_attr_set_wintype(t_torch_rmsoverlap_max *x, void *attr, long argc, t_atom *argv);
t_max_err torch_rmsoverlap_max_attr_set_zeropad(t_torch_rmsoverlap_max *x, void *attr, long argc, t_atom *argv);
t_max_err torch_rmsoverlap_max_attr_set_winalign(t_torch_rmsoverlap_max *x, void *attr, long argc, t_atom *argv);
t_max_err torch_rmsoverlap_max_attr_set_normtype(t_torch_rmsoverlap_max *x, void *attr, long argc, t_atom *argv);
t_max_err torch_rmsoverlap_max_attr_set_normfixedval(t_torch_rmsoverlap_max *x, void *attr, long argc, t_atom *argv);

// Custom Message Methods
void torch_rmsoverlap_max_dump(t_torch_rmsoverlap_max *x);
void torch_rmsoverlap_max_get_window(t_torch_rmsoverlap_max *x);
void torch_rmsoverlap_max_get_window_sum(t_torch_rmsoverlap_max *x);
void torch_rmsoverlap_max_get_norm_vals(t_torch_rmsoverlap_max *x);

// Global class pointer
static t_class *s_torch_rmsoverlap_max_class = nullptr;

// --- Helper Functions ---
static contorchionist::core::util_windowing::Type atom_to_window_type(t_atom *ap) {
    if (atom_gettype(ap) == A_SYM) {
        std::string str = atom_getsym(ap)->s_name;
        if (str == "rect" || str == "rectangular") return contorchionist::core::util_windowing::Type::RECTANGULAR;
        if (str == "hann" || str == "hanning") return contorchionist::core::util_windowing::Type::HANN;
        if (str == "tri" || str == "triangular") return contorchionist::core::util_windowing::Type::BARTLETT;
        if (str == "hamm" || str == "hamming") return contorchionist::core::util_windowing::Type::HAMMING;
        if (str == "black" || str == "blackman") return contorchionist::core::util_windowing::Type::BLACKMAN;
        if (str == "cos" || str == "cosine") return contorchionist::core::util_windowing::Type::COSINE;
    }
    object_post(nullptr, "torch.rmsoverlap~: unknown window type, defaulting to hann");
    return contorchionist::core::util_windowing::Type::HANN;
}

static contorchionist::core::util_windowing::Alignment atom_to_window_alignment(t_atom *ap) {
    if (atom_gettype(ap) == A_SYM) {
        std::string str = atom_getsym(ap)->s_name;
        if (str == "l" || str == "left") return contorchionist::core::util_windowing::Alignment::LEFT;
        if (str == "c" || str == "center") return contorchionist::core::util_windowing::Alignment::CENTER;
        if (str == "r" || str == "right") return contorchionist::core::util_windowing::Alignment::RIGHT;
    }
    object_post(nullptr, "torch.rmsoverlap~: unknown window alignment, defaulting to center");
    return contorchionist::core::util_windowing::Alignment::CENTER;
}

static RMSOverlap::NormalizationType atom_to_normalization_type(t_atom *ap) {
    if (atom_gettype(ap) == A_SYM) {
        std::string str = atom_getsym(ap)->s_name;
        if (str == "win_rms") return RMSOverlap::NormalizationType::WINDOW_OVERLAP_RMS;
        if (str == "win_mean") return RMSOverlap::NormalizationType::WINDOW_OVERLAP_MEAN;
        if (str == "win_vals") return RMSOverlap::NormalizationType::WINDOW_OVERLAP_VALS;
        if (str == "overlap_inverse" || str == "overlap") return RMSOverlap::NormalizationType::OVERLAP_INVERSE;
        if (str == "fixed") return RMSOverlap::NormalizationType::FIXED_MULTIPLIER;
        if (str == "none") return RMSOverlap::NormalizationType::NONE;
    }
    object_post(nullptr, "torch.rmsoverlap~: unknown normalization type, defaulting to win_rms");
    return RMSOverlap::NormalizationType::WINDOW_OVERLAP_RMS;
}

// --- Max Object Lifecycle & DSP ---
void ext_main(void *r) {
    t_class *c = class_new("torch.rmsoverlap~", (method)torch_rmsoverlap_max_new, (method)torch_rmsoverlap_max_free,
                           (long)sizeof(t_torch_rmsoverlap_max), 0L, A_GIMME, 0);

    class_addmethod(c, (method)torch_rmsoverlap_max_dsp64, "dsp64", A_CANT, 0);
    class_addmethod(c, (method)torch_rmsoverlap_max_assist, "assist", A_CANT, 0);
    class_addmethod(c, (method)torch_rmsoverlap_max_dump, "dump", 0);

    class_addmethod(c, (method)torch_rmsoverlap_max_get_window, "get_window", 0);
    class_addmethod(c, (method)torch_rmsoverlap_max_get_window_sum, "get_window_sum", 0);
    class_addmethod(c, (method)torch_rmsoverlap_max_get_norm_vals, "get_norm_vals", 0);

    // Attributes
    CLASS_ATTR_LONG(c, "winsize", 0, t_torch_rmsoverlap_max, x_window_size);
    CLASS_ATTR_ACCESSORS(c, "winsize", NULL, (method)torch_rmsoverlap_max_attr_set_winsize);
    CLASS_ATTR_LABEL(c, "winsize", 0, "Window Size (samples)");
    CLASS_ATTR_SAVE(c, "winsize", 1);

    CLASS_ATTR_LONG(c, "hopsize", 0, t_torch_rmsoverlap_max, x_hop_size);
    CLASS_ATTR_ACCESSORS(c, "hopsize", NULL, (method)torch_rmsoverlap_max_attr_set_hopsize);
    CLASS_ATTR_LABEL(c, "hopsize", 0, "Hop Size (samples)");
    CLASS_ATTR_SAVE(c, "hopsize", 1);

    CLASS_ATTR_SYM(c, "wintype", 0, t_torch_rmsoverlap_max, s_win_type_attr);
    CLASS_ATTR_ACCESSORS(c, "wintype", NULL, (method)torch_rmsoverlap_max_attr_set_wintype);
    CLASS_ATTR_LABEL(c, "wintype", 0, "Window Type (symbol: rect, hann, etc.)");
    CLASS_ATTR_SAVE(c, "wintype", 1);

    CLASS_ATTR_FLOAT(c, "zeropad", 0, t_torch_rmsoverlap_max, x_zero_padding_factor);
    CLASS_ATTR_ACCESSORS(c, "zeropad", NULL, (method)torch_rmsoverlap_max_attr_set_zeropad);
    CLASS_ATTR_LABEL(c, "zeropad", 0, "Zero Padding Factor (0.0 to <1.0)");
    CLASS_ATTR_SAVE(c, "zeropad", 1);

    CLASS_ATTR_SYM(c, "winalign", 0, t_torch_rmsoverlap_max, s_winalign_attr);
    CLASS_ATTR_ACCESSORS(c, "winalign", NULL, (method)torch_rmsoverlap_max_attr_set_winalign);
    CLASS_ATTR_LABEL(c, "winalign", 0, "Window Alignment (symbol: left, center, right)");
    CLASS_ATTR_SAVE(c, "winalign", 1);

    CLASS_ATTR_SYM(c, "normtype", 0, t_torch_rmsoverlap_max, s_normtype_attr);
    CLASS_ATTR_ACCESSORS(c, "normtype", NULL, (method)torch_rmsoverlap_max_attr_set_normtype);
    CLASS_ATTR_LABEL(c, "normtype", 0, "Normalization Type (symbol: win_rms, none, etc.)");
    CLASS_ATTR_SAVE(c, "normtype", 1);

    CLASS_ATTR_FLOAT(c, "normfixedval", 0, t_torch_rmsoverlap_max, x_fixed_norm_multiplier);
    CLASS_ATTR_ACCESSORS(c, "normfixedval", NULL, (method)torch_rmsoverlap_max_attr_set_normfixedval);
    CLASS_ATTR_LABEL(c, "normfixedval", 0, "Fixed Normalization Multiplier (if normtype is 'fixed')");
    CLASS_ATTR_SAVE(c, "normfixedval", 1);

    CLASS_ATTR_LONG(c, "verbose", 0, t_torch_rmsoverlap_max, verbose);
    CLASS_ATTR_LABEL(c, "verbose", 0, "Enable Verbose Logging (0 or 1)");
    CLASS_ATTR_SAVE(c, "verbose", 1);

    class_dspinit(c);
    class_register(CLASS_BOX, c);
    s_torch_rmsoverlap_max_class = c;

    post("torch.rmsoverlap~ (libtorch) loaded");
}

void *torch_rmsoverlap_max_new(t_symbol *s, long argc, t_atom *argv) {
    t_torch_rmsoverlap_max *x = (t_torch_rmsoverlap_max *)object_alloc(s_torch_rmsoverlap_max_class);

    if (x) {
        dsp_setup((t_pxobject *)x, 1);

        // Initialize parameters to defaults
        x->x_window_size = 1024;
        x->x_hop_size = 512;
        x->x_win_type = contorchionist::core::util_windowing::Type::HANN;
        x->s_win_type_attr = gensym("hann");
        x->x_zero_padding_factor = 0.0;
        x->x_win_align = contorchionist::core::util_windowing::Alignment::CENTER;
        x->s_winalign_attr = gensym("center");
        x->x_norm_type = RMSOverlap::NormalizationType::WINDOW_OVERLAP_RMS;
        x->s_normtype_attr = gensym("win_rms");
        x->x_fixed_norm_multiplier = 1.0;
        x->verbose = 0;
        x->analyzer = nullptr;

        // Process attribute arguments
        attr_args_process(x, argc, argv);

        // Create outlets
        x->m_outlet_list = listout((t_object *)x);
        x->m_outlet_rms = outlet_new((t_object *)x, "signal");

        // Create the C++ analyzer object in a try-catch block
        try {
            x->analyzer = new RMSOverlap(
                (int)x->x_window_size,
                (int)x->x_hop_size,
                x->x_win_type,
                (float)x->x_zero_padding_factor,
                x->x_win_align,
                x->x_norm_type,
                (float)x->x_fixed_norm_multiplier,
                64, // Placeholder block size
                (bool)x->verbose
            );
        } catch (const std::exception& e) {
            object_error((t_object *)x, "Failed to create torch.rmsoverlap~ analyzer: %s", e.what());
            // object_free will be called by Max, which will call our _free method
        }
    }
    return (x);
}

void torch_rmsoverlap_max_free(t_torch_rmsoverlap_max *x) {
    dsp_free((t_pxobject *)x);
    if (x->analyzer) {
        delete x->analyzer;
        x->analyzer = nullptr;
    }
}

void torch_rmsoverlap_max_assist(t_torch_rmsoverlap_max *x, void *b, long m, long a, char *s) {
    if (m == ASSIST_INLET) {
        snprintf(s, 256, "Signal Input / Messages");
    } else { // ASSIST_OUTLET
        if (a == 0) {
            snprintf(s, 256, "RMS Signal Output");
        } else if (a == 1) {
            snprintf(s, 256, "List Output (get_window, etc.)");
        }
    }
}

void torch_rmsoverlap_max_dsp64(t_torch_rmsoverlap_max *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags) {
    if (!x->analyzer) {
        object_error((t_object*)x, "torch.rmsoverlap~: analyzer not initialized. Cannot start DSP.");
        return;
    }

    // Update analyzer with actual block size and reset its state
    x->analyzer->setBlockSize(maxvectorsize);
    x->analyzer->reset();

    // Resize temp buffers
    try {
        x->m_input_buffer_float.resize(maxvectorsize);
        x->m_output_buffer_float.resize(maxvectorsize);
    } catch (const std::bad_alloc& e) {
        object_error((t_object*)x, "Failed to allocate temporary buffers: %s", e.what());
        return;
    }

    object_method(dsp64, gensym("dsp_add64"), x, torch_rmsoverlap_max_perform64, 0, NULL);
}

void torch_rmsoverlap_max_perform64(t_torch_rmsoverlap_max *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam) {
    if (!x->analyzer || numins < 1 || numouts < 1) {
        // Output silence if analyzer is not ready or no I/O
        double *out = outs[0];
        for (long i = 0; i < sampleframes; ++i) out[i] = 0.0;
        return;
    }

    double *in_signal = ins[0];
    double *out_signal = outs[0];
    long n = sampleframes;

    // Convert input from double to float
    for (long i = 0; i < n; ++i) {
        x->m_input_buffer_float[i] = static_cast<float>(in_signal[i]);
    }

    // Post data to the analyzer's circular buffer
    x->analyzer->post_input_data(x->m_input_buffer_float.data(), n);

    // Process data from the circular buffer
    bool success = x->analyzer->process(nullptr, x->m_output_buffer_float.data(), n);
    if (!success) {
        if (x->verbose) {
            object_post((t_object*)x, "torch.rmsoverlap~: process() returned false. Outputting silence.");
        }
    }

    // Convert output from float to double
    for (long i = 0; i < n; ++i) {
        out_signal[i] = static_cast<double>(x->m_output_buffer_float[i]);
    }
}

// --- Custom Message Handlers ---
void torch_rmsoverlap_max_dump(t_torch_rmsoverlap_max *x) {
    object_post((t_object *)x, "--- torch.rmsoverlap~ Parameters ---");
    object_post((t_object *)x, "  Window Size: %ld", x->x_window_size);
    object_post((t_object *)x, "  Hop Size: %ld", x->x_hop_size);
    object_post((t_object *)x, "  Window Type: %s", x->s_win_type_attr->s_name);
    object_post((t_object *)x, "  Zero Padding: %.4f", x->x_zero_padding_factor);
    object_post((t_object *)x, "  Window Align: %s", x->s_winalign_attr->s_name);
    object_post((t_object *)x, "  Norm Type: %s", x->s_normtype_attr->s_name);
    object_post((t_object *)x, "  Fixed Norm Val: %.4f", x->x_fixed_norm_multiplier);
    object_post((t_object *)x, "  Verbose: %ld", x->verbose);
    if (x->analyzer) {
        object_post((t_object *)x, "  Analyzer State: Initialized (Block size: %d)", x->analyzer->getBlockSize());
    } else {
        object_post((t_object *)x, "  Analyzer State: Not Initialized");
    }
    object_post((t_object *)x, "------------------------------------");
}

void torch_rmsoverlap_max_get_window(t_torch_rmsoverlap_max *x) {
    if (x->analyzer) {
        const auto& win_tensor = x->analyzer->getWindowFunction();
        if (win_tensor.numel() > 0) {
            long n_elems = win_tensor.numel();
            t_atom* atom_list = (t_atom*)sysmem_newptr(n_elems * sizeof(t_atom));
            if (atom_list) {
                auto tensor_cpu = win_tensor.contiguous().cpu();
                const float* data = tensor_cpu.data_ptr<float>();
                for (long i = 0; i < n_elems; ++i) atom_setfloat(&atom_list[i], data[i]);
                outlet_anything(x->m_outlet_list, gensym("window"), (short)n_elems, atom_list);
                sysmem_freeptr(atom_list);
            }
        }
    }
}

void torch_rmsoverlap_max_get_window_sum(t_torch_rmsoverlap_max *x) {
    if (x->analyzer) {
        const auto& sum_tensor = x->analyzer->getWindowOverlapSum();
        if (sum_tensor.numel() > 0) {
            long n_elems = sum_tensor.numel();
            t_atom* atom_list = (t_atom*)sysmem_newptr(n_elems * sizeof(t_atom));
            if (atom_list) {
                auto tensor_cpu = sum_tensor.contiguous().cpu();
                const float* data = tensor_cpu.data_ptr<float>();
                for (long i = 0; i < n_elems; ++i) atom_setfloat(&atom_list[i], data[i]);
                outlet_anything(x->m_outlet_list, gensym("window_sum"), (short)n_elems, atom_list);
                sysmem_freeptr(atom_list);
            }
        }
    }
}

void torch_rmsoverlap_max_get_norm_vals(t_torch_rmsoverlap_max *x) {
    if (x->analyzer) {
        const auto& norm_tensor = x->analyzer->getNormalizationBuffer();
        if (norm_tensor.numel() > 0) {
            long n_elems = norm_tensor.numel();
            t_atom* atom_list = (t_atom*)sysmem_newptr(n_elems * sizeof(t_atom));
            if (atom_list) {
                auto tensor_cpu = norm_tensor.contiguous().cpu();
                const float* data = tensor_cpu.data_ptr<float>();
                for (long i = 0; i < n_elems; ++i) atom_setfloat(&atom_list[i], data[i]);
                outlet_anything(x->m_outlet_list, gensym("norm_factors"), (short)n_elems, atom_list);
                sysmem_freeptr(atom_list);
            }
        }
    }
}


// --- Attribute Setters ---
t_max_err torch_rmsoverlap_max_attr_set_winsize(t_torch_rmsoverlap_max *x, void *attr, long argc, t_atom *argv) {
    if (argc > 0 && atom_gettype(argv) == A_LONG) {
        long new_size = atom_getlong(argv);
        if (new_size > 0) {
            x->x_window_size = new_size;
            if (x->analyzer) x->analyzer->setWindowSize(new_size);
        }
    }
    return MAX_ERR_NONE;
}

t_max_err torch_rmsoverlap_max_attr_set_hopsize(t_torch_rmsoverlap_max *x, void *attr, long argc, t_atom *argv) {
    if (argc > 0 && atom_gettype(argv) == A_LONG) {
        long new_hop = atom_getlong(argv);
        if (new_hop > 0) {
            x->x_hop_size = new_hop;
            if (x->analyzer) x->analyzer->setHopSize(new_hop);
        }
    }
    return MAX_ERR_NONE;
}

t_max_err torch_rmsoverlap_max_attr_set_wintype(t_torch_rmsoverlap_max *x, void *attr, long argc, t_atom *argv) {
    if (argc > 0 && atom_gettype(argv) == A_SYM) {
        x->s_win_type_attr = atom_getsym(argv);
        x->x_win_type = atom_to_window_type(argv);
        if (x->analyzer) x->analyzer->setWindowType(x->x_win_type);
    }
    return MAX_ERR_NONE;
}

t_max_err torch_rmsoverlap_max_attr_set_winalign(t_torch_rmsoverlap_max *x, void *attr, long argc, t_atom *argv) {
    if (argc > 0 && atom_gettype(argv) == A_SYM) {
        x->s_winalign_attr = atom_getsym(argv);
        x->x_win_align = atom_to_window_alignment(argv);
        if (x->analyzer) x->analyzer->setZeroPadding((float)x->x_zero_padding_factor, x->x_win_align);
    }
    return MAX_ERR_NONE;
}

t_max_err torch_rmsoverlap_max_attr_set_zeropad(t_torch_rmsoverlap_max *x, void *attr, long argc, t_atom *argv) {
    if (argc > 0 && atom_gettype(argv) == A_FLOAT) {
        double factor = atom_getfloat(argv);
        if (factor >= 0.0 && factor < 1.0) {
            x->x_zero_padding_factor = factor;
            if (x->analyzer) x->analyzer->setZeroPadding((float)x->x_zero_padding_factor, x->x_win_align);
        }
    }
    return MAX_ERR_NONE;
}

t_max_err torch_rmsoverlap_max_attr_set_normtype(t_torch_rmsoverlap_max *x, void *attr, long argc, t_atom *argv) {
    if (argc > 0 && atom_gettype(argv) == A_SYM) {
        x->s_normtype_attr = atom_getsym(argv);
        x->x_norm_type = atom_to_normalization_type(argv);
        if (x->analyzer) x->analyzer->setNormalization(x->x_norm_type, (float)x->x_fixed_norm_multiplier);
    }
    return MAX_ERR_NONE;
}

t_max_err torch_rmsoverlap_max_attr_set_normfixedval(t_torch_rmsoverlap_max *x, void *attr, long argc, t_atom *argv) {
    if (argc > 0 && atom_gettype(argv) == A_FLOAT) {
        x->x_fixed_norm_multiplier = atom_getfloat(argv);
        if (x->analyzer && x->x_norm_type == RMSOverlap::NormalizationType::FIXED_MULTIPLIER) {
            x->analyzer->setNormalization(x->x_norm_type, (float)x->x_fixed_norm_multiplier);
        }
    }
    return MAX_ERR_NONE;
}

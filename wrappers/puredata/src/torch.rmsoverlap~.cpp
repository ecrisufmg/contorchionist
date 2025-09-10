#include "m_pd.h"
#include "../../../core/include/core_ap_rmsoverlap.h" // Include the C++ header
#include "../../../core/include/core_util_windowing.h"    
#include "../utils/include/pd_arg_parser.h" // Include the new argument parser
#include <string>
#include <vector> // Required for std::vector used in get_window
#include <torch/torch.h> // Required for torch::Tensor

// Using declarations for easier access to the namespaced classes
using RMSOverlap = contorchionist::core::ap_rmsoverlap::RMSOverlap<float>;

// Helper function to convert string to WindowType
static contorchionist::core::util_windowing::Type string_to_window_type(const char* s) { // Changed to contorchionist::core::util_windowing::Type
    std::string str = s;
    if (str == "rect" || str == "rectangular" || str == "0") return contorchionist::core::util_windowing::Type::RECTANGULAR;
    if (str == "hann" || str == "hanning" || str == "1") return contorchionist::core::util_windowing::Type::HANN; // Corrected to HANN
    if (str == "tri" || str == "triangular" || str == "2") return contorchionist::core::util_windowing::Type::BARTLETT; // Corrected to BARTLETT
    if (str == "hamm" || str == "hamming" || str == "3") return contorchionist::core::util_windowing::Type::HAMMING;
    if (str == "black" || str == "blackman" || str == "4") return contorchionist::core::util_windowing::Type::BLACKMAN;
    if (str == "cos" || str == "cosine" || str == "5") return contorchionist::core::util_windowing::Type::COSINE;
    return contorchionist::core::util_windowing::Type::RECTANGULAR; // Default
}

// Helper function to convert string to WindowAlignment
static contorchionist::core::util_windowing::Alignment string_to_window_alignment(const char* s) { // Changed to contorchionist::core::util_windowing::Alignment
    std::string str = s;
    if (str == "l" || str == "left" || str == "0") return contorchionist::core::util_windowing::Alignment::LEFT;
    if (str == "c" || str == "center" || str == "1") return contorchionist::core::util_windowing::Alignment::CENTER;
    if (str == "r" || str == "right" || str == "2") return contorchionist::core::util_windowing::Alignment::RIGHT;
    return contorchionist::core::util_windowing::Alignment::LEFT; // Default
}

// Helper function to convert string to NormalizationType
static RMSOverlap::NormalizationType string_to_normalization_type(const char* s) {
    std::string str = s;
    if (str == "win_rms" || str == "0") return RMSOverlap::NormalizationType::WINDOW_OVERLAP_RMS;
    if (str == "win_mean" || str == "1") return RMSOverlap::NormalizationType::WINDOW_OVERLAP_MEAN;
    if (str == "win_vals" || str == "2") return RMSOverlap::NormalizationType::WINDOW_OVERLAP_VALS;
    if (str == "overlap_inverse" || str == "overlap" || str == "3") return RMSOverlap::NormalizationType::OVERLAP_INVERSE;
    if (str == "fixed") return RMSOverlap::NormalizationType::FIXED_MULTIPLIER; 
    if (str == "none" || str == "5") return RMSOverlap::NormalizationType::NONE;
    return RMSOverlap::NormalizationType::WINDOW_OVERLAP_RMS; // Default
}

// Define a C-style struct to hold the C++ object
typedef struct _torch_rmsoverlap_tilde_cpp {
    t_object x_obj;
    t_float x_f; // Dummy float for Pure Data, signals come from inlets
    RMSOverlap* analyzer; // Pointer to the C++ RMSOverlap object
    t_outlet *x_out_rms;   // Outlet for RMS values
    t_outlet *x_list_out;  // Outlet for list messages

    // Stored parameters
    int x_window_size;
    int x_hop_size;
    contorchionist::core::util_windowing::Type x_win_type; // Changed to contorchionist::core::util_windowing::Type
    float x_zero_padding_factor;
    contorchionist::core::util_windowing::Alignment x_win_align; // Changed to contorchionist::core::util_windowing::Alignment
    RMSOverlap::NormalizationType x_norm_type;
    float x_fixed_norm_multiplier;
    bool verbose_; // Flag for verbose logging

} t_torch_rmsoverlap_tilde_cpp;

// Declare the class
static t_class *torch_rmsoverlap_tilde_cpp_class;

// --- PD Method Implementations ---

// DSP processing function
static t_int *torch_rmsoverlap_tilde_cpp_perform(t_int *w) {
    t_torch_rmsoverlap_tilde_cpp *x = (t_torch_rmsoverlap_tilde_cpp *)(w[1]);
    t_sample *in = (t_sample *)(w[2]);
    t_sample *out_rms = (t_sample *)(w[3]);
    int n = (int)(w[4]); // Block size

    if (x->analyzer) {
        // First, post the input data to the analyzer's circular buffer
        x->analyzer->post_input_data(in, n);
        // Then, process the data from the circular buffer to produce output
        bool success = x->analyzer->process(nullptr, out_rms, n);

        if (!success) {
            // process() should fill out_rms with 0.0f if it returns false.
            // Add a verbose log if enabled.
            if (x->verbose_) {
                post("torch.rmsoverlap~: process() returned false. Outputting silence. Samples in buffer: %lld, Hop size: %d",
                     x->analyzer->getSamplesInCircularBuffer(),
                     x->analyzer->getHopSize());
            }
            // As a safeguard, ensure output is zero if process didn't (though it should)
            // for(int i = 0; i < n; ++i) { out_rms[i] = 0.0f; }
        }
    } else {
        // Analyzer not initialized, output silence
        for(int i = 0; i < n; ++i) {
            out_rms[i] = 0.0f;
        }
    }
    return (w + 5);
}

// Add object to DSP chain
static void torch_rmsoverlap_tilde_cpp_dsp(t_torch_rmsoverlap_tilde_cpp *x, t_signal **sp) {
    // Check for invalid parameters first
    if (x->x_window_size <= 0 || x->x_hop_size <= 0) {
        pd_error(&x->x_obj, "torch.rmsoverlap~: window size and hop size must be positive. DSP not started.");
        return;
    }

    // Ensure the analyzer object exists before configuring it
    if (x->analyzer) {
        // Update the block size from the signal properties. No memory allocation here.
        x->analyzer->setBlockSize(sp[0]->s_n);

        // Reset the analyzer's internal state to ensure it's clean on DSP toggle.
        x->analyzer->reset();

        if (x->verbose_) {
            post("torch.rmsoverlap~: DSP configured. Block size: %d", sp[0]->s_n);
        }

        // Add the perform routine to the DSP chain
        dsp_add(torch_rmsoverlap_tilde_cpp_perform, 4, x, sp[0]->s_vec, sp[1]->s_vec, sp[0]->s_n);

    } else {
        // This case should ideally not be reached if _new succeeds.
        pd_error(&x->x_obj, "torch.rmsoverlap~: analyzer not initialized. DSP not started.");
    }
}

// Cleanup function when object is destroyed
static void torch_rmsoverlap_tilde_cpp_free(t_torch_rmsoverlap_tilde_cpp *x) {
    if (x->analyzer) {
        delete x->analyzer;
        x->analyzer = nullptr;
    }
    outlet_free(x->x_out_rms);
    outlet_free(x->x_list_out);
}

// Method for setting window type
static void torch_rmsoverlap_tilde_cpp_window(t_torch_rmsoverlap_tilde_cpp *x, t_symbol *s, int argc, t_atom *argv) {
    if (argc > 0 && argv[0].a_type == A_SYMBOL) {
        x->x_win_type = string_to_window_type(atom_getsymbol(&argv[0])->s_name);
        if (x->analyzer) {
            x->analyzer->setWindowType(x->x_win_type);
        }
    } else if (argc > 0 && argv[0].a_type == A_FLOAT) {
        char buf[32]; // Sufficient buffer for floating point numbers
        snprintf(buf, sizeof(buf), "%f", atom_getfloat(&argv[0]));
        x->x_win_type = string_to_window_type(buf);
         if (x->analyzer) {
            x->analyzer->setWindowType(x->x_win_type);
        }
    } else {
        pd_error(x, "torch.rmsoverlap~: window method requires a symbol or number argument (e.g., rect, hann, 0, 1)");
    }
}

// Method for setting window alignment
static void torch_rmsoverlap_tilde_cpp_winalign(t_torch_rmsoverlap_tilde_cpp *x, t_symbol *s, int argc, t_atom *argv) {
    if (argc > 0 && argv[0].a_type == A_SYMBOL) {
        x->x_win_align = string_to_window_alignment(atom_getsymbol(&argv[0])->s_name);
        if (x->analyzer) {
            x->analyzer->setZeroPadding(x->x_zero_padding_factor, x->x_win_align);
        }
    } else if (argc > 0 && argv[0].a_type == A_FLOAT) {
        char buf[32]; // Sufficient buffer for floating point numbers
        snprintf(buf, sizeof(buf), "%f", atom_getfloat(&argv[0]));
        x->x_win_align = string_to_window_alignment(buf);
        if (x->analyzer) {
            x->analyzer->setZeroPadding(x->x_zero_padding_factor, x->x_win_align);
        }
    } else {
        pd_error(x, "torch.rmsoverlap~: winalign method requires a symbol argument (e.g., left, center, right)");
    }
}

// Method for setting zero padding factor
static void torch_rmsoverlap_tilde_cpp_zeropadding(t_torch_rmsoverlap_tilde_cpp *x, t_symbol *s, int argc, t_atom *argv) {
    if (argc > 0 && argv[0].a_type == A_FLOAT) {
        float factor = atom_getfloat(&argv[0]);
        if (factor >= 0.0f && factor < 1.0f) { // Factor should be < 1.0
            x->x_zero_padding_factor = factor;
            if (x->analyzer) {
                x->analyzer->setZeroPadding(x->x_zero_padding_factor, x->x_win_align);
            }
        } else {
            pd_error(x, "torch.rmsoverlap~: zeropadding factor must be between 0.0 and < 1.0");
        }
    } else {
        pd_error(x, "torch.rmsoverlap~: zeropadding method requires a float argument (factor)");
    }
}

// Method for setting normalization type
static void torch_rmsoverlap_tilde_cpp_normalize(t_torch_rmsoverlap_tilde_cpp *x, t_symbol *s, int argc, t_atom *argv) {
    if (argc > 0) {
        pd_utils::ArgParser parser(argc, argv, (t_object*)x); // Use parser for method args too

        if (parser.has_flag("type")) { // Expecting @type <name>
            std::string norm_str = parser.get_string("type", "win_rms");
            x->x_norm_type = string_to_normalization_type(norm_str.c_str());
            
            if (x->x_norm_type == RMSOverlap::NormalizationType::FIXED_MULTIPLIER) {
                x->x_fixed_norm_multiplier = parser.get_float("value", 1.0f); // Expecting @value <float>
            }
        } else if (argc == 1 && argv[0].a_type == A_SYMBOL) { // Legacy: normalize <name>
             const char* norm_str_legacy = atom_getsymbol(&argv[0])->s_name;
             x->x_norm_type = string_to_normalization_type(norm_str_legacy);
             if (x->x_norm_type == RMSOverlap::NormalizationType::FIXED_MULTIPLIER) {
                 pd_error(x, "torch.rmsoverlap~: 'fixed' normalization type via method requires a subsequent float argument or use @type fixed @value <float>.");
                 x->x_fixed_norm_multiplier = 1.0f; // Default if no value provided
             }
        } else if (argc == 2 && argv[0].a_type == A_SYMBOL && std::string(atom_getsymbol(&argv[0])->s_name) == "fixed" && argv[1].a_type == A_FLOAT) { // Legacy: normalize fixed <value>
            x->x_norm_type = RMSOverlap::NormalizationType::FIXED_MULTIPLIER;
            x->x_fixed_norm_multiplier = atom_getfloat(&argv[1]);
        }
         else if (argc == 1 && argv[0].a_type == A_FLOAT) { // Legacy: normalize <value> (implies fixed)
            x->x_norm_type = RMSOverlap::NormalizationType::FIXED_MULTIPLIER;
            x->x_fixed_norm_multiplier = atom_getfloat(&argv[0]);
        }
        else {
             pd_error(x, "torch.rmsoverlap~: normalize method usage: normalize @type <name> [@value <float_if_fixed>], or normalize <name> [<value_if_fixed>]");
             return;
        }

        if (x->analyzer) {
            x->analyzer->setNormalization(x->x_norm_type, x->x_fixed_norm_multiplier);
        }
        if (x->verbose_) {
            post("torch.rmsoverlap~: Normalization set to %s, fixed_val %.2f", RMSOverlap::toString(x->x_norm_type).c_str(), x->x_fixed_norm_multiplier);
        }

    } else {
        // Only post current state if verbose, or if no arguments were given (original behavior for querying)
        if (x->verbose_ || argc == 0) { 
            post("torch.rmsoverlap~: Current normalization: type=%s, fixed_val=%.2f", RMSOverlap::toString(x->x_norm_type).c_str(), x->x_fixed_norm_multiplier);
        }
    }
}

// Method to get window function
static void torch_rmsoverlap_tilde_cpp_get_window(t_torch_rmsoverlap_tilde_cpp *x) {
    if (x->analyzer) {
        const torch::Tensor& win_func_tensor = x->analyzer->getWindowFunction();
        if (win_func_tensor.numel() > 0) {
            long num_elements = win_func_tensor.numel();
            t_atom* atom_list = (t_atom*)getbytes(num_elements * sizeof(t_atom));
            if (atom_list) {
                // Ensure tensor is on CPU and is contiguous
                torch::Tensor tensor_cpu = win_func_tensor.contiguous().cpu();
                const float* tensor_data = tensor_cpu.data_ptr<float>();
                for (long i = 0; i < num_elements; ++i) {
                    SETFLOAT(&atom_list[i], tensor_data[i]);
                }
                outlet_list(x->x_list_out, &s_list, static_cast<int>(num_elements), atom_list);
                freebytes(atom_list, num_elements * sizeof(t_atom));
            } else {
                pd_error(x, "torch.rmsoverlap~: could not allocate memory for window list");
            }
        } else {
             outlet_anything(x->x_list_out, gensym("window"), 0, NULL); // Send empty list with "window" selector
        }
    } else {
        pd_error(x, "torch.rmsoverlap~: analyzer not initialized, cannot get window");
    }
}

// Method to get window overlap sum (for normalization type 2 debugging)
static void torch_rmsoverlap_tilde_cpp_get_window_sum(t_torch_rmsoverlap_tilde_cpp *x) {
    if (x->analyzer) {
        const torch::Tensor& sum_tensor = x->analyzer->getWindowOverlapSum();
        if (sum_tensor.numel() > 0) {
            long num_elements = sum_tensor.numel();
            t_atom* atom_list = (t_atom*)getbytes(num_elements * sizeof(t_atom));
            if (atom_list) {
                torch::Tensor tensor_cpu = sum_tensor.contiguous().cpu();
                const float* tensor_data = tensor_cpu.data_ptr<float>();
                for (long i = 0; i < num_elements; ++i) {
                    SETFLOAT(&atom_list[i], tensor_data[i]);
                }
                outlet_list(x->x_list_out, gensym("window_sum"), static_cast<int>(num_elements), atom_list);
                freebytes(atom_list, num_elements * sizeof(t_atom));
            } else {
                pd_error(x, "torch.rmsoverlap~: could not allocate memory for window_sum list");
            }
        } else {
             outlet_anything(x->x_list_out, gensym("window_sum"), 0, NULL);
        }
    } else {
        pd_error(x, "torch.rmsoverlap~: analyzer not initialized, cannot get window_sum");
    }
}

// Method to get normalization values (factors applied to output)
static void torch_rmsoverlap_tilde_cpp_get_norm_vals(t_torch_rmsoverlap_tilde_cpp *x) {
    if (x->analyzer) {
        const torch::Tensor& norm_tensor = x->analyzer->getNormalizationBuffer();
        if (norm_tensor.numel() > 0) {
            long num_elements = norm_tensor.numel();
            t_atom* atom_list = (t_atom*)getbytes(num_elements * sizeof(t_atom));
            if (atom_list) {
                torch::Tensor tensor_cpu = norm_tensor.contiguous().cpu();
                const float* tensor_data = tensor_cpu.data_ptr<float>();
                for (long i = 0; i < num_elements; ++i) {
                    SETFLOAT(&atom_list[i], tensor_data[i]);
                }
                // Use a different selector for clarity, e.g., "norm_factors"
                outlet_list(x->x_list_out, gensym("norm_factors"), static_cast<int>(num_elements), atom_list);
                freebytes(atom_list, num_elements * sizeof(t_atom));
            } else {
                pd_error(x, "torch.rmsoverlap~: could not allocate memory for norm_factors list");
            }
        } else {
            outlet_anything(x->x_list_out, gensym("norm_factors"), 0, NULL);
        }
    } else {
        pd_error(x, "torch.rmsoverlap~: analyzer not initialized, cannot get norm_factors");
    }
}

// Method to dump current parameters
static void torch_rmsoverlap_tilde_cpp_dump(t_torch_rmsoverlap_tilde_cpp *x) {
    t_atom PDBuffer[7]; // For 7 parameters
    SETSYMBOL(&PDBuffer[0], gensym("winsize"));
    SETFLOAT(&PDBuffer[1], x->x_window_size);
    outlet_list(x->x_list_out, &s_list, 2, PDBuffer);

    SETSYMBOL(&PDBuffer[0], gensym("hopsize"));
    SETFLOAT(&PDBuffer[1], x->x_hop_size);
    outlet_list(x->x_list_out, &s_list, 2, PDBuffer);

    SETSYMBOL(&PDBuffer[0], gensym("window"));
    SETSYMBOL(&PDBuffer[1], gensym(contorchionist::core::util_windowing::torch_window_type_to_string(x->x_win_type).c_str())); // Corrected function call
    outlet_list(x->x_list_out, &s_list, 2, PDBuffer);
    
    SETSYMBOL(&PDBuffer[0], gensym("zeropad"));
    SETFLOAT(&PDBuffer[1], x->x_zero_padding_factor);
    outlet_list(x->x_list_out, &s_list, 2, PDBuffer);

    SETSYMBOL(&PDBuffer[0], gensym("winalign"));
    SETSYMBOL(&PDBuffer[1], gensym(contorchionist::core::util_windowing::torch_window_alignment_to_string(x->x_win_align).c_str())); // Corrected function call
    outlet_list(x->x_list_out, &s_list, 2, PDBuffer);

    SETSYMBOL(&PDBuffer[0], gensym("normtype"));
    SETSYMBOL(&PDBuffer[1], gensym(RMSOverlap::toString(x->x_norm_type).c_str())); // Assuming RMSOverlap::toString exists
    outlet_list(x->x_list_out, &s_list, 2, PDBuffer);

    if (x->x_norm_type == RMSOverlap::NormalizationType::FIXED_MULTIPLIER) {
        SETSYMBOL(&PDBuffer[0], gensym("normval"));
        SETFLOAT(&PDBuffer[1], x->x_fixed_norm_multiplier);
        outlet_list(x->x_list_out, &s_list, 2, PDBuffer);
    }
}

// Constructor function
static void *torch_rmsoverlap_tilde_cpp_new(t_symbol *s, int argc, t_atom *argv) {
    t_torch_rmsoverlap_tilde_cpp *x = (t_torch_rmsoverlap_tilde_cpp *)pd_new(torch_rmsoverlap_tilde_cpp_class);
    if (!x) {
        return nullptr;
    }

    // 1. Initialize all pointers and members to a safe, default state.
    x->analyzer = nullptr;
    x->x_out_rms = nullptr;
    x->x_list_out = nullptr;
    x->x_window_size = 1024;
    x->x_hop_size = 512;
    x->x_win_type = contorchionist::core::util_windowing::Type::HANN;
    x->x_zero_padding_factor = 0.0f;
    x->x_win_align = contorchionist::core::util_windowing::Alignment::CENTER;
    x->x_norm_type = RMSOverlap::NormalizationType::WINDOW_OVERLAP_RMS;
    x->x_fixed_norm_multiplier = 1.0f;
    x->verbose_ = false;

    // 2. Parse arguments using the utility.
    pd_utils::ArgParser parser(argc, argv, (t_object*)x);
    x->verbose_ = parser.has_flag("v") || parser.has_flag("verbose");
    x->x_window_size = static_cast<int>(parser.get_float("winsize", x->x_window_size));
    x->x_hop_size = static_cast<int>(parser.get_float("hopsize", x->x_hop_size));
    if (parser.has_flag("window")) {
        x->x_win_type = string_to_window_type(parser.get_string("window", "hann").c_str());
    }
    x->x_zero_padding_factor = parser.get_float("zeropad", x->x_zero_padding_factor);
    if (parser.has_flag("winalign")) {
        x->x_win_align = string_to_window_alignment(parser.get_string("winalign", "center").c_str());
    }
    if (parser.has_flag("norm")) {
        x->x_norm_type = string_to_normalization_type(parser.get_string("norm", "win_rms").c_str());
        if (x->x_norm_type == RMSOverlap::NormalizationType::FIXED_MULTIPLIER) {
            x->x_fixed_norm_multiplier = parser.get_float("normval", 1.0f);
        }
    }

    // 3. Perform all memory allocations in a single try-catch block for safety.
    try {
        // Validate parameters before allocation
        if (x->x_window_size <= 0) throw std::runtime_error("window size must be positive");
        if (x->x_hop_size <= 0) throw std::runtime_error("hop size must be positive");
        if (x->x_zero_padding_factor < 0.0f || x->x_zero_padding_factor >= 1.0f) {
             throw std::runtime_error("zeropadding factor must be 0.0 <= factor < 1.0");
        }

        // Allocate the C++ object. Use a placeholder block size; _dsp will set the correct one.
        const int placeholder_block_size = 64;
        x->analyzer = new RMSOverlap(
            x->x_window_size, x->x_hop_size, x->x_win_type,
            x->x_zero_padding_factor, x->x_win_align, x->x_norm_type,
            x->x_fixed_norm_multiplier, placeholder_block_size, x->verbose_
        );

        // Allocate Pd resources
        x->x_out_rms = outlet_new(&x->x_obj, &s_signal);
        x->x_list_out = outlet_new(&x->x_obj, &s_list);
        if (!x->x_out_rms || !x->x_list_out) {
            throw std::runtime_error("failed to create outlets");
        }

        if (x->verbose_) {
            post("torch.rmsoverlap~: initialized successfully.");
        }

    } catch (const std::exception& e) {
        pd_error(&x->x_obj, "torch.rmsoverlap~: creation failed: %s", e.what());
        torch_rmsoverlap_tilde_cpp_free(x); // Safely clean up any partial allocations
        return nullptr; // Return null to indicate failure
    }

    return (void *)x;
}

// --- Setup function for PD ---
extern "C" void setup_torch0x2ermsoverlap_tilde(void) {
        torch_rmsoverlap_tilde_cpp_class = class_new(gensym("torch.rmsoverlap~"),
                                           (t_newmethod)torch_rmsoverlap_tilde_cpp_new,
                                           (t_method)torch_rmsoverlap_tilde_cpp_free,
                                           sizeof(t_torch_rmsoverlap_tilde_cpp),
                                           CLASS_DEFAULT,
                                           A_GIMME, // Arguments
                                           0);

        class_addmethod(torch_rmsoverlap_tilde_cpp_class, (t_method)torch_rmsoverlap_tilde_cpp_dsp, gensym("dsp"), A_CANT, 0);
        CLASS_MAINSIGNALIN(torch_rmsoverlap_tilde_cpp_class, t_torch_rmsoverlap_tilde_cpp, x_f);

        // Add methods for parameters
        class_addmethod(torch_rmsoverlap_tilde_cpp_class, (t_method)torch_rmsoverlap_tilde_cpp_window, gensym("window"), A_GIMME, 0);
        class_addmethod(torch_rmsoverlap_tilde_cpp_class, (t_method)torch_rmsoverlap_tilde_cpp_winalign, gensym("winalign"), A_GIMME, 0);
        class_addmethod(torch_rmsoverlap_tilde_cpp_class, (t_method)torch_rmsoverlap_tilde_cpp_zeropadding, gensym("zeropadding"), A_GIMME, 0);
        class_addmethod(torch_rmsoverlap_tilde_cpp_class, (t_method)torch_rmsoverlap_tilde_cpp_normalize, gensym("normalize"), A_GIMME, 0);
        class_addmethod(torch_rmsoverlap_tilde_cpp_class, (t_method)torch_rmsoverlap_tilde_cpp_dump, gensym("dump"), A_NULL, 0);
        
        // Add method for getting window
        class_addmethod(torch_rmsoverlap_tilde_cpp_class, (t_method)torch_rmsoverlap_tilde_cpp_get_window, gensym("get_window"), A_NULL, 0);
        // Add new methods for getting window sum and norm factors
        class_addmethod(torch_rmsoverlap_tilde_cpp_class, (t_method)torch_rmsoverlap_tilde_cpp_get_window_sum, gensym("get_window_sum"), A_NULL, 0);
        class_addmethod(torch_rmsoverlap_tilde_cpp_class, (t_method)torch_rmsoverlap_tilde_cpp_get_norm_vals, gensym("get_norm_vals"), A_NULL, 0);
}

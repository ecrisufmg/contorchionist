extern "C" {
#include "m_pd.h"
}

#include "core_ap_melspectrogram.h"
#include "core_ap_mfcc.h"
#include "core_util_windowing.h"
#include "core_util_normalizations.h"
#include "core_util_conversions.h"

#include "../utils/include/pd_arg_parser.h"
#include "../utils/include/pd_torch_device_adapter.h"
#include "torch/torch.h"
#include <string>
#include <vector>
#include <memory>

static t_class *torch_mfcc_tilde_class;

typedef struct _torch_mfcc_tilde {
    t_object x_obj;
    t_sample x_f;

    std::unique_ptr<contorchionist::core::ap_mfcc::MFCCProcessor<t_sample>> mfcc_processor_;

    t_outlet* out_list_;

    // Parameters
    int p_n_fft_;
    int p_hop_length_;
    int p_win_length_;
    contorchionist::core::util_windowing::Type p_window_type_;
    t_sample p_sample_rate_;
    int p_n_mels_;
    t_sample p_fmin_mel_;
    t_sample p_fmax_mel_;
    contorchionist::core::util_conversions::MelFormulaType p_mel_formula_;
    std::string p_filterbank_norm_;
    contorchionist::core::ap_melspectrogram::MelNormMode p_mel_norm_mode_;
    contorchionist::core::util_normalizations::NormalizationType p_rfft_norm_type_;
    contorchionist::core::util_conversions::SpectrumDataFormat p_output_format_;

    // MFCC params
    int p_n_mfcc_;
    int p_first_mfcc_;
    int p_dct_type_;
    std::string p_dct_norm_;

    // Common params
    torch::Device p_device_;
    bool p_verbose_;

    std::vector<t_atom> atom_list_buffer_;
    std::vector<t_sample> mfcc_output_buffer_;
    bool dsp_is_on_; // Flag to track DSP state

} t_torch_mfcc_tilde;


// Forward declarations
static void torch_mfcc_tilde_configure_processors(t_torch_mfcc_tilde *x);
static void torch_mfcc_tilde_output_frame(t_torch_mfcc_tilde *x, const std::vector<float>& frame_data);

// DSP perform routine
static t_int *torch_mfcc_tilde_perform(t_int *w) {
    t_torch_mfcc_tilde *x = (t_torch_mfcc_tilde *)(w[1]);
    t_sample *in_buf = (t_sample *)(w[2]);
    int n_block_samples = (int)(w[3]);

    // Add safety checks
    if (!x || !x->dsp_is_on_) {
        return (w + 4);
    }

    if (!x->mfcc_processor_ || n_block_samples <= 0) {
        return (w + 4);
    }

    try {
        if (x->mfcc_processor_->process(in_buf, n_block_samples, x->mfcc_output_buffer_)) {
            if (!x->mfcc_output_buffer_.empty()) {
                torch_mfcc_tilde_output_frame(x, x->mfcc_output_buffer_);
            }
        }
    } catch (const std::exception& e) {
        if (x->p_verbose_) {
            pd_error(x, "torch.mfcc~: Error in perform routine: %s", e.what());
        }
    } catch (...) {
        if (x->p_verbose_) {
            pd_error(x, "torch.mfcc~: Unknown error in perform routine");
        }
    }
    
    return (w + 4);
}

static void torch_mfcc_tilde_output_frame(t_torch_mfcc_tilde *x, const std::vector<t_sample>& frame_data) {
    size_t num_elements = frame_data.size();
    if (num_elements == 0) return;

    if (x->atom_list_buffer_.size() < num_elements) {
        x->atom_list_buffer_.resize(num_elements);
    }

    for (size_t i = 0; i < num_elements; ++i) {
        SETFLOAT(&x->atom_list_buffer_[i], static_cast<float>(frame_data[i]));
    }
    outlet_list(x->out_list_, &s_list, static_cast<int>(num_elements), x->atom_list_buffer_.data());
}

// DSP add method
static void torch_mfcc_tilde_dsp(t_torch_mfcc_tilde *x, t_signal **sp) {
    if (!x) return;
    
    x->dsp_is_on_ = sp[0]->s_n > 0;
    post("torch.mfcc~: DSP state changed. Is on: %s", x->dsp_is_on_ ? "true" : "false");

    try {
        if (x->dsp_is_on_) {
            t_sample current_sr = sp[0]->s_sr;
            if (x->p_sample_rate_ != current_sr && current_sr > 0) {
                x->p_sample_rate_ = current_sr;
                torch_mfcc_tilde_configure_processors(x);
            } else if (!x->mfcc_processor_ && x->p_sample_rate_ > 0) {
                torch_mfcc_tilde_configure_processors(x);
            }

            if (x->mfcc_processor_) {
                dsp_add(torch_mfcc_tilde_perform, 3, x, sp[0]->s_vec, (t_int)sp[0]->s_n);
                if (x->p_verbose_) {
                    post("torch.mfcc~: DSP chain added. SR: %.f", static_cast<float>(x->p_sample_rate_));
                }
            } else if (x->p_verbose_) {
                post("torch.mfcc~: Processor not configured. DSP chain not added.");
            }
        }
    } catch (const std::exception& e) {
        pd_error(x, "torch.mfcc~: Error in DSP method: %s", e.what());
    } catch (...) {
        pd_error(x, "torch.mfcc~: Unknown error in DSP method");
    }
}

static void torch_mfcc_tilde_configure_processors(t_torch_mfcc_tilde *x) {
    if (x->p_n_fft_ <= 0 || x->p_hop_length_ <= 0 || x->p_sample_rate_ <= 0) {
        if (x->p_verbose_)
            pd_error(x, "torch.mfcc~: n_fft, hop_length, and sample_rate must be positive. Processor not configured.");
        x->mfcc_processor_.reset();
        return;
    }

    try {
        if (x->p_verbose_) {
            post("torch.mfcc~: Configuring MFCC processor...");
        }

        x->mfcc_processor_ = std::make_unique<contorchionist::core::ap_mfcc::MFCCProcessor<t_sample>>(
            x->p_n_mfcc_,
            x->p_first_mfcc_,
            x->p_dct_type_,
            x->p_dct_norm_,
            x->p_n_fft_,
            x->p_hop_length_,
            x->p_win_length_,
            x->p_window_type_,
            x->p_rfft_norm_type_, // Use the RFFT norm type variable
            x->p_output_format_,
            x->p_sample_rate_,
            x->p_n_mels_,
            x->p_fmin_mel_,
            x->p_fmax_mel_,
            x->p_mel_formula_,
            x->p_filterbank_norm_,
            x->p_mel_norm_mode_,
            x->p_device_,
            x->p_verbose_
        );

         if (x->p_verbose_) {
            post("torch.mfcc~: MFCC processor configured successfully.");
        }
    } catch (const std::exception& e) {
        pd_error(x, "torch.mfcc~: Error configuring processor: %s", e.what());
        x->mfcc_processor_.reset();
    }
}

// Methods for messages from Pd
static void torch_mfcc_tilde_n_mfcc(t_torch_mfcc_tilde *x, t_floatarg val) {
    int new_val = static_cast<int>(val);
    if (new_val <= 0) { pd_error(x, "n_mfcc must be positive"); return; }
    if (x->p_n_mfcc_ != new_val) {
        x->p_n_mfcc_ = new_val;
        x->atom_list_buffer_.resize(x->p_n_mfcc_);
        x->mfcc_output_buffer_.resize(x->p_n_mfcc_);
        if (x->mfcc_processor_) x->mfcc_processor_->set_n_mfcc(x->p_n_mfcc_);
        else torch_mfcc_tilde_configure_processors(x);
    }
}

static void torch_mfcc_tilde_first_mfcc(t_torch_mfcc_tilde *x, t_floatarg val) {
    int new_val = static_cast<int>(val);
    if (new_val < 0) { pd_error(x, "first_mfcc must be non-negative"); return; }
    if (x->p_first_mfcc_ != new_val) {
        x->p_first_mfcc_ = new_val;
        if (x->mfcc_processor_) x->mfcc_processor_->set_first_mfcc(x->p_first_mfcc_);
        else torch_mfcc_tilde_configure_processors(x);
    }
}

static void torch_mfcc_tilde_n_mels(t_torch_mfcc_tilde *x, t_floatarg val) {
    int new_val = static_cast<int>(val);
    if (new_val <= 0) { pd_error(x, "n_mels must be positive"); return; }
    if (x->p_n_mels_ != new_val) {
        x->p_n_mels_ = new_val;
        torch_mfcc_tilde_configure_processors(x); // Reconfigure both
    }
}

static void torch_mfcc_tilde_n_fft(t_torch_mfcc_tilde *x, t_floatarg val) {
    int new_val = static_cast<int>(val);
    if (new_val <= 0) { pd_error(x, "n_fft must be positive"); return; }
    if (x->p_n_fft_ != new_val) {
        x->p_n_fft_ = new_val;
        torch_mfcc_tilde_configure_processors(x);
    }
}

// New functions for unit and norm messages
static void torch_mfcc_tilde_unit(t_torch_mfcc_tilde *x, t_symbol *s) {
    try {
        contorchionist::core::util_conversions::SpectrumDataFormat new_format =
            contorchionist::core::util_conversions::string_to_spectrum_data_format(s->s_name);
        if (x->p_output_format_ != new_format) {
            x->p_output_format_ = new_format;
            if (x->mfcc_processor_) {
                x->mfcc_processor_->set_output_unit(x->p_output_format_);
                if (x->p_verbose_) {
                    post("torch.mfcc~: Output unit set to: %s", s->s_name);
                }
            } else {
                if (x->p_verbose_) {
                    post("torch.mfcc~: Output unit set to: %s (processor not yet configured)", s->s_name);
                }
            }
        }
    } catch (const std::invalid_argument& e) {
        pd_error(x, "torch.mfcc~: Invalid unit type '%s': %s", s->s_name, e.what());
    }
}

static void torch_mfcc_tilde_melformula(t_torch_mfcc_tilde *x, t_symbol *s) {
    try {
        contorchionist::core::util_conversions::MelFormulaType new_formula =
            contorchionist::core::util_conversions::string_to_mel_formula_type(s->s_name);
        if (x->p_mel_formula_ != new_formula) {
            x->p_mel_formula_ = new_formula;
            if (x->mfcc_processor_) {
                x->mfcc_processor_->set_mel_formula(x->p_mel_formula_);
                if (x->p_verbose_) {
                    post("torch.mfcc~: Mel formula set to: %s", s->s_name);
                }
            } else {
                if (x->p_verbose_) {
                    post("torch.mfcc~: Mel formula set to: %s (processor not yet configured)", s->s_name);
                }
            }
        }
    } catch (const std::invalid_argument& e) {
        pd_error(x, "torch.mfcc~: Invalid mel formula '%s': %s", s->s_name, e.what());
    }
}

// ... other setters for mel params would go here, they all call configure_processors ...

static void torch_mfcc_tilde_norm(t_torch_mfcc_tilde *x, t_symbol *s) {
    try {
        contorchionist::core::util_normalizations::NormalizationType new_norm =
            contorchionist::core::util_normalizations::string_to_normalization_type(s->s_name, false); // false for forward op
        if (x->p_rfft_norm_type_ != new_norm) {
            x->p_rfft_norm_type_ = new_norm;
            // Since RFFT norm affects the entire pipeline, we need to reconfigure
            if (x->mfcc_processor_) {
                torch_mfcc_tilde_configure_processors(x);
                if (x->p_verbose_) {
                    post("torch.mfcc~: RFFT normalization set to: %s (processor reconfigured)", s->s_name);
                }
            } else {
                if (x->p_verbose_) {
                    post("torch.mfcc~: RFFT normalization set to: %s (processor not yet configured)", s->s_name);
                }
            }
        }
    } catch (const std::invalid_argument& e) {
        pd_error(x, "torch.mfcc~: Invalid RFFT normalization '%s': %s", s->s_name, e.what());
    }
}

static void torch_mfcc_tilde_verbose(t_torch_mfcc_tilde *x, t_floatarg val) {
    bool new_verbose = static_cast<bool>(val);
    if (x->p_verbose_ != new_verbose) {
        x->p_verbose_ = new_verbose;
        if (x->mfcc_processor_) x->mfcc_processor_->set_verbose(x->p_verbose_);
        post("torch.mfcc~: Verbose mode %s.", x->p_verbose_ ? "enabled" : "disabled");
    }
}

// Constructor
void *torch_mfcc_tilde_new(t_symbol *s_dummy, int argc, t_atom *argv) {
    t_torch_mfcc_tilde *x = (t_torch_mfcc_tilde *)pd_new(torch_mfcc_tilde_class);

    // Default parameters
    x->p_n_fft_ = 2048;
    x->p_hop_length_ = 512;
    x->p_win_length_ = -1;
    x->p_window_type_ = contorchionist::core::util_windowing::Type::HANN;
    x->p_sample_rate_ = sys_getsr();
    x->p_n_mels_ = 40;
    x->p_fmin_mel_ = 20.0f;
    x->p_fmax_mel_ = 20000.0f;
    x->p_mel_formula_ = contorchionist::core::util_conversions::MelFormulaType::SLANEY;
    x->p_filterbank_norm_ = "slaney";
    x->p_mel_norm_mode_ = contorchionist::core::ap_melspectrogram::MelNormMode::NONE;
    x->p_rfft_norm_type_ = contorchionist::core::util_normalizations::NormalizationType::NONE;
    x->p_output_format_ = contorchionist::core::util_conversions::SpectrumDataFormat::POWERPHASE;
    x->p_n_mfcc_ = 13;
    x->p_first_mfcc_ = 0;
    x->p_dct_type_ = 2;
    x->p_dct_norm_ = "ortho";
    x->p_device_ = torch::kCPU;
    x->p_verbose_ = false;
    x->mfcc_processor_ = nullptr;
    x->dsp_is_on_ = false;

    // Parse arguments
    pd_utils::ArgParser parser(argc, argv, &x->x_obj);
    x->p_verbose_ = parser.has_flag("verbose v");
    x->p_n_mfcc_ = static_cast<int>(parser.get_float("n_mfcc nmfcc", x->p_n_mfcc_));
    x->p_first_mfcc_ = static_cast<int>(parser.get_float("first_mfcc firstmfcc", x->p_first_mfcc_));
    x->p_n_mels_ = static_cast<int>(parser.get_float("n_mels M", x->p_n_mels_));
    x->p_n_fft_ = static_cast<int>(parser.get_float("n_fft N", x->p_n_fft_));
    x->p_hop_length_ = static_cast<int>(parser.get_float("hop H", x->p_hop_length_));
    // Add more parsers for other mel params if needed...

    x->out_list_ = outlet_new(&x->x_obj, &s_list);
    x->atom_list_buffer_.resize(x->p_n_mfcc_);
    x->mfcc_output_buffer_.resize(x->p_n_mfcc_);

    if (x->p_sample_rate_ > 0) {
        torch_mfcc_tilde_configure_processors(x);
    }

    return (void *)x;
}

void torch_mfcc_tilde_free(t_torch_mfcc_tilde *x) {
    post("torch.mfcc~: Freeing object and its resources.");
    // The unique_ptr mfcc_processor_ will be automatically destroyed,
    // calling the destructor of MFCCProcessor safely.
}

extern "C" void setup_torch0x2emfcc_tilde(void) {
    torch_mfcc_tilde_class = class_new(
        gensym("torch.mfcc~"),
        (t_newmethod)torch_mfcc_tilde_new,
        (t_method)torch_mfcc_tilde_free,
        sizeof(t_torch_mfcc_tilde),
        CLASS_DEFAULT, A_GIMME, 0);

    CLASS_MAINSIGNALIN(torch_mfcc_tilde_class, t_torch_mfcc_tilde, x_f);
    class_addmethod(torch_mfcc_tilde_class, (t_method)torch_mfcc_tilde_dsp, gensym("dsp"), A_CANT, 0);

    // Parameter setting methods
    class_addmethod(torch_mfcc_tilde_class, (t_method)torch_mfcc_tilde_n_mfcc, gensym("n_mfcc"), A_FLOAT, 0);
    class_addmethod(torch_mfcc_tilde_class, (t_method)torch_mfcc_tilde_first_mfcc, gensym("first_mfcc"), A_FLOAT, 0);
    class_addmethod(torch_mfcc_tilde_class, (t_method)torch_mfcc_tilde_n_mels, gensym("n_mels"), A_FLOAT, 0);
    class_addmethod(torch_mfcc_tilde_class, (t_method)torch_mfcc_tilde_n_fft, gensym("n_fft"), A_FLOAT, 0);
    class_addmethod(torch_mfcc_tilde_class, (t_method)torch_mfcc_tilde_unit, gensym("unit"), A_SYMBOL, 0);
    class_addmethod(torch_mfcc_tilde_class, (t_method)torch_mfcc_tilde_norm, gensym("norm"), A_SYMBOL, 0);
    class_addmethod(torch_mfcc_tilde_class, (t_method)torch_mfcc_tilde_melformula, gensym("melformula"), A_SYMBOL, 0);
    class_addmethod(torch_mfcc_tilde_class, (t_method)torch_mfcc_tilde_melformula, gensym("melnorm"), A_SYMBOL, 0);
    class_addmethod(torch_mfcc_tilde_class, (t_method)torch_mfcc_tilde_verbose, gensym("verbose"), A_FLOAT, 0);

    class_sethelpsymbol(torch_mfcc_tilde_class, gensym("torch.mfcc~"));
    post("torch.mfcc~ v0.1 - MFCC analysis for Pure Data");
}

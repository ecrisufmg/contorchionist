extern "C" {
#include "m_pd.h"
}

#include "../../../core/include/core_ap_melspectrogram.h"
#include "../utils/include/pd_arg_parser.h"
#include "../utils/include/pd_torch_device_adapter.h"
#include "torch/torch.h"
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

static t_class *torch_melspectrogram_tilde_class;

typedef struct _torch_melspectrogram_tilde {
    t_object x_obj;
    t_sample x_f;

    std::unique_ptr<contorchionist::core::ap_melspectrogram::MelSpectrogramProcessor<float>> processor_;
    t_outlet* out_list_;

    // Parameters stored in the wrapper
    int p_n_fft_;
    int p_hop_length_;
    int p_win_length_;
    contorchionist::core::util_windowing::Type p_window_type_;
    contorchionist::core::util_normalizations::NormalizationType p_rfft_norm_type_;
    contorchionist::core::util_conversions::SpectrumDataFormat p_output_format_;
    float p_sample_rate_;
    int p_n_mels_;
    float p_fmin_mel_;
    float p_fmax_mel_;
    contorchionist::core::util_conversions::MelFormulaType p_mel_formula_;
    std::string p_filterbank_norm_;
    contorchionist::core::ap_melspectrogram::MelNormMode p_mel_norm_mode_;
    torch::Device p_device_;
    bool p_verbose_;

} t_torch_melspectrogram_tilde;


// Forward declarations
static void torch_melspectrogram_tilde_configure_processor(t_torch_melspectrogram_tilde *x);
static void torch_melspectrogram_tilde_output_frame(t_torch_melspectrogram_tilde *x, const std::vector<float>& frame_data);

// DSP perform routine
static t_int *torch_melspectrogram_tilde_perform(t_int *w) {
    t_torch_melspectrogram_tilde *x = (t_torch_melspectrogram_tilde *)(w[1]);
    t_sample *in_buf = (t_sample *)(w[2]);
    int n_block_samples = (int)(w[3]);

    if (!x->processor_ || n_block_samples <= 0) {
        return (w + 4);
    }

    std::vector<float> frame_data1, frame_data2;

    if (x->processor_->process(in_buf, n_block_samples, frame_data1, frame_data2)) {
        if (!frame_data1.empty()) {
            torch_melspectrogram_tilde_output_frame(x, frame_data1);
        }
    }
    return (w + 4);
}

// Helper to output a spectral frame (list) to the primary outlet
static void torch_melspectrogram_tilde_output_frame(t_torch_melspectrogram_tilde *x, const std::vector<float>& frame_data) {
    size_t num_elements = frame_data.size();
    if (num_elements == 0) return;

    t_atom* atom_list = (t_atom*)getbytes(num_elements * sizeof(t_atom));
    if (!atom_list) {
        pd_error(x, "torch.melspectrogram~: Could not allocate memory for output list.");
        return;
    }

    for (size_t i = 0; i < num_elements; ++i) {
        SETFLOAT(&atom_list[i], frame_data[i]);
    }
    outlet_list(x->out_list_, &s_list, static_cast<int>(num_elements), atom_list);
    freebytes(atom_list, num_elements * sizeof(t_atom));
}

// DSP add method
static void torch_melspectrogram_tilde_dsp(t_torch_melspectrogram_tilde *x, t_signal **sp) {
    float current_sr = sp[0]->s_sr;
    bool sr_changed = (x->p_sample_rate_ != current_sr && current_sr > 0);

    if (sr_changed) {
        x->p_sample_rate_ = current_sr;
        if (x->processor_) {
            x->processor_->set_sample_rate(x->p_sample_rate_);
            if (x->p_verbose_) post("torch.melspectrogram~: Sample rate updated to %.f Hz.", x->p_sample_rate_);
        } else {
            torch_melspectrogram_tilde_configure_processor(x);
        }
    } else if (!x->processor_ && x->p_sample_rate_ > 0) {
        torch_melspectrogram_tilde_configure_processor(x);
    }

    if (x->processor_) {
        dsp_add(torch_melspectrogram_tilde_perform, 3, x, sp[0]->s_vec, (t_int)sp[0]->s_n);
        if (x->p_verbose_) {
            post("torch.melspectrogram~: DSP chain %s. SR: %.f", x->processor_ ? "active" : "inactive (processor not ready)", x->p_sample_rate_);
        }
    } else if (x->p_verbose_) {
        post("torch.melspectrogram~: Processor not configured. DSP chain not added. Ensure SR is valid.");
    }
}

// Method for -melmode
static void torch_melspectrogram_tilde_melmode(t_torch_melspectrogram_tilde *x, t_symbol *s) {
    try {
        auto new_mode = contorchionist::core::ap_melspectrogram::string_to_mel_norm_mode(s->s_name);
        if (x->p_mel_norm_mode_ != new_mode) {
            x->p_mel_norm_mode_ = new_mode;
            if (x->processor_) {
                x->processor_->set_mel_norm_mode(x->p_mel_norm_mode_);
            }
            if (x->p_verbose_) post("torch.melspectrogram~: Mel mode set to %s", s->s_name);
        }
    } catch (const std::invalid_argument& e) {
        pd_error(x, "torch.melspectrogram~: Invalid melmode '%s': %s", s->s_name, e.what());
    }
}

// (Re)create and configure the MelSpectrogramProcessor
static void torch_melspectrogram_tilde_configure_processor(t_torch_melspectrogram_tilde *x) {
    if (x->p_n_fft_ <= 0 || x->p_hop_length_ <= 0) {
        if (x->p_verbose_ || x->processor_)
            pd_error(x, "torch.melspectrogram~: n_fft and hop_length must be positive. Processor not configured.");
        x->processor_.reset();
        return;
    }
    if (x->p_sample_rate_ <= 0) {
         if (x->p_verbose_ || x->processor_)
            pd_error(x, "torch.melspectrogram~: Sample rate must be positive (current: %.f). Processor not configured.", x->p_sample_rate_);
        x->processor_.reset();
        return;
    }

    try {
        if (x->p_verbose_) {
            post("torch.melspectrogram~: Configuring processor: n_fft=%d, hop=%d, win_len=%d, win_type=%s, rfft_norm=%s, output_unit=%s, sr=%.f, n_mels=%d, fmin=%.1f, fmax=%.1f, mel_formula=%s, filterbank_norm=%s, melmode=%s, device=%s",
                 x->p_n_fft_, x->p_hop_length_, x->p_win_length_,
                 contorchionist::core::util_windowing::torch_window_type_to_string(x->p_window_type_).c_str(),
                 contorchionist::core::util_normalizations::normalization_type_to_string(x->p_rfft_norm_type_).c_str(),
                 contorchionist::core::util_conversions::spectrum_data_format_to_string(x->p_output_format_).c_str(),
                 x->p_sample_rate_, x->p_n_mels_, x->p_fmin_mel_, x->p_fmax_mel_,
                 contorchionist::core::util_conversions::mel_formula_type_to_string(x->p_mel_formula_).c_str(),
                 x->p_filterbank_norm_.c_str(),
                 contorchionist::core::ap_melspectrogram::mel_norm_mode_to_string(x->p_mel_norm_mode_).c_str(),
                 x->p_device_.str().c_str());
        }

        x->processor_ = std::make_unique<contorchionist::core::ap_melspectrogram::MelSpectrogramProcessor<float>>(
            x->p_n_fft_,
            x->p_hop_length_,
            x->p_win_length_,
            x->p_window_type_,
            x->p_rfft_norm_type_,
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
            post("torch.melspectrogram~: Processor configured successfully.");
        }
    } catch (const std::exception& e) {
        pd_error(x, "torch.melspectrogram~: Error configuring processor: %s", e.what());
        x->processor_.reset();
    }
}


// Methods for messages from Pd
static void torch_melspectrogram_tilde_n_fft(t_torch_melspectrogram_tilde *x, t_floatarg val) {
    int new_val = static_cast<int>(val);
    if (new_val <= 0) { pd_error(x, "n_fft must be positive"); return; }
    if (x->p_n_fft_ != new_val) {
        x->p_n_fft_ = new_val;
        if (x->processor_) x->processor_->set_n_fft(x->p_n_fft_);
        else torch_melspectrogram_tilde_configure_processor(x);
    }
}

static void torch_melspectrogram_tilde_hop_length(t_torch_melspectrogram_tilde *x, t_floatarg val) {
    int new_val = static_cast<int>(val);
    if (new_val <= 0) { pd_error(x, "hop_length must be positive"); return; }
    if (x->p_hop_length_ != new_val) {
        x->p_hop_length_ = new_val;
        if (x->processor_) x->processor_->set_hop_length(x->p_hop_length_);
        else torch_melspectrogram_tilde_configure_processor(x);
    }
}

static void torch_melspectrogram_tilde_win_length(t_torch_melspectrogram_tilde *x, t_floatarg val) {
    int new_val = static_cast<int>(val);
    if (x->p_win_length_ != new_val) {
        x->p_win_length_ = new_val;
        if (x->processor_) x->processor_->set_win_length(x->p_win_length_);
        else torch_melspectrogram_tilde_configure_processor(x);
    }
}

static void torch_melspectrogram_tilde_window(t_torch_melspectrogram_tilde *x, t_symbol *s) {
    try {
        auto new_type = contorchionist::core::util_windowing::string_to_torch_window_type(s->s_name);
        if (x->p_window_type_ != new_type) {
            x->p_window_type_ = new_type;
            if (x->processor_) x->processor_->set_window_type(x->p_window_type_);
            else torch_melspectrogram_tilde_configure_processor(x);
        }
    } catch (const std::invalid_argument& e) {
        pd_error(x, "torch.melspectrogram~: Invalid window type '%s': %s", s->s_name, e.what());
    }
}

static void torch_melspectrogram_tilde_norm(t_torch_melspectrogram_tilde *x, t_symbol *s) {
    try {
        auto new_type = contorchionist::core::util_normalizations::string_to_normalization_type(s->s_name, false);
        if (x->p_rfft_norm_type_ != new_type) {
            x->p_rfft_norm_type_ = new_type;
            if (x->processor_) x->processor_->set_normalization_type(x->p_rfft_norm_type_);
            else torch_melspectrogram_tilde_configure_processor(x);
        }
    } catch (const std::invalid_argument& e) {
        pd_error(x, "torch.melspectrogram~: Invalid RFFT normalization type '%s': %s", s->s_name, e.what());
    }
}

static void torch_melspectrogram_tilde_n_mels(t_torch_melspectrogram_tilde *x, t_floatarg val) {
    int new_val = static_cast<int>(val);
    if (new_val <= 0) { pd_error(x, "n_mels must be positive"); return; }
    if (x->p_n_mels_ != new_val) {
        x->p_n_mels_ = new_val;
        if (x->processor_) x->processor_->set_n_mels(x->p_n_mels_);
        else torch_melspectrogram_tilde_configure_processor(x);
    }
}

static void torch_melspectrogram_tilde_fmin_mel(t_torch_melspectrogram_tilde *x, t_floatarg val) {
    if (val < 0) { pd_error(x, "fmin_mel must be non-negative"); return; }
    if (x->p_fmin_mel_ != val) {
        x->p_fmin_mel_ = val;
        if (x->processor_) x->processor_->set_fmin_mel(x->p_fmin_mel_);
        else torch_melspectrogram_tilde_configure_processor(x);
    }
}

static void torch_melspectrogram_tilde_fmax_mel(t_torch_melspectrogram_tilde *x, t_floatarg val) {
    if (x->p_fmax_mel_ != val) {
        x->p_fmax_mel_ = val;
        if (x->processor_) x->processor_->set_fmax_mel(x->p_fmax_mel_);
        else torch_melspectrogram_tilde_configure_processor(x);
    }
}

static void torch_melspectrogram_tilde_melcalc(t_torch_melspectrogram_tilde *x, t_symbol *s) {
    try {
        auto new_formula = contorchionist::core::util_conversions::string_to_mel_formula_type(s->s_name);
        if (x->p_mel_formula_ != new_formula) {
            x->p_mel_formula_ = new_formula;
            if (x->processor_) {
                x->processor_->set_mel_formula(x->p_mel_formula_);
            } else {
                torch_melspectrogram_tilde_configure_processor(x);
            }
            if (x->p_verbose_) post("torch.melspectrogram~: Mel formula set to %s", s->s_name);
        }
    } catch (const std::invalid_argument& e) {
        pd_error(x, "torch.melspectrogram~: Invalid mel_formula '%s': %s", s->s_name, e.what());
    }
}

static void torch_melspectrogram_tilde_filterbank_norm(t_torch_melspectrogram_tilde *x, t_symbol *s) {
    std::string new_norm_str = s->s_name;
    if (x->p_filterbank_norm_ != new_norm_str) {
        x->p_filterbank_norm_ = new_norm_str;
        if (x->processor_) {
            x->processor_->set_filterbank_norm(x->p_filterbank_norm_);
        } else {
            torch_melspectrogram_tilde_configure_processor(x);
        }
        if (x->p_verbose_) post("torch.melspectrogram~: Filterbank norm set to %s", x->p_filterbank_norm_.c_str());
    }
}

static void torch_melspectrogram_tilde_device(t_torch_melspectrogram_tilde *x, t_symbol *s) {
    auto device_result = get_device_from_string(s->s_name);
    if (device_result.second) {
        if (x->p_device_ != device_result.first) {
            x->p_device_ = device_result.first;
            if (x->processor_) x->processor_->set_device(x->p_device_);
            else torch_melspectrogram_tilde_configure_processor(x);
        }
    } else {
        pd_error(x, "torch.melspectrogram~: Invalid device string '%s'", s->s_name);
    }
}

static void torch_melspectrogram_tilde_verbose(t_torch_melspectrogram_tilde *x, t_floatarg val) {
    bool new_verbose = static_cast<bool>(val);
    if (x->p_verbose_ != new_verbose) {
        x->p_verbose_ = new_verbose;
        if (x->processor_) x->processor_->set_verbose(x->p_verbose_);
        post("torch.melspectrogram~: Verbose mode %s.", x->p_verbose_ ? "enabled" : "disabled");
    }
}

static void torch_melspectrogram_tilde_unit(t_torch_melspectrogram_tilde *x, t_symbol *s) {
    try {
        auto new_format = contorchionist::core::util_conversions::string_to_spectrum_data_format(s->s_name);
        if (x->p_output_format_ != new_format) {
            x->p_output_format_ = new_format;
            if (x->processor_) {
                x->processor_->set_output_format(x->p_output_format_);
                 if (x->p_verbose_) post("torch.melspectrogram~: Output unit set to %s", s->s_name);
            }
        }
    } catch (const std::invalid_argument& e) {
        pd_error(x, "torch.melspectrogram~: Invalid unit type '%s': %s", s->s_name, e.what());
    }
}

// Constructor
void *torch_melspectrogram_tilde_new(t_symbol *s_dummy, int argc, t_atom *argv) {
    t_torch_melspectrogram_tilde *x = (t_torch_melspectrogram_tilde *)pd_new(torch_melspectrogram_tilde_class);

    // Initialize parameter defaults
    x->p_n_fft_ = 2048;
    x->p_hop_length_ = 512;
    x->p_win_length_ = -1;
    x->p_window_type_ = contorchionist::core::util_windowing::Type::HANN;
    x->p_rfft_norm_type_ = contorchionist::core::util_normalizations::NormalizationType::N_FFT;
    x->p_output_format_ = contorchionist::core::util_conversions::SpectrumDataFormat::MAGPHASE;
    x->p_sample_rate_ = sys_getsr();
    x->p_n_mels_ = 128;
    x->p_fmin_mel_ = 0.0f;
    x->p_fmax_mel_ = -1.0f;
    x->p_mel_formula_ = contorchionist::core::util_conversions::MelFormulaType::CALC2;
    x->p_filterbank_norm_ = "slaney";
    x->p_mel_norm_mode_ = contorchionist::core::ap_melspectrogram::MelNormMode::ENERGY_POWER;
    x->p_device_ = torch::kCPU;
    x->p_verbose_ = false;
    x->processor_ = nullptr;

    // Parse arguments
    pd_utils::ArgParser parser(argc, argv, &x->x_obj);
    x->p_verbose_ = parser.has_flag("verbose v");

    x->p_n_fft_ = static_cast<int>(parser.get_float("n_fft N", x->p_n_fft_));
    x->p_hop_length_ = static_cast<int>(parser.get_float("hop H", x->p_hop_length_));
    x->p_win_length_ = static_cast<int>(parser.get_float("win_length W", x->p_win_length_));

    if (parser.has_flag("window w")) {
        try {
            x->p_window_type_ = contorchionist::core::util_windowing::string_to_torch_window_type(parser.get_string("window w"));
        } catch (const std::invalid_argument&) {
            pd_error(x, "torch.melspectrogram~: Invalid window type '%s' in args. Using default.", parser.get_string("window w").c_str());
        }
    }
    if (parser.has_flag("norm n")) {
        try {
            x->p_rfft_norm_type_ = contorchionist::core::util_normalizations::string_to_normalization_type(parser.get_string("norm n"), false);
        } catch (const std::invalid_argument&) {
             pd_error(x, "torch.melspectrogram~: Invalid RFFT norm type '%s' in args. Using default.", parser.get_string("norm n").c_str());
        }
    }

    x->p_n_mels_ = static_cast<int>(parser.get_float("n_mels M", x->p_n_mels_));
    x->p_fmin_mel_ = parser.get_float("fmin FMIN", x->p_fmin_mel_);
    x->p_fmax_mel_ = parser.get_float("fmax FMAX", x->p_fmax_mel_);

    std::string mel_formula_str_default = contorchionist::core::util_conversions::mel_formula_type_to_string(x->p_mel_formula_);
    if (parser.has_flag("melcalc mc mel_calc")) {
        try {
            x->p_mel_formula_ = contorchionist::core::util_conversions::string_to_mel_formula_type(
                parser.get_string("melcalc mc mel_calc", mel_formula_str_default)
            );
        } catch (const std::invalid_argument& e) {
            pd_error(x, "torch.melspectrogram~: Invalid melcalc '%s' in args. Using default %s. Error: %s",
                     parser.get_string("melcalc mc mel_calc").c_str(), mel_formula_str_default.c_str(), e.what());
        }
    }

    if (parser.has_flag("filterbank_norm fb_norm mel_filter_norm")) {
         x->p_filterbank_norm_ = parser.get_string("filterbank_norm fb_norm mel_filter_norm");
    }

    std::string mel_mode_str_default = contorchionist::core::ap_melspectrogram::mel_norm_mode_to_string(x->p_mel_norm_mode_);
    if (parser.has_flag("melmode mm")) {
        try {
            x->p_mel_norm_mode_ = contorchionist::core::ap_melspectrogram::string_to_mel_norm_mode(parser.get_string("melmode mm", mel_mode_str_default));
        } catch (const std::invalid_argument& e) {
            pd_error(x, "torch.melspectrogram~: Invalid melmode '%s'. Using default %s. Error: %s",
                     parser.get_string("melmode mm").c_str(), mel_mode_str_default.c_str(), e.what());
        }
    }

    std::string unit_str_default = contorchionist::core::util_conversions::spectrum_data_format_to_string(x->p_output_format_);
    if (parser.has_flag("unit u format fmt")) {
        try {
            x->p_output_format_ = contorchionist::core::util_conversions::string_to_spectrum_data_format(parser.get_string("unit u format fmt", unit_str_default));
        } catch (const std::invalid_argument& e) {
             pd_error(x, "torch.melspectrogram~: Invalid unit type '%s' in args. Using default %s.",
                     parser.get_string("unit u format fmt").c_str(), unit_str_default.c_str());
        }
    }

    std::string device_arg_str = parser.get_string("device D", "cpu");
    bool device_flag_present = parser.has_flag("device D");
    auto device_result = get_device_from_string(device_arg_str);
    x->p_device_ = device_result.first;
    if (!device_result.second && device_flag_present) {
         pd_error(x, "torch.melspectrogram~: Invalid device string '%s'. Defaulting to CPU.", device_arg_str.c_str());
         x->p_device_ = torch::kCPU;
    }

    x->out_list_ = outlet_new(&x->x_obj, &s_list);

    if (x->p_sample_rate_ > 0) {
        torch_melspectrogram_tilde_configure_processor(x);
    }

    return (void *)x;
}

void torch_melspectrogram_tilde_free(t_torch_melspectrogram_tilde *x) {
    // x->processor_ is unique_ptr, handles its own deletion.
}


extern "C" void setup_torch0x2emelspectrogram_tilde(void) {
    torch_melspectrogram_tilde_class = class_new(
        gensym("torch.melspectrogram~"),
        (t_newmethod)torch_melspectrogram_tilde_new,
        (t_method)torch_melspectrogram_tilde_free,
        sizeof(t_torch_melspectrogram_tilde),
        CLASS_DEFAULT, A_GIMME, 0);

    CLASS_MAINSIGNALIN(torch_melspectrogram_tilde_class, t_torch_melspectrogram_tilde, x_f);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_dsp, gensym("dsp"), A_CANT, 0);
    
    // Parameter setting methods
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_n_fft, gensym("n_fft"), A_FLOAT, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_hop_length, gensym("hop_length"), A_FLOAT, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_win_length, gensym("win_length"), A_FLOAT, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_window, gensym("window"), A_DEFSYMBOL, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_norm, gensym("norm"), A_DEFSYMBOL, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_n_mels, gensym("n_mels"), A_FLOAT, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_fmin_mel, gensym("fmin_mel"), A_FLOAT, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_fmax_mel, gensym("fmax_mel"), A_FLOAT, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_melcalc, gensym("melcalc"), A_DEFSYMBOL, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_filterbank_norm, gensym("filterbank_norm"), A_DEFSYMBOL, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_device, gensym("device"), A_DEFSYMBOL, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_verbose, gensym("verbose"), A_FLOAT, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_unit, gensym("unit"), A_DEFSYMBOL, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_melmode, gensym("melmode"), A_DEFSYMBOL, 0);

    // Aliases
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_n_fft, gensym("N"), A_FLOAT, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_hop_length, gensym("H"), A_FLOAT, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_win_length, gensym("W"), A_FLOAT, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_window, gensym("w"), A_DEFSYMBOL, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_norm, gensym("n"), A_DEFSYMBOL, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_n_mels, gensym("M"), A_FLOAT, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_device, gensym("dev"), A_DEFSYMBOL, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_verbose, gensym("v"), A_FLOAT, 0);
    class_addmethod(torch_melspectrogram_tilde_class, (t_method)torch_melspectrogram_tilde_unit, gensym("u"), A_DEFSYMBOL, 0);

    class_sethelpsymbol(torch_melspectrogram_tilde_class, gensym("torch.melspectrogram~"));
    post("torch.melspectrogram~ v1.3 - Mel spectrogram analysis for Pure Data with header-only core");
}

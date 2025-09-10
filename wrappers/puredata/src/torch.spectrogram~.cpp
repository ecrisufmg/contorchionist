#include "m_pd.h"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <algorithm> // For std::transform for lowercase string conversion

#include "../../../core/include/core_ap_spectrogram.h"
#include "../utils/include/pd_arg_parser.h"
#include "../utils/include/pd_torch_device_adapter.h" // For get_device_from_string
#include "../../../core/include/core_util_windowing.h" 
#include "../../../core/include/core_util_normalizations.h" // For NormalizationType and string_to_normalization_type
#include "../../../core/include/core_util_conversions.h" // For SpectrumDataFormat and string_to_spectrum_data_format

struct SpectrogramTildeHelper {
    std::unique_ptr<contorchionist::core::ap_spectrogram::SpectrogramProcessor<float>> processor;
    std::mutex processor_mutex;
};

static t_class *torch_spectrogram_tilde_class;

typedef struct _torch_spectrogram_tilde {
    t_object x_obj;
    t_sample x_f; // Dummy for CLASS_MAINSIGNALIN

    SpectrogramTildeHelper *helper;

    // Outlets for list data
    t_outlet *out_frame1_; // For the first component of the spectrum (e.g., magnitude)
    t_outlet *out_frame2_; // For the second component of the spectrum (e.g., phase)

    // Parameters stored in the wrapper to re-create/re-configure the processor
    int n_fft_;
    int hop_size_;
    contorchionist::core::util_windowing::Type window_type_;
    contorchionist::core::util_conversions::SpectrumDataFormat output_format_;
    contorchionist::core::util_normalizations::NormalizationType norm_type_;
    torch::Device device_;
    float sampling_rate_;
    bool verbose_;

} t_torch_spectrogram_tilde;

// Forward declarations
static void torch_spectrogram_tilde_configure_processor(t_torch_spectrogram_tilde *x);
static void torch_spectrogram_tilde_output_frame(t_torch_spectrogram_tilde *x, const std::vector<float>& frame_data, t_outlet* target_outlet);

// DSP perform routine
static t_int *torch_spectrogram_tilde_perform(t_int *w) {
    t_torch_spectrogram_tilde *x = (t_torch_spectrogram_tilde *)(w[1]);
    t_sample *in_buf = (t_sample *)(w[2]);
    int n_block_samples = (int)(w[3]); // Number of samples in the input Pd block

    std::lock_guard<std::mutex> guard(x->helper->processor_mutex);

    if (!x->helper->processor || n_block_samples <= 0) {
        return (w + 4); // Object not ready or empty block
    }

    std::vector<float> frame1_data, frame2_data;

    if (x->helper->processor->process(in_buf, n_block_samples, frame1_data, frame2_data)) {
        torch_spectrogram_tilde_output_frame(x, frame1_data, x->out_frame1_);
        if (!frame2_data.empty()) {
            torch_spectrogram_tilde_output_frame(x, frame2_data, x->out_frame2_);
        }
    }

    return (w + 4);
}

// Helper to output a spectral frame (list) to a given outlet
static void torch_spectrogram_tilde_output_frame(t_torch_spectrogram_tilde *x, const std::vector<float>& frame_data, t_outlet* target_outlet) {
    if (frame_data.empty()) {
        return;
    }

    size_t num_elements = frame_data.size();
    t_atom* atom_list = (t_atom*)getbytes(num_elements * sizeof(t_atom));
    if (!atom_list) {
        pd_error(x, "torch.spectrogram~: Could not allocate memory for output list.");
        return;
    }

    for (size_t i = 0; i < num_elements; ++i) {
        SETFLOAT(&atom_list[i], frame_data[i]);
    }
    outlet_list(target_outlet, &s_list, static_cast<int>(num_elements), atom_list);
    freebytes(atom_list, num_elements * sizeof(t_atom));
}

// DSP add method
static void torch_spectrogram_tilde_dsp(t_torch_spectrogram_tilde *x, t_signal **sp) {
    std::lock_guard<std::mutex> guard(x->helper->processor_mutex);
    float current_sr = sp[0]->s_sr;
    if (x->sampling_rate_ != current_sr && current_sr > 0) { // ensure current_sr is valid
        x->sampling_rate_ = current_sr;
        if (x->helper->processor) {
            x->helper->processor->set_sampling_rate(x->sampling_rate_);
             if (x->verbose_) post("torch.spectrogram~: Sample rate updated to %.f Hz via DSP method.", x->sampling_rate_);
        } else {
            torch_spectrogram_tilde_configure_processor(x);
        }
    } else if (x->sampling_rate_ <= 0 && current_sr > 0) { // Initial valid SR obtained
        x->sampling_rate_ = current_sr;
        torch_spectrogram_tilde_configure_processor(x);
    }
    
    dsp_add(torch_spectrogram_tilde_perform, 3, x, sp[0]->s_vec, (t_int)sp[0]->s_n);
    if (x->verbose_) {
         post("torch.spectrogram~: DSP chain %s. Sample rate: %.f", x->helper->processor ? "updated" : "added (processor may require configuration)", x->sampling_rate_);
    }
}

// Centralized method to (re)create and configure the SpectrogramProcessor
static void torch_spectrogram_tilde_configure_processor(t_torch_spectrogram_tilde *x) {
    if (x->n_fft_ <= 0 || x->hop_size_ <= 0) {
        if (x->verbose_ || x->helper->processor)
            pd_error(x, "torch.spectrogram~: n_fft and hop_size must be positive. Processor not configured.");
        x->helper->processor.reset();
        return;
    }
    if (x->sampling_rate_ <= 0) {
         if (x->verbose_ || x->helper->processor)
            pd_error(x, "torch.spectrogram~: Sampling rate must be positive (current: %.f). Processor not configured.", x->sampling_rate_);
        x->helper->processor.reset();
        return;
    }

    try {
        if (x->verbose_) {
            post("torch.spectrogram~: Configuring processor: n_fft=%d, hop=%d, win=%s, norm=%s, unit=%s, sr=%.f, device=%s",
                 x->n_fft_, x->hop_size_,
                 contorchionist::core::util_windowing::torch_window_type_to_string(x->window_type_).c_str(),
                 contorchionist::core::util_normalizations::normalization_type_to_string(x->norm_type_).c_str(),
                 contorchionist::core::util_conversions::spectrum_data_format_to_string(x->output_format_).c_str(),
                 x->sampling_rate_, x->device_.str().c_str());
        }

        x->helper->processor = std::make_unique<contorchionist::core::ap_spectrogram::SpectrogramProcessor<float>>(
            x->n_fft_,
            x->hop_size_,
            x->window_type_,
            x->output_format_,
            x->norm_type_,
            x->device_,
            x->sampling_rate_,
            x->verbose_
        );
         if (x->verbose_) {
            post("torch.spectrogram~: Processor configured successfully.");
        }
    } catch (const std::exception& e) {
        pd_error(x, "torch.spectrogram~: Error configuring processor: %s", e.what());
        x->helper->processor.reset();
    }
}

// Methods for messages from Pd
static void torch_spectrogram_tilde_nfft(t_torch_spectrogram_tilde *x, t_floatarg val) {
    std::lock_guard<std::mutex> guard(x->helper->processor_mutex);
    int new_n_fft = static_cast<int>(val);
    if (new_n_fft <= 0) {
        pd_error(x, "torch.spectrogram~: n_fft must be positive, got %d", new_n_fft);
        return;
    }
    if (x->n_fft_ != new_n_fft) {
        x->n_fft_ = new_n_fft;
        torch_spectrogram_tilde_configure_processor(x);
    }
}

static void torch_spectrogram_tilde_hop(t_torch_spectrogram_tilde *x, t_floatarg val) {
    std::lock_guard<std::mutex> guard(x->helper->processor_mutex);
    int new_hop_size = static_cast<int>(val);
    if (new_hop_size <= 0) {
        pd_error(x, "torch.spectrogram~: hop_size must be positive, got %d", new_hop_size);
        return;
    }
    if (x->hop_size_ != new_hop_size) {
        x->hop_size_ = new_hop_size;
        torch_spectrogram_tilde_configure_processor(x);
    }
}

static void torch_spectrogram_tilde_window(t_torch_spectrogram_tilde *x, t_symbol *s) {
    std::lock_guard<std::mutex> guard(x->helper->processor_mutex);
    try {
        contorchionist::core::util_windowing::Type new_type = contorchionist::core::util_windowing::string_to_torch_window_type(s->s_name);
        if (x->window_type_ != new_type) {
            x->window_type_ = new_type;
            if (x->helper->processor) x->helper->processor->set_window_type(x->window_type_);
            else torch_spectrogram_tilde_configure_processor(x);
            if (x->verbose_) post("torch.spectrogram~: Window type set to %s", s->s_name);
        }
    } catch (const std::invalid_argument& e) {
        pd_error(x, "torch.spectrogram~: Invalid window type '%s': %s", s->s_name, e.what());
    }
}

static void torch_spectrogram_tilde_norm(t_torch_spectrogram_tilde *x, t_symbol *s) {
    std::lock_guard<std::mutex> guard(x->helper->processor_mutex);
    try {
        contorchionist::core::util_normalizations::NormalizationType new_type =
            contorchionist::core::util_normalizations::string_to_normalization_type(s->s_name, false);
        if (x->norm_type_ != new_type) {
            x->norm_type_ = new_type;
            if (x->helper->processor) x->helper->processor->set_normalization_type(x->norm_type_);
            else torch_spectrogram_tilde_configure_processor(x);
             if (x->verbose_) post("torch.spectrogram~: Normalization type set to %s", s->s_name);
        }
    } catch (const std::invalid_argument& e) {
        pd_error(x, "torch.spectrogram~: Invalid normalization type '%s': %s", s->s_name, e.what());
    }
}

static void torch_spectrogram_tilde_unit(t_torch_spectrogram_tilde *x, t_symbol *s) {
    std::lock_guard<std::mutex> guard(x->helper->processor_mutex);
    try {
        contorchionist::core::util_conversions::SpectrumDataFormat new_format =
            contorchionist::core::util_conversions::string_to_spectrum_data_format(s->s_name);
        if (x->output_format_ != new_format) {
            x->output_format_ = new_format;
            if (x->helper->processor) x->helper->processor->set_output_format(x->output_format_);
            else torch_spectrogram_tilde_configure_processor(x);
            if (x->verbose_) post("torch.spectrogram~: Output unit set to %s", s->s_name);
        }
    } catch (const std::invalid_argument& e) {
        pd_error(x, "torch.spectrogram~: Invalid unit type '%s': %s", s->s_name, e.what());
    }
}

static void torch_spectrogram_tilde_device(t_torch_spectrogram_tilde *x, t_symbol *s) {
    std::lock_guard<std::mutex> guard(x->helper->processor_mutex);
    auto device_result = get_device_from_string(s->s_name);
    if (device_result.second) {
        if (x->device_.type() != device_result.first.type() || x->device_.index() != device_result.first.index()) {
            x->device_ = device_result.first;
            torch_spectrogram_tilde_configure_processor(x);
            if (x->verbose_) post("torch.spectrogram~: Device set to %s", s->s_name);
        }
    } else {
        pd_error(x, "torch.spectrogram~: Invalid device string '%s'", s->s_name);
    }
}

static void torch_spectrogram_tilde_verbose(t_torch_spectrogram_tilde *x, t_floatarg f) {
    std::lock_guard<std::mutex> guard(x->helper->processor_mutex);
    bool new_verbose = static_cast<bool>(f);
    if (x->verbose_ != new_verbose) {
        x->verbose_ = new_verbose; // Update the stored verbose state
        if (x->helper->processor) {
            x->helper->processor->set_verbose(x->verbose_); // Call the new setter on SpectrogramProcessor
        } else {
            // If processor doesn't exist, a call to configure_processor
            // (e.g., due to a later SR update or explicit parameter change)
            // will use the updated x->verbose_.
            // For consistency with other setters that attempt reconfiguration if processor is null:
            torch_spectrogram_tilde_configure_processor(x);
        }
        post("torch.spectrogram~: Verbose mode %s.", x->verbose_ ? "enabled" : "disabled");
    }
}

// Constructor
static void *torch_spectrogram_tilde_new(t_symbol *s, int argc, t_atom *argv) {
    t_torch_spectrogram_tilde *x = (t_torch_spectrogram_tilde *)pd_new(torch_spectrogram_tilde_class);
    x->helper = new SpectrogramTildeHelper();

    x->n_fft_ = 1024;
    x->hop_size_ = x->n_fft_ / 4;
    x->window_type_ = contorchionist::core::util_windowing::Type::HANN;
    x->output_format_ = contorchionist::core::util_conversions::SpectrumDataFormat::MAGPHASE;
    x->norm_type_ = contorchionist::core::util_normalizations::NormalizationType::WINDOW;
    x->device_ = torch::kCPU;
    x->sampling_rate_ = sys_getsr();
    x->verbose_ = false;
    x->helper->processor = nullptr;

    pd_utils::ArgParser parser(argc, argv, &x->x_obj);
    x->verbose_ = parser.has_flag("verbose v");

    x->n_fft_ = static_cast<int>(parser.get_float("n_fft N nfft", x->n_fft_));
    x->hop_size_ = static_cast<int>(parser.get_float("hop H hopsize", static_cast<float>(x->n_fft_ / 4)));

    std::string win_str_default = contorchionist::core::util_windowing::torch_window_type_to_string(x->window_type_);
    if (parser.has_flag("window win w")) {
        try {
            x->window_type_ = contorchionist::core::util_windowing::string_to_torch_window_type(parser.get_string("window win w", win_str_default));
        } catch (const std::invalid_argument& e) {
            pd_error(x, "torch.spectrogram~: Invalid window type '%s' in args. Using default %s.",
                     parser.get_string("window win w").c_str(), win_str_default.c_str());
        }
    }

    std::string norm_str_default = contorchionist::core::util_normalizations::normalization_type_to_string(x->norm_type_);
    if (parser.has_flag("norm n")) {
        try {
            x->norm_type_ = contorchionist::core::util_normalizations::string_to_normalization_type(parser.get_string("norm n", norm_str_default), false);
        } catch (const std::invalid_argument& e) {
            pd_error(x, "torch.spectrogram~: Invalid norm type '%s' in args. Using default %s.",
                     parser.get_string("norm n").c_str(), norm_str_default.c_str());
        }
    }

    std::string unit_str_default = contorchionist::core::util_conversions::spectrum_data_format_to_string(x->output_format_);
    if (parser.has_flag("unit u format fmt")) {
        try {
            x->output_format_ = contorchionist::core::util_conversions::string_to_spectrum_data_format(parser.get_string("unit u format fmt", unit_str_default));
        } catch (const std::invalid_argument& e) {
             pd_error(x, "torch.spectrogram~: Invalid unit type '%s' in args. Using default %s.",
                     parser.get_string("unit u format fmt").c_str(), unit_str_default.c_str());
        }
    }
    
    std::string device_arg_str = parser.get_string("device d dev", "cpu");
    bool device_flag_present = parser.has_flag("device d dev");
    auto device_result = get_device_from_string(device_arg_str);
    x->device_ = device_result.first;
    if (!device_result.second && device_flag_present) {
         pd_error(x, "torch.spectrogram~: Invalid device string '%s'. Defaulting to CPU.", device_arg_str.c_str());
         x->device_ = torch::kCPU;
    }

    x->out_frame1_ = outlet_new(&x->x_obj, &s_list);
    x->out_frame2_ = outlet_new(&x->x_obj, &s_list);

    if (x->sampling_rate_ <= 0) {
        x->sampling_rate_ = 44100.0f;
        if (x->verbose_) {
            post("torch.spectrogram~: sys_getsr() was 0 or invalid at creation, defaulting SR to %.f. Will update in DSP.", x->sampling_rate_);
        }
    }

    if (x->sampling_rate_ > 0) {
        torch_spectrogram_tilde_configure_processor(x);
    } else if (x->verbose_) {
        post("torch.spectrogram~: Deferring processor configuration until valid sample rate is received in DSP method.");
    }

    return (void *)x;
}

static void torch_spectrogram_tilde_free(t_torch_spectrogram_tilde *x) {
    delete x->helper;
}

extern "C" void setup_torch0x2espectrogram_tilde(void) {
    torch_spectrogram_tilde_class = class_new(gensym("torch.spectrogram~"),
                                       (t_newmethod)torch_spectrogram_tilde_new,
                                       (t_method)torch_spectrogram_tilde_free,
                                       sizeof(t_torch_spectrogram_tilde),
                                       CLASS_DEFAULT, A_GIMME, 0);

    CLASS_MAINSIGNALIN(torch_spectrogram_tilde_class, t_torch_spectrogram_tilde, x_f);
    class_addmethod(torch_spectrogram_tilde_class, (t_method)torch_spectrogram_tilde_dsp, gensym("dsp"), A_CANT, 0);

    class_addmethod(torch_spectrogram_tilde_class, (t_method)torch_spectrogram_tilde_nfft, gensym("n_fft"), A_FLOAT, 0);
    class_addmethod(torch_spectrogram_tilde_class, (t_method)torch_spectrogram_tilde_nfft, gensym("N"), A_FLOAT, 0);
    class_addmethod(torch_spectrogram_tilde_class, (t_method)torch_spectrogram_tilde_hop, gensym("hop_size"), A_FLOAT, 0);
    class_addmethod(torch_spectrogram_tilde_class, (t_method)torch_spectrogram_tilde_hop, gensym("hop"), A_FLOAT, 0);
    class_addmethod(torch_spectrogram_tilde_class, (t_method)torch_spectrogram_tilde_hop, gensym("H"), A_FLOAT, 0);
    class_addmethod(torch_spectrogram_tilde_class, (t_method)torch_spectrogram_tilde_window, gensym("window"), A_DEFSYMBOL, 0);
    class_addmethod(torch_spectrogram_tilde_class, (t_method)torch_spectrogram_tilde_window, gensym("win"), A_DEFSYMBOL, 0);
    class_addmethod(torch_spectrogram_tilde_class, (t_method)torch_spectrogram_tilde_norm, gensym("norm"), A_DEFSYMBOL, 0);
    class_addmethod(torch_spectrogram_tilde_class, (t_method)torch_spectrogram_tilde_unit, gensym("unit"), A_DEFSYMBOL, 0);
    class_addmethod(torch_spectrogram_tilde_class, (t_method)torch_spectrogram_tilde_unit, gensym("format"), A_DEFSYMBOL, 0);
    class_addmethod(torch_spectrogram_tilde_class, (t_method)torch_spectrogram_tilde_device, gensym("device"), A_DEFSYMBOL, 0);
    class_addmethod(torch_spectrogram_tilde_class, (t_method)torch_spectrogram_tilde_device, gensym("dev"), A_DEFSYMBOL, 0);
    class_addmethod(torch_spectrogram_tilde_class, (t_method)torch_spectrogram_tilde_verbose, gensym("verbose"), A_FLOAT, 0);
    class_addmethod(torch_spectrogram_tilde_class, (t_method)torch_spectrogram_tilde_verbose, gensym("v"), A_FLOAT, 0);

    class_sethelpsymbol(torch_spectrogram_tilde_class, gensym("torch.spectrogram~"));
}

#include "m_pd.h"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring> // Para memcpy

#include "../utils/include/pd_arg_parser.h"
#include "../../../core/include/core_util_windowing.h"
#include "../utils/include/pd_torch_device_adapter.h"
#include "../../../core/include/core_ap_rfft.h"
#include "../../../core/include/core_util_conversions.h"
#include "../../../core/include/core_util_normalizations.h"

// Classe global
static t_class *torch_irfft_tilde_class;

// Estrutura de dados do objeto
typedef struct _torch_irfft_tilde {
    t_object x_obj;
    t_sample x_f_dummy; // Dummy para o primeiro inlet de sinal
    
    // Inlets e Outlets
    t_inlet *in2_;     // Inlet para o segundo componente (Imag/Phase)
    t_outlet *out_;    // Outlet para o sinal de tempo

    // Instância do processador
    contorchionist::core::ap_rfft::RFFTProcessor<float> rfft_processor_;
    
    // Configurações do wrapper para reconfigurar o processador
    contorchionist::core::util_conversions::SpectrumDataFormat input_format_;
    contorchionist::core::util_windowing::Type win_ref_type_; // Janela de referência para de-normalização
    contorchionist::core::util_normalizations::NormalizationType normalization_type_;
    int overlap_factor_;
    long output_n_; // Tamanho da saída da IRFFT
    bool use_input_phase_; // Se deve usar a fase de entrada ou zerá-la

    // Estado do ambiente Pd
    int current_block_size_;
    float sampling_rate_;
    
} t_torch_irfft_tilde;


// Protótipo da função de configuração central
static void update_processor_settings(t_torch_irfft_tilde *x);

// DSP perform routine
static t_int *torch_irfft_tilde_perform(t_int *w) {
    t_torch_irfft_tilde *x = (t_torch_irfft_tilde *)(w[1]);
    t_sample *in1_buf = (t_sample *)(w[2]); // Real / Mag / Power / dB
    t_sample *in2_buf = (t_sample *)(w[3]); // Imag / Phase
    t_sample *out_buf = (t_sample *)(w[4]);
    int n = (int)(w[5]);

    auto zero_output_and_return = [&]() {
        for (int i = 0; i < n; ++i) { out_buf[i] = 0.0f; }
        return (w + 6);
    };
    
    if (n <= 0) {
        return zero_output_and_return();
    }
    
    try {
        auto cpu_tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        
        torch::Tensor first_component = torch::from_blob(in1_buf, {n}, cpu_tensor_options);
        torch::Tensor second_component = torch::from_blob(in2_buf, {n}, cpu_tensor_options);
        std::vector<torch::Tensor> input_tensors = {first_component, second_component};

        // O tamanho da saída da IRFFT é o tamanho do bloco atual, a menos que especificado pelo usuário.
        long irfft_n = (x->output_n_ > 0) ? x->output_n_ : n;

        // Chamada única ao processador, que faz a conversão e a IRFFT
        torch::Tensor time_signal = x->rfft_processor_.process_irfft(
            input_tensors,
            irfft_n,
            x->input_format_,
            x->use_input_phase_
        );
        
        torch::Tensor time_signal_cpu = time_signal.cpu().contiguous();

        // Zera o buffer antes de copiar
        memset(out_buf, 0, n * sizeof(float));

        // Copia os dados para a saída do Pd
        int num_to_copy = std::min(n, static_cast<int>(time_signal_cpu.numel()));
        if (num_to_copy > 0)
            memcpy(out_buf, time_signal_cpu.data_ptr<float>(), num_to_copy * sizeof(float));

    } catch (const c10::Error& e) {
        pd_error(&x->x_obj, "torch.irfft~: LibTorch error in DSP: %s", e.what());
        return zero_output_and_return();
    } catch (const std::exception& e) {
        pd_error(&x->x_obj, "torch.irfft~: Standard error in DSP: %s", e.what());
        return zero_output_and_return();
    }

    return (w + 6);
}

// DSP add method
static void torch_irfft_tilde_dsp(t_torch_irfft_tilde *x, t_signal **sp) {
    bool block_size_changed = (x->current_block_size_ != sp[0]->s_n);
    bool sample_rate_changed = (x->sampling_rate_ != sys_getsr());

    if (sp[0]->s_n != sp[1]->s_n) {
        pd_error(&x->x_obj, "torch.irfft~: Block sizes of input signals must match.");
        return;
    }

    if (block_size_changed) {
        x->current_block_size_ = sp[0]->s_n;
    }
    if (sample_rate_changed) {
        x->sampling_rate_ = sys_getsr();
    }

    // Se qualquer parâmetro chave do DSP mudou, reconfiguramos o processador.
    if (block_size_changed || sample_rate_changed) {
        update_processor_settings(x);
    }
    
    dsp_add(torch_irfft_tilde_perform, 5, x, sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, (t_int)sp[0]->s_n);
    
    if (x->rfft_processor_.is_verbose()) {
        post("torch.irfft~: DSP chain added. Block size: %d, Sample rate: %.f", x->current_block_size_, x->sampling_rate_);
    }
}

// Função centralizadora de configuração
static void update_processor_settings(t_torch_irfft_tilde *x) {
    if (x->current_block_size_ <= 0 || x->sampling_rate_ <= 0) {
        if (x->rfft_processor_.is_verbose()) {
            post("torch.irfft~: Cannot update settings, block size or sample rate is invalid.");
        }
        return;
    }
    
    // Atualiza a janela de referência no processador antes de calcular a normalização
    x->rfft_processor_.set_window_type(x->win_ref_type_);
    // Para IRFFT, a flag 'windowing_enabled' não se aplica, a janela é apenas para referência.
    x->rfft_processor_.enable_windowing(false);

    try {
        // Chama set_normalization para o processo INVERSO
        x->rfft_processor_.set_normalization(
            -1, // -1 = inverse FFT
            2,  // 2 = RFFT mode
            x->current_block_size_, // O tamanho da FFT original é o tamanho do bloco de entrada
            x->normalization_type_,
            x->sampling_rate_,
            static_cast<float>(x->overlap_factor_)
        );

        if (x->rfft_processor_.is_verbose()) {
            post("torch.irfft~: Processor settings updated. n_fft_ref=%d, fs=%.f, norm=%s, win_ref=%s, overlap=%d",
                 x->current_block_size_,
                 x->sampling_rate_,
                 contorchionist::core::util_normalizations::normalization_type_to_string(x->normalization_type_).c_str(),
                 contorchionist::core::util_windowing::torch_window_type_to_string(x->win_ref_type_).c_str(),
                 x->overlap_factor_);
        }
    } catch (const std::exception& e) {
        pd_error(&x->x_obj, "torch.irfft~: Error updating processor settings: %s", e.what());
    }
}


// Métodos para receber mensagens do Pd
static void torch_irfft_tilde_winref(t_torch_irfft_tilde *x, t_symbol *s) {
    try {
        x->win_ref_type_ = contorchionist::core::util_windowing::string_to_torch_window_type(s->s_name);
        update_processor_settings(x);
    } catch (const std::invalid_argument& e) {
        pd_error(&x->x_obj, "torch.irfft~: Invalid window reference type '%s'. %s", s->s_name, e.what());
    }
}

static void torch_irfft_tilde_norm(t_torch_irfft_tilde *x, t_symbol *s) {
    try {
        // Para IRFFT, is_complex deve ser `true` para irfft (inverse operation)
        x->normalization_type_ = contorchionist::core::util_normalizations::string_to_normalization_type(s->s_name, true);
        update_processor_settings(x);
    } catch (const std::exception& e) {
        pd_error(&x->x_obj, "torch.irfft~: Invalid normalization mode '%s'. %s", s->s_name, e.what());
    }
}

static void torch_irfft_tilde_unit(t_torch_irfft_tilde *x, t_symbol *s) {
    try {
        x->input_format_ = contorchionist::core::util_conversions::string_to_spectrum_data_format(s->s_name);
        // A mudança do formato de entrada não requer recálculo dos fatores de normalização.
        if (x->rfft_processor_.is_verbose()) {
             post("torch.irfft~: Input unit format set to '%s'.", s->s_name);
        }
    } catch (const std::invalid_argument& e) {
        pd_error(&x->x_obj, "torch.irfft~: Invalid unit format '%s'. %s", s->s_name, e.what());
    }
}

static void torch_irfft_tilde_overlap(t_torch_irfft_tilde *x, t_floatarg f) {
    int new_overlap = static_cast<int>(f);
    if (new_overlap > 0 && (new_overlap & (new_overlap - 1)) == 0) { // Checa se é potência de 2
        x->overlap_factor_ = new_overlap;
        update_processor_settings(x);
    } else {
        pd_error(&x->x_obj, "torch.irfft~: overlap factor must be a power of 2, got %d", new_overlap);
    }
}

static void torch_irfft_tilde_output_n(t_torch_irfft_tilde *x, t_floatarg f) {
    x->output_n_ = static_cast<long>(f);
    if (x->rfft_processor_.is_verbose()) {
        post("torch.irfft~: Output size (n) set to %ld.", x->output_n_);
    }
}

static void torch_irfft_tilde_use_phase(t_torch_irfft_tilde *x, t_floatarg f) {
    x->use_input_phase_ = static_cast<bool>(f);
    if (x->rfft_processor_.is_verbose()) {
        post("torch.irfft~: Use input phase set to %s.", x->use_input_phase_ ? "true" : "false");
    }
}

static void torch_irfft_tilde_verbose(t_torch_irfft_tilde *x, t_floatarg f) {
    x->rfft_processor_.set_verbose(static_cast<bool>(f));
    post("torch.irfft~: Verbose mode %s.", static_cast<bool>(f) ? "enabled" : "disabled");
}

static void torch_irfft_tilde_device(t_torch_irfft_tilde *x, t_symbol *s) {
    std::string device_str = s->s_name;
    torch::Device current_device = x->rfft_processor_.get_device(); // Get current device
    pd_parse_and_set_torch_device(&x->x_obj, current_device, device_str, x->rfft_processor_.is_verbose(), "torch.irfft~", true);
    x->rfft_processor_.set_device(current_device); // Set the modified device back to the processor
}


// Construtor
static void *torch_irfft_tilde_new(t_symbol *s, int argc, t_atom *argv) {
    t_torch_irfft_tilde *x = (t_torch_irfft_tilde *)pd_new(torch_irfft_tilde_class);

    // Valores padrão
    x->current_block_size_ = 0;
    x->sampling_rate_ = sys_getsr() > 0 ? sys_getsr() : 44100.f;
    x->input_format_ = contorchionist::core::util_conversions::SpectrumDataFormat::COMPLEX;
    x->win_ref_type_ = contorchionist::core::util_windowing::Type::RECTANGULAR;
    x->normalization_type_ = contorchionist::core::util_normalizations::NormalizationType::NONE;
    x->overlap_factor_ = 1;
    x->output_n_ = 0; // 0 significa usar o tamanho do bloco como padrão
    x->use_input_phase_ = true; // Por padrão, usa a fase fornecida
    
    // Parser de argumentos
    pd_utils::ArgParser parser(argc, argv, &x->x_obj);
    bool verbose_arg = parser.has_flag("verbose v");
    
    // Parse device using the correct approach
    bool device_flag_present = parser.has_flag("device d");
    std::string device_arg_str = parser.get_string("device d", "cpu");
    
    // Get device from string
    auto device_result = get_device_from_string(device_arg_str);
    torch::Device device = device_result.first;
    bool device_parse_success = device_result.second;
    
    x->input_format_ = contorchionist::core::util_conversions::string_to_spectrum_data_format(parser.get_string("unit u format", "complex"));
    x->win_ref_type_ = contorchionist::core::util_windowing::string_to_torch_window_type(parser.get_string("winref wr w window", "rectangular"));
    x->normalization_type_ = contorchionist::core::util_normalizations::string_to_normalization_type(parser.get_string("norm n", "none"), true);
    x->overlap_factor_ = static_cast<int>(parser.get_float("overlap of", 1));
    x->output_n_ = static_cast<long>(parser.get_float("n output_n", 0));
    x->use_input_phase_ = static_cast<bool>(parser.get_float("use_phase", 1));

    // Instancia o RFFTProcessor. A maioria dos parâmetros são apenas placeholders,
    // pois a configuração real para IRFFT acontece em update_processor_settings.
    new (&x->rfft_processor_) contorchionist::core::ap_rfft::RFFTProcessor<float>(
        device,
        x->win_ref_type_, // Passa a janela de referência
        false,            // windowing_enabled é sempre false para irfft
        verbose_arg,
        x->normalization_type_,
        x->input_format_, // A processor 'output' is our 'input'
        x->sampling_rate_
    );
    
    // Use pd_parse_and_set_torch_device to handle device parsing and logging
    torch::Device final_device = device;
    pd_parse_and_set_torch_device(&x->x_obj, final_device, device_arg_str, x->rfft_processor_.is_verbose(), "torch.irfft~", device_flag_present);
    x->rfft_processor_.set_device(final_device);
    
    // Cria inlets e outlets
    x->in2_ = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
    x->out_ = outlet_new(&x->x_obj, &s_signal);
    
    return (void *)x;
}

// Destrutor
static void torch_irfft_tilde_free(t_torch_irfft_tilde *x) {
    inlet_free(x->in2_);
    x->rfft_processor_.~RFFTProcessor<float>();
}

// Função de setup da classe
extern "C" void setup_torch0x2eirfft_tilde(void) {
    torch_irfft_tilde_class = class_new(gensym("torch.irfft~"),
                                       (t_newmethod)torch_irfft_tilde_new,
                                       (t_method)torch_irfft_tilde_free,
                                       sizeof(t_torch_irfft_tilde),
                                       CLASS_DEFAULT, A_GIMME, 0);

    CLASS_MAINSIGNALIN(torch_irfft_tilde_class, t_torch_irfft_tilde, x_f_dummy);
    class_addmethod(torch_irfft_tilde_class, (t_method)torch_irfft_tilde_dsp, gensym("dsp"), A_CANT, 0);
    
    // Métodos
    class_addmethod(torch_irfft_tilde_class, (t_method)torch_irfft_tilde_winref, gensym("winref"), A_DEFSYMBOL, 0);
    class_addmethod(torch_irfft_tilde_class, (t_method)torch_irfft_tilde_winref, gensym("wr"), A_DEFSYMBOL, 0);
    class_addmethod(torch_irfft_tilde_class, (t_method)torch_irfft_tilde_winref, gensym("w"), A_DEFSYMBOL, 0);
    class_addmethod(torch_irfft_tilde_class, (t_method)torch_irfft_tilde_winref, gensym("window"), A_DEFSYMBOL, 0);
    class_addmethod(torch_irfft_tilde_class, (t_method)torch_irfft_tilde_norm, gensym("norm"), A_DEFSYMBOL, 0);
    class_addmethod(torch_irfft_tilde_class, (t_method)torch_irfft_tilde_unit, gensym("unit"), A_DEFSYMBOL, 0);
    class_addmethod(torch_irfft_tilde_class, (t_method)torch_irfft_tilde_overlap, gensym("overlap"), A_FLOAT, 0);
    class_addmethod(torch_irfft_tilde_class, (t_method)torch_irfft_tilde_output_n, gensym("n"), A_FLOAT, 0);
    class_addmethod(torch_irfft_tilde_class, (t_method)torch_irfft_tilde_use_phase, gensym("use_phase"), A_FLOAT, 0);
    class_addmethod(torch_irfft_tilde_class, (t_method)torch_irfft_tilde_verbose, gensym("verbose"), A_FLOAT, 0);

    class_sethelpsymbol(torch_irfft_tilde_class, gensym("torch.irfft~"));
}
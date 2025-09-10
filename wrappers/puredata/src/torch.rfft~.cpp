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
static t_class *torch_rfft_tilde_class;

// -> ADAPTADO: Estrutura de dados simplificada
typedef struct _torch_rfft_tilde {
    t_object x_obj;
    t_sample x_f; // Dummy para CLASS_MAINSIGNALIN

    // Saídas
    t_outlet *out1_; // Saída para componente 1 (Real, Mag, Power, dB)
    t_outlet *out2_; // Saída para componente 2 (Imag, Phase)

    // Instância do processador
    contorchionist::core::ap_rfft::RFFTProcessor<float> rfft_processor_;
    
    // Configurações do wrapper para reconfigurar o processador
    contorchionist::core::util_windowing::Type window_type_;
    contorchionist::core::util_normalizations::NormalizationType normalization_type_;
    bool windowing_enabled_;
    int overlap_factor_;

    // Estado do ambiente Pd
    int current_block_size_;
    float sampling_rate_;
    
} t_torch_rfft_tilde;

// Protótipo da função de configuração central
static void update_processor_settings(t_torch_rfft_tilde *x);

// DSP perform routine
static t_int *torch_rfft_tilde_perform(t_int *w) {
    t_torch_rfft_tilde *x = (t_torch_rfft_tilde *)(w[1]);
    t_sample *in_buf = (t_sample *)(w[2]);
    t_sample *out1_buf = (t_sample *)(w[3]);
    t_sample *out2_buf = (t_sample *)(w[4]);
    int n = (int)(w[5]);

    auto zero_outputs_and_return = [&]() {
        for (int i = 0; i < n; ++i) { out1_buf[i] = 0.0f; out2_buf[i] = 0.0f; }
        return (w + 6);
    };
    
    if (n <= 0) {
        return zero_outputs_and_return();
    }

    // -> ADAPTADO: A detecção de mudança de tamanho do bloco agora é feita no _dsp
    // O perform assume que a configuração já está correta.
    
    try {
        auto cpu_tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        torch::Tensor input_tensor_cpu = torch::from_blob(in_buf, {n}, cpu_tensor_options);

        // -> ADAPTADO: Chamada única ao processador, que já faz tudo internamente.
        std::vector<torch::Tensor> result_vec = x->rfft_processor_.process_rfft(input_tensor_cpu);
        
        if (result_vec.empty()) {
             return zero_outputs_and_return();
        }

        contorchionist::core::util_conversions::SpectrumDataFormat current_format = x->rfft_processor_.get_output_format();
        torch::Tensor first_component_tensor, second_component_tensor;

        if (current_format == contorchionist::core::util_conversions::SpectrumDataFormat::COMPLEX) {
            if (result_vec.size() != 1 || !result_vec[0].is_complex()) return zero_outputs_and_return();
            torch::Tensor complex_spectrum = result_vec[0].cpu(); 
            first_component_tensor = torch::real(complex_spectrum).contiguous();
            second_component_tensor = torch::imag(complex_spectrum).contiguous();
        } else {
            if (result_vec.size() != 2) return zero_outputs_and_return();
            first_component_tensor = result_vec[0].cpu().contiguous();
            second_component_tensor = result_vec[1].cpu().contiguous();
        }

        // Zera os buffers antes de copiar
        memset(out1_buf, 0, n * sizeof(float));
        memset(out2_buf, 0, n * sizeof(float));

        // Copia os dados para as saídas do Pd
        int num_to_copy_first = std::min(n, static_cast<int>(first_component_tensor.numel()));
        if (num_to_copy_first > 0)
            memcpy(out1_buf, first_component_tensor.data_ptr<float>(), num_to_copy_first * sizeof(float));

        int num_to_copy_second = std::min(n, static_cast<int>(second_component_tensor.numel()));
        if (num_to_copy_second > 0)
            memcpy(out2_buf, second_component_tensor.data_ptr<float>(), num_to_copy_second * sizeof(float));
            
        // -> ADAPTADO: TODA a lógica de correção de overlap e output de fator de normalização foi REMOVIDA.
        // O RFFTProcessor já aplicou a normalização correta diretamente nos valores do espectro.

    } catch (const c10::Error& e) {
        pd_error(x, "torch.rfft~: LibTorch error in DSP: %s", e.what());
        return zero_outputs_and_return();
    } catch (const std::exception& e) {
        pd_error(x, "torch.rfft~: Standard error in DSP: %s", e.what());
        return zero_outputs_and_return();
    }

    return (w + 6);
}

// DSP add method
static void torch_rfft_tilde_dsp(t_torch_rfft_tilde *x, t_signal **sp) {
    bool block_size_changed = (x->current_block_size_ != sp[0]->s_n);
    bool sample_rate_changed = (x->sampling_rate_ != sys_getsr());

    if (block_size_changed) {
        x->current_block_size_ = sp[0]->s_n;
    }
    if (sample_rate_changed) {
        x->sampling_rate_ = sys_getsr();
    }

    // -> ADAPTADO: Se qualquer parâmetro chave do DSP mudou, reconfiguramos o processador.
    if (block_size_changed || sample_rate_changed) {
        update_processor_settings(x);
    }
    
    dsp_add(torch_rfft_tilde_perform, 5, x, sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, (t_int)sp[0]->s_n);
    
    if (x->rfft_processor_.is_verbose()) {
        post("torch.rfft~: DSP chain added. Block size: %d, Sample rate: %.f", x->current_block_size_, x->sampling_rate_);
    }
}

// -> ADAPTADO: Nova função centralizadora de configuração
static void update_processor_settings(t_torch_rfft_tilde *x) {
    if (x->current_block_size_ <= 0 || x->sampling_rate_ <= 0) {
        if (x->rfft_processor_.is_verbose()) {
            post("torch.rfft~: Cannot update settings, block size or sample rate is invalid.");
        }
        return;
    }
    
    try {
        // Usa a nova assinatura de set_normalization, que é autossuficiente
        x->rfft_processor_.set_normalization(
            1,  // 1 = forward FFT
            2,  // 2 = RFFT mode (contorchionist::FFTMode::RFFT)
            x->current_block_size_,
            x->normalization_type_,
            x->sampling_rate_,
            static_cast<float>(x->overlap_factor_)
        );

        if (x->rfft_processor_.is_verbose()) {
            post("torch.rfft~: Processor settings updated. n=%d, fs=%.f, norm=%s, win=%s (enabled: %s), overlap=%d",
                 x->current_block_size_,
                 x->sampling_rate_,
                 contorchionist::core::util_normalizations::normalization_type_to_string(x->normalization_type_).c_str(),
                 contorchionist::core::util_windowing::torch_window_type_to_string(x->window_type_).c_str(),
                 x->windowing_enabled_ ? "yes" : "no",
                 x->overlap_factor_);
        }
    } catch (const std::exception& e) {
        pd_error(x, "torch.rfft~: Error updating processor settings: %s", e.what());
    }
}

// Métodos para receber mensagens do Pd
static void torch_rfft_tilde_window(t_torch_rfft_tilde *x, t_symbol *s) {
    try {
        x->window_type_ = contorchionist::core::util_windowing::string_to_torch_window_type(s->s_name);
        x->windowing_enabled_ = true; // Definir uma janela a habilita
        x->rfft_processor_.set_window_type(x->window_type_); // Atualiza na classe
        x->rfft_processor_.enable_windowing(true);
        update_processor_settings(x);
    } catch (const std::invalid_argument& e) {
        pd_error(x, "torch.rfft~: Invalid window type '%s'. %s", s->s_name, e.what());
    }
}

static void torch_rfft_tilde_norm(t_torch_rfft_tilde *x, t_symbol *s) {
    try {
        // O segundo argumento de string_to_normalization_type é `is_complex`, que deve ser false para rfft
        x->normalization_type_ = contorchionist::core::util_normalizations::string_to_normalization_type(s->s_name, false);
        update_processor_settings(x);
    } catch (const std::exception& e) {
        pd_error(x, "torch.rfft~: Invalid normalization mode '%s'. %s", s->s_name, e.what());
    }
}

static void torch_rfft_tilde_unit(t_torch_rfft_tilde *x, t_symbol *s) {
    try {
        auto new_format = contorchionist::core::util_conversions::string_to_spectrum_data_format(s->s_name);
        x->rfft_processor_.set_output_format(new_format);
        if (x->rfft_processor_.is_verbose()) {
             post("torch.rfft~: Output unit format set to '%s'.", s->s_name);
        }
    } catch (const std::invalid_argument& e) {
        pd_error(x, "torch.rfft~: Invalid unit format '%s'. %s", s->s_name, e.what());
    }
}

static void torch_rfft_tilde_window_enable(t_torch_rfft_tilde *x, t_floatarg f) {
    x->windowing_enabled_ = static_cast<bool>(f);
    x->rfft_processor_.enable_windowing(x->windowing_enabled_);
    update_processor_settings(x);
}

static void torch_rfft_tilde_overlap(t_torch_rfft_tilde *x, t_floatarg f) {
    int new_overlap = static_cast<int>(f);
    if (new_overlap > 0 && (new_overlap & (new_overlap - 1)) == 0) {
        x->overlap_factor_ = new_overlap;
        update_processor_settings(x);
    } else {
        pd_error(x, "torch.rfft~: overlap factor must be a power of 2, got %d", new_overlap);
    }
}

static void torch_rfft_tilde_verbose(t_torch_rfft_tilde *x, t_floatarg f) {
    x->rfft_processor_.set_verbose(static_cast<bool>(f));
    post("torch.rfft~: Verbose mode %s.", static_cast<bool>(f) ? "enabled" : "disabled");
}

// Construtor
static void *torch_rfft_tilde_new(t_symbol *s, int argc, t_atom *argv) {
    t_torch_rfft_tilde *x = (t_torch_rfft_tilde *)pd_new(torch_rfft_tilde_class);

    // Valores padrão
    x->current_block_size_ = 0;
    x->sampling_rate_ = sys_getsr() > 0 ? sys_getsr() : 44100.f;
    x->window_type_ = contorchionist::core::util_windowing::Type::RECTANGULAR;
    x->windowing_enabled_ = false;
    x->normalization_type_ = contorchionist::core::util_normalizations::NormalizationType::NONE;
    x->overlap_factor_ = 1;
    
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
    
    auto format = contorchionist::core::util_conversions::string_to_spectrum_data_format(parser.get_string("unit u format", "complex"));

    if (parser.has_flag("window win w")) {
        x->window_type_ = contorchionist::core::util_windowing::string_to_torch_window_type(parser.get_string("window win w", "rectangular"));
        x->windowing_enabled_ = true;
        x->normalization_type_ = contorchionist::core::util_normalizations::NormalizationType::WINDOW;
    }
    if (parser.has_flag("norm n")) {
        x->normalization_type_ = contorchionist::core::util_normalizations::string_to_normalization_type(parser.get_string("norm n", "none"), false);
    }
    x->overlap_factor_ = static_cast<int>(parser.get_float("overlap of", 1));

    // -> ADAPTADO: Construtor do RFFTProcessor com os valores iniciais.
    new (&x->rfft_processor_) contorchionist::core::ap_rfft::RFFTProcessor<float>(
        device,
        x->window_type_,
        x->windowing_enabled_,
        verbose_arg,
        x->normalization_type_,
        format,
        x->sampling_rate_
    );
    
    
    // Use pd_parse_and_set_torch_device to handle device parsing and logging
    torch::Device final_device = device;
    pd_parse_and_set_torch_device(&x->x_obj, final_device, device_arg_str, x->rfft_processor_.is_verbose(), "torch.rfft~", device_flag_present);
    x->rfft_processor_.set_device(final_device);
    
    // -> ADAPTADO: Criação de saídas simplificada.
    x->out1_ = outlet_new(&x->x_obj, &s_signal);
    x->out2_ = outlet_new(&x->x_obj, &s_signal);
    
    return (void *)x;
}

// Destrutor
static void torch_rfft_tilde_free(t_torch_rfft_tilde *x) {
    x->rfft_processor_.~RFFTProcessor<float>();
}

// Função de setup da classe
extern "C" void setup_torch0x2erfft_tilde(void) {
    torch_rfft_tilde_class = class_new(gensym("torch.rfft~"),
                                       (t_newmethod)torch_rfft_tilde_new,
                                       (t_method)torch_rfft_tilde_free,
                                       sizeof(t_torch_rfft_tilde),
                                       CLASS_DEFAULT, A_GIMME, 0);

    CLASS_MAINSIGNALIN(torch_rfft_tilde_class, t_torch_rfft_tilde, x_f);
    class_addmethod(torch_rfft_tilde_class, (t_method)torch_rfft_tilde_dsp, gensym("dsp"), A_CANT, 0);
    
    // -> ADAPTADO: Lista de métodos limpa e atualizada.
    class_addmethod(torch_rfft_tilde_class, (t_method)torch_rfft_tilde_window, gensym("window"), A_DEFSYMBOL, 0);
    class_addmethod(torch_rfft_tilde_class, (t_method)torch_rfft_tilde_norm, gensym("norm"), A_DEFSYMBOL, 0);
    class_addmethod(torch_rfft_tilde_class, (t_method)torch_rfft_tilde_unit, gensym("unit"), A_DEFSYMBOL, 0);
    class_addmethod(torch_rfft_tilde_class, (t_method)torch_rfft_tilde_window_enable, gensym("window_enable"), A_FLOAT, 0);
    class_addmethod(torch_rfft_tilde_class, (t_method)torch_rfft_tilde_overlap, gensym("overlap"), A_FLOAT, 0);
    class_addmethod(torch_rfft_tilde_class, (t_method)torch_rfft_tilde_verbose, gensym("verbose"), A_FLOAT, 0);

    class_sethelpsymbol(torch_rfft_tilde_class, gensym("torch.rfft~"));
}
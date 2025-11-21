#include "m_pd.h"
#include "core_ap_spectrails.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include <torch/torch.h>

#include "../utils/include/pd_arg_parser.h"
#include "../utils/include/pd_torch_device_adapter.h"

using Processor = contorchionist::core::ap_spectrails::SpectralTrailsProcessor<float>;

typedef struct _torch_amb_spectrails_tilde {
    t_object x_obj;
    t_sample x_f_dummy; // Dummy para CLASS_MAINSIGNALIN
    
    int ambi_order_; // Ordem ambisônica (1, 2, 3, etc.)
    int num_channels_; // B = (N+1)^2
    torch::Device device_; // Dispositivo de processamento (CPU/CUDA/MPS)
    
    // Um processador por canal (todos compartilham envelope do canal 0)
    std::vector<std::unique_ptr<Processor>> processors_;
    
    size_t fft_size_;
    size_t num_bins_;

    float threshold_;
    float attack_;
    float decay_;
    float min_peak_distance_hz_;
    int overlap_factor_;
    float pregain_db_;      // Ganho prévio em dB
    float pregain_linear_;  // Ganho prévio convertido para linear
    
    // Inlets e outlets dinâmicos
    std::vector<t_inlet*> inlets_;
    std::vector<t_outlet*> outlets_;
    
    // Estado do ambiente Pd
    int current_block_size_;
    float sampling_rate_;
} t_torch_amb_spectrails_tilde;

static t_class *torch_amb_spectrails_tilde_class = nullptr;

// Converte tempo de decaimento (em segundos para -6dB) em fator de decay
static float decaytime_to_decay_factor(float decay_time_sec, float sample_rate, size_t fft_size, int overlap_factor) {
    if (decay_time_sec <= 0.0f || sample_rate <= 0.0f || fft_size == 0 || overlap_factor <= 0) {
        return 0.999f;
    }
    
    float hopsize = static_cast<float>(fft_size) / static_cast<float>(overlap_factor);
    float update_rate_hz = sample_rate / hopsize;
    float num_frames = decay_time_sec * update_rate_hz;
    float decay_factor = std::pow(0.5f, 1.0f / num_frames);
    
    return decay_factor;
}

static void torch_amb_spectrails_tilde_configure_processors(t_torch_amb_spectrails_tilde *x) {
    for (auto& proc : x->processors_) {
        if (proc) {
            proc->set_threshold(x->threshold_);
            proc->set_attack(x->attack_);
            proc->set_decay(x->decay_);
            proc->set_min_peak_distance_hz(x->min_peak_distance_hz_, x->sampling_rate_);
        }
    }
}

static void torch_amb_spectrails_tilde_resize_processors(t_torch_amb_spectrails_tilde *x, size_t fft_size) {
    fft_size = std::max<size_t>(2, fft_size);
    x->fft_size_ = fft_size;
    x->num_bins_ = x->fft_size_ / 2 + 1;
    
    for (auto& proc : x->processors_) {
        if (proc) {
            proc->resize(x->num_bins_);
        }
    }
    torch_amb_spectrails_tilde_configure_processors(x);
}

static t_int *torch_amb_spectrails_tilde_perform(t_int *w) {
    // PD coloca a função em w[0] automaticamente
    // w[1] = x (objeto), w[2] = n (block size), w[3+] = signal pointers
    auto *x = reinterpret_cast<t_torch_amb_spectrails_tilde *>(w[1]);
    int n = static_cast<int>(w[2]);
    
    // Validação de segurança
    if (!x || x->num_channels_ <= 0 || x->num_channels_ > 64 || !x->processors_.size()) {
        // Retorna w + (número de args passados para dsp_addv + 1)
        // Args: x, n, (num_channels * 4 signals)
        int num_args = 2 + (x ? x->num_channels_ * 4 : 0);
        return w + num_args + 1;
    }
    
    // Ponteiros para buffers: mag0, phase0, mag1, phase1, ..., out_mag0, out_phase0, ...
    std::vector<t_sample*> in_mags;
    std::vector<t_sample*> in_phases;
    std::vector<t_sample*> out_mags;
    std::vector<t_sample*> out_phases;
    
    int offset = 3; // w[1]=x, w[2]=n, w[3...] = buffers
    
    // Reserva espaço para evitar realocações
    in_mags.reserve(x->num_channels_);
    in_phases.reserve(x->num_channels_);
    out_mags.reserve(x->num_channels_);
    out_phases.reserve(x->num_channels_);
    
    for (int ch = 0; ch < x->num_channels_; ++ch) {
        t_sample* mag_ptr = reinterpret_cast<t_sample*>(w[offset++]);
        t_sample* phase_ptr = reinterpret_cast<t_sample*>(w[offset++]);
        in_mags.push_back(mag_ptr);
        in_phases.push_back(phase_ptr);
    }
    for (int ch = 0; ch < x->num_channels_; ++ch) {
        t_sample* mag_ptr = reinterpret_cast<t_sample*>(w[offset++]);
        t_sample* phase_ptr = reinterpret_cast<t_sample*>(w[offset++]);
        out_mags.push_back(mag_ptr);
        out_phases.push_back(phase_ptr);
    }
    
    auto zero_outputs = [&]() {
        for (int ch = 0; ch < x->num_channels_; ++ch) {
            std::fill_n(out_mags[ch], n, 0.0f);
            std::fill_n(out_phases[ch], n, 0.0f);
        }
    };

    size_t num_bins = static_cast<size_t>(n) / 2 + 1;
    if (num_bins != x->num_bins_) {
        torch_amb_spectrails_tilde_resize_processors(x, static_cast<size_t>(n));
        num_bins = x->num_bins_;
    }

    try {
        // Processa canal 0 (W - omnidirecional) primeiro para detectar picos
        auto w_mag_tensor = torch::from_blob(in_mags[0],
                                             {static_cast<long>(num_bins)},
                                             torch::TensorOptions().dtype(torch::kFloat32))
                                 .clone();
        auto w_phase_tensor = torch::from_blob(in_phases[0],
                                               {static_cast<long>(num_bins)},
                                               torch::TensorOptions().dtype(torch::kFloat32))
                                   .clone();
        
        // Aplica pregain (apenas em magnitude, fase não muda)
        if (x->pregain_linear_ != 1.0f) {
            w_mag_tensor *= x->pregain_linear_;
        }

        // Processa W e detecta onde houve escrita de novos picos
        auto w_envelope_before = x->processors_[0]->get_envelope_positions();
        auto w_outputs = x->processors_[0]->process_frame(w_mag_tensor, w_phase_tensor);
        auto w_envelope_after = x->processors_[0]->get_envelope_positions();
        
        // Detecta bins onde envelope foi resetado (novos picos escritos)
        // Se envelope passou de >0.9 para ~0.0, houve escrita
        auto bins_written = (w_envelope_before > 0.9f) & (w_envelope_after < 0.1f);
        
        // Sincroniza envelope positions em todos os canais
        for (int ch = 1; ch < x->num_channels_; ++ch) {
            if (x->processors_[ch]) {
                x->processors_[ch]->set_envelope_positions(w_envelope_after);
            }
        }
        
        // Copia saída de W (move para CPU se necessário)
        auto w_mag_out = w_outputs[0].to(torch::kCPU).contiguous();
        auto w_phase_out = w_outputs[1].to(torch::kCPU).contiguous();
        std::memcpy(out_mags[0], w_mag_out.data_ptr<float>(), num_bins * sizeof(float));
        std::memcpy(out_phases[0], w_phase_out.data_ptr<float>(), num_bins * sizeof(float));
        
        // Processa outros canais (X, Y, Z, ...) e força escrita nos bins detectados em W
        for (int ch = 1; ch < x->num_channels_; ++ch) {
            auto mag_tensor = torch::from_blob(in_mags[ch],
                                               {static_cast<long>(num_bins)},
                                               torch::TensorOptions().dtype(torch::kFloat32))
                                   .clone();
            auto phase_tensor = torch::from_blob(in_phases[ch],
                                                 {static_cast<long>(num_bins)},
                                                 torch::TensorOptions().dtype(torch::kFloat32))
                                     .clone();
            
            // Aplica pregain (apenas em magnitude)
            if (x->pregain_linear_ != 1.0f) {
                mag_tensor *= x->pregain_linear_;
            }

            // CRÍTICO: Força escrita nos mesmos bins que W detectou
            if (bins_written.any().item<bool>()) {
                x->processors_[ch]->force_write_bins(bins_written.to(torch::kFloat32), 
                                                      mag_tensor, phase_tensor);
            }
            
            auto outputs = x->processors_[ch]->process_frame(mag_tensor, phase_tensor);
            auto processed_mag = outputs[0].to(torch::kCPU).contiguous();
            auto processed_phase = outputs[1].to(torch::kCPU).contiguous();
            
            std::memcpy(out_mags[ch], processed_mag.data_ptr<float>(), num_bins * sizeof(float));
            std::memcpy(out_phases[ch], processed_phase.data_ptr<float>(), num_bins * sizeof(float));
        }
        
        // Zera bins acima de Nyquist
        for (int ch = 0; ch < x->num_channels_; ++ch) {
            for (size_t i = num_bins; i < static_cast<size_t>(n); ++i) {
                out_mags[ch][i] = 0.0f;
                out_phases[ch][i] = 0.0f;
            }
        }
    } catch (const std::exception &e) {
        pd_error(x, "torch.amb.spectrails~: %s", e.what());
        zero_outputs();
    }

    return w + offset;
}

static void torch_amb_spectrails_tilde_dsp(t_torch_amb_spectrails_tilde *x, t_signal **sp) {
    post("torch.amb.spectrails~: === DSP SETUP BEGIN ===");
    post("torch.amb.spectrails~: block_size=%d, sr=%.0f, num_channels=%d", 
         sp[0]->s_n, sys_getsr(), x->num_channels_);
    
    // Validação crítica
    if (!x || !sp || x->num_channels_ <= 0) {
        pd_error(x, "torch.amb.spectrails~: invalid state in dsp()");
        return;
    }
    
    // Verifica se temos sinais suficientes
    int expected_signals = x->num_channels_ * 4; // 2 in + 2 out por canal
    post("torch.amb.spectrails~: expecting %d total signals (%d channels × 4)", 
         expected_signals, x->num_channels_);
    
    // Verifica cada sinal
    for (int i = 0; i < expected_signals && i < 32; i++) {
        if (sp[i]) {
            post("torch.amb.spectrails~: sp[%d] = %p (vec=%p, n=%d)", 
                 i, sp[i], sp[i]->s_vec, sp[i]->s_n);
        } else {
            post("torch.amb.spectrails~: sp[%d] = NULL!", i);
        }
    }
    
    bool block_size_changed = (x->current_block_size_ != sp[0]->s_n);
    bool sample_rate_changed = (x->sampling_rate_ != sys_getsr());

    if (block_size_changed) {
        x->current_block_size_ = sp[0]->s_n;
        size_t block_fft = static_cast<size_t>(sp[0]->s_n);
        if (block_fft != x->fft_size_) {
            torch_amb_spectrails_tilde_resize_processors(x, block_fft);
            post("torch.amb.spectrails~: resized to fftsize %zu", x->fft_size_);
        }
    }
    
    if (sample_rate_changed) {
        x->sampling_rate_ = sys_getsr();
        torch_amb_spectrails_tilde_configure_processors(x);
    }

    // CRÍTICO: NÃO incluir ponteiro da função no vetor!
    // PD adiciona a função em w[0] automaticamente
    // Vetor deve conter apenas: x, n, signal pointers
    std::vector<t_int> dsp_vec;
    dsp_vec.push_back(reinterpret_cast<t_int>(x));
    dsp_vec.push_back(static_cast<t_int>(sp[0]->s_n));
    
    post("torch.amb.spectrails~: preparing DSP chain: %d channels, %d signals", 
         x->num_channels_, x->num_channels_ * 4);
    
    // Entradas: mag0, phase0, mag1, phase1, ...
    // sp[] indexa todos os sinais conectados aos inlets/outlets
    // Total: num_channels_ * 2 inlets + num_channels_ * 2 outlets
    for (int ch = 0; ch < x->num_channels_; ++ch) {
        int sp_idx = ch * 2;
        if (sp_idx >= 0 && sp[sp_idx] && sp[sp_idx + 1]) {
            dsp_vec.push_back(reinterpret_cast<t_int>(sp[sp_idx]->s_vec));       // mag
            dsp_vec.push_back(reinterpret_cast<t_int>(sp[sp_idx + 1]->s_vec));   // phase
        } else {
            pd_error(x, "torch.amb.spectrails~: invalid input signal at channel %d", ch);
            return;
        }
    }
    
    // Saídas: out_mag0, out_phase0, out_mag1, out_phase1, ...
    int out_offset = x->num_channels_ * 2;
    for (int ch = 0; ch < x->num_channels_; ++ch) {
        int sp_idx = out_offset + ch * 2;
        if (sp_idx >= 0 && sp[sp_idx] && sp[sp_idx + 1]) {
            dsp_vec.push_back(reinterpret_cast<t_int>(sp[sp_idx]->s_vec));       // out_mag
            dsp_vec.push_back(reinterpret_cast<t_int>(sp[sp_idx + 1]->s_vec));   // out_phase
        } else {
            pd_error(x, "torch.amb.spectrails~: invalid output signal at channel %d", ch);
            return;
        }
    }
    
    post("torch.amb.spectrails~: calling dsp_addv with %zu args", dsp_vec.size());
    dsp_addv(torch_amb_spectrails_tilde_perform, static_cast<int>(dsp_vec.size()), dsp_vec.data());
    post("torch.amb.spectrails~: === DSP SETUP COMPLETE ===");
}

static void *torch_amb_spectrails_tilde_new(t_symbol *, int argc, t_atom *argv) {
    post("torch.amb.spectrails~: new() called with %d args", argc);
    
    auto *x = reinterpret_cast<t_torch_amb_spectrails_tilde *>(pd_new(torch_amb_spectrails_tilde_class));
    if (!x) {
        pd_error(nullptr, "torch.amb.spectrails~: failed to allocate object");
        return nullptr;
    }
    
    post("torch.amb.spectrails~: object allocated at %p", x);

    // Valores padrão
    x->current_block_size_ = 0;
    x->sampling_rate_ = sys_getsr() > 0 ? sys_getsr() : 48000.0f;
    post("torch.amb.spectrails~: sample rate = %.0f", x->sampling_rate_);
    x->threshold_ = 0.01f;
    x->attack_ = 0.7f;
    x->decay_ = 0.999f;
    x->min_peak_distance_hz_ = 100.0f;
    x->overlap_factor_ = 4;
    x->ambi_order_ = 1;
    x->pregain_db_ = 0.0f;
    x->pregain_linear_ = 1.0f;

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
    
    x->ambi_order_ = static_cast<int>(parser.get_float("order ord o", 1));
    if (x->ambi_order_ < 1) {
        pd_error(x, "torch.amb.spectrails~: order must be >= 1, using default 1");
        x->ambi_order_ = 1;
    }
    
    // TEMPORÁRIO: Força ordem 1 para debugging
    if (x->ambi_order_ > 1) {
        post("torch.amb.spectrails~: WARNING - forcing order=1 for debugging");
        x->ambi_order_ = 1;
    }
    
    x->num_channels_ = (x->ambi_order_ + 1) * (x->ambi_order_ + 1); // B = (N+1)^2
    post("torch.amb.spectrails~: order=%d, num_channels=%d", x->ambi_order_, x->num_channels_);
    
    x->threshold_ = parser.get_float("threshold thresh", 0.01f);
    x->attack_ = parser.get_float("attack att", 0.7f);
    x->overlap_factor_ = static_cast<int>(parser.get_float("overlap of", 4));
    
    post("torch.amb.spectrails~: params - thresh=%.3f, attack=%.3f, overlap=%d", 
         x->threshold_, x->attack_, x->overlap_factor_);
    
    float decaytime_sec = parser.get_float("decaytime decayt dtime", -1.0f);
    if (decaytime_sec > 0.0f) {
        size_t temp_fft_size = static_cast<size_t>(parser.get_float("fftsize fft n", 1024));
        x->decay_ = decaytime_to_decay_factor(decaytime_sec, x->sampling_rate_, temp_fft_size, x->overlap_factor_);
    } else {
        x->decay_ = parser.get_float("decay dec", 0.999f);
    }
    
    x->min_peak_distance_hz_ = parser.get_float("min_peak_distance mindist mpd", 100.0f);
    
    x->pregain_db_ = parser.get_float("pregaindb pregain", 0.0f);
    x->pregain_linear_ = std::pow(10.0f, x->pregain_db_ / 20.0f);
    
    x->fft_size_ = static_cast<size_t>(parser.get_float("fftsize fft n", 0));
    if (x->fft_size_ == 0) {
        x->fft_size_ = 1024;
    }
    
    x->num_bins_ = x->fft_size_ / 2 + 1;

    // Initialize device and apply device parsing
    x->device_ = device;
    torch::Device final_device = device;
    pd_parse_and_set_torch_device(&x->x_obj, final_device, device_arg_str, verbose_arg, "torch.amb.spectrails~", device_flag_present);
    x->device_ = final_device;
    
    // Cria processadores para cada canal
    post("torch.amb.spectrails~: creating %d processors...", x->num_channels_);
    try {
        for (int ch = 0; ch < x->num_channels_; ++ch) {
            x->processors_.push_back(std::make_unique<Processor>(x->num_bins_, x->device_));
        }
        post("torch.amb.spectrails~: processors created successfully");
        torch_amb_spectrails_tilde_configure_processors(x);
    } catch (const std::exception &e) {
        pd_error(x, "torch.amb.spectrails~: failed to create processors: %s", e.what());
        return nullptr;
    }

    // Cria inlets: mag0 (main), phase0, mag1, phase1, ...
    // O primeiro mag já existe (CLASS_MAINSIGNALIN), criamos os demais
    // Total de inlets a criar: (num_channels * 2) - 1
    // Ordem: phase0, mag1, phase1, mag2, phase2, ...
    
    post("torch.amb.spectrails~: creating %d inlets...", (x->num_channels_ * 2) - 1);
    
    // Primeiro inlet adicional: phase0
    x->inlets_.push_back(inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal));
    
    // Demais canais: mag e phase
    for (int ch = 1; ch < x->num_channels_; ++ch) {
        x->inlets_.push_back(inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal)); // mag
        x->inlets_.push_back(inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal)); // phase
    }
    
    post("torch.amb.spectrails~: inlets created, creating %d outlets...", x->num_channels_ * 2);
    
    // Cria outlets: out_mag0, out_phase0, out_mag1, out_phase1, ...
    for (int ch = 0; ch < x->num_channels_; ++ch) {
        x->outlets_.push_back(outlet_new(&x->x_obj, &s_signal));
        x->outlets_.push_back(outlet_new(&x->x_obj, &s_signal));
    }

    post("torch.amb.spectrails~: order=%d channels=%d (%.1fHz thresh, %.3f att, %.4f dec, overlap=%d) [%s] - READY", 
         x->ambi_order_, x->num_channels_, x->min_peak_distance_hz_, x->attack_, x->decay_, x->overlap_factor_,
         pd_torch_device_friendly_name(x->device_).c_str());
    
    return x;
}

static void torch_amb_spectrails_tilde_free(t_torch_amb_spectrails_tilde *x) {
    for (auto& inlet : x->inlets_) {
        if (inlet) inlet_free(inlet);
    }
    x->processors_.clear();
}

static void torch_amb_spectrails_tilde_threshold(t_torch_amb_spectrails_tilde *x, t_floatarg f) {
    x->threshold_ = f;
    torch_amb_spectrails_tilde_configure_processors(x);
}

static void torch_amb_spectrails_tilde_threshold_db(t_torch_amb_spectrails_tilde *x, t_floatarg f_db) {
    // Converte dB para amplitude linear com máxima resolução
    // threshold_linear = 10^(dB/20)
    // Exemplo: -60dB → 0.001, -120dB → 0.000001
    // Quanto mais negativo o dB, menor o threshold (mais sensível, sustenta mais)
    // Clamp para evitar problemas de precisão numérica abaixo de -140dB
    float db_clamped = std::max(f_db, -140.0f);
    x->threshold_ = std::pow(10.0f, db_clamped / 20.0f);
    
    if (f_db < -140.0f) {
        post("torch.amb.spectrails~: thresholddb clamped to -140dB (min) from %.1fdB", f_db);
    }
    
    torch_amb_spectrails_tilde_configure_processors(x);
}

static void torch_amb_spectrails_tilde_attack(t_torch_amb_spectrails_tilde *x, t_floatarg f) {
    x->attack_ = f;
    torch_amb_spectrails_tilde_configure_processors(x);
}

static void torch_amb_spectrails_tilde_decay(t_torch_amb_spectrails_tilde *x, t_floatarg f) {
    x->decay_ = f;
    torch_amb_spectrails_tilde_configure_processors(x);
}

static void torch_amb_spectrails_tilde_decaytime(t_torch_amb_spectrails_tilde *x, t_floatarg f) {
    if (f > 0.0f) {
        x->decay_ = decaytime_to_decay_factor(f, x->sampling_rate_, x->fft_size_, x->overlap_factor_);
        torch_amb_spectrails_tilde_configure_processors(x);
    } else {
        pd_error(x, "torch.amb.spectrails~: decaytime must be positive");
    }
}

static void torch_amb_spectrails_tilde_overlap(t_torch_amb_spectrails_tilde *x, t_floatarg f) {
    int new_overlap = static_cast<int>(f);
    if (new_overlap > 0) {
        x->overlap_factor_ = new_overlap;
    } else {
        pd_error(x, "torch.amb.spectrails~: overlap factor must be positive");
    }
}

static void torch_amb_spectrails_tilde_min_peak_distance(t_torch_amb_spectrails_tilde *x, t_floatarg f) {
    x->min_peak_distance_hz_ = f;
    torch_amb_spectrails_tilde_configure_processors(x);
}

static void torch_amb_spectrails_tilde_reset(t_torch_amb_spectrails_tilde *x) {
    for (auto& proc : x->processors_) {
        if (proc) {
            proc->reset_memory();
        }
    }
}

static void torch_amb_spectrails_tilde_pregaindb(t_torch_amb_spectrails_tilde *x, t_floatarg f_db) {
    x->pregain_db_ = f_db;
    x->pregain_linear_ = std::pow(10.0f, f_db / 20.0f);
    post("torch.amb.spectrails~: pregain set to %.1f dB (linear: %.6f)", f_db, x->pregain_linear_);
}

extern "C" void setup_torch0x2eamb0x2espectrails_tilde(void) {
    torch_amb_spectrails_tilde_class = class_new(gensym("torch.amb.spectrails~"),
                                                 reinterpret_cast<t_newmethod>(torch_amb_spectrails_tilde_new),
                                                 reinterpret_cast<t_method>(torch_amb_spectrails_tilde_free),
                                                 sizeof(t_torch_amb_spectrails_tilde),
                                                 CLASS_DEFAULT,
                                                 A_GIMME, 0);

    CLASS_MAINSIGNALIN(torch_amb_spectrails_tilde_class, t_torch_amb_spectrails_tilde, x_f_dummy);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_dsp),
                    gensym("dsp"), A_CANT, 0);

    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_threshold),
                    gensym("threshold"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_threshold),
                    gensym("thresh"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_threshold_db),
                    gensym("thresholddb"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_threshold_db),
                    gensym("threshdb"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_attack),
                    gensym("attack"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_attack),
                    gensym("att"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_decay),
                    gensym("decay"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_decay),
                    gensym("dec"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_decaytime),
                    gensym("decaytime"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_decaytime),
                    gensym("decayt"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_decaytime),
                    gensym("dtime"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_decaytime),
                    gensym("dt"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_overlap),
                    gensym("overlap"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_min_peak_distance),
                    gensym("min_peak_distance"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_reset),
                    gensym("reset"), A_NULL, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_pregaindb),
                    gensym("pregaindb"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_pregaindb),
                    gensym("pregain"), A_FLOAT, 0);
    
    post("torch.amb.spectrails~: ambisonic spectral trails processor");
}

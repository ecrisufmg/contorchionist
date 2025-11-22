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
    int detection_mode_;    // 0=slope, 1=prominence
    float prominence_threshold_; // Para modo prominence (padrão 0.6)
    int max_peaks_;         // Máximo de picos simultâneos (0 = ilimitado)
    float floor_db_;        // Piso de ruído para release (-1 = usar threshold)
    
    bool midi_mode_;        // Output freq as MIDI note
    int velocity_mode_;     // 0=off (dB), 1=log, 2=linear
    
    // Inlets e outlets dinâmicos
    std::vector<t_inlet*> inlets_;
    std::vector<t_outlet*> outlets_;
    t_outlet *info_outlet_; // Outlet para dados de controle
    t_clock *info_clock_;   // Clock para enviar dados de controle
    
    // Buffer para dados de picos (thread-safe copy)
    std::vector<contorchionist::core::ap_spectrails::PeakInfo> peak_data_buffer_;
    bool peak_data_ready_;
    
    // Estado do ambiente Pd
    int current_block_size_;
    float sampling_rate_;
} t_torch_amb_spectrails_tilde;

static t_class *torch_amb_spectrails_tilde_class = nullptr;

// Converte tempo de decaimento (em segundos para -6dB) em fator de decay
static float decay6db_to_decay_factor(float decay_time_sec, float sample_rate, size_t fft_size, int overlap_factor) {
    if (decay_time_sec <= 0.0f || sample_rate <= 0.0f || fft_size == 0 || overlap_factor <= 0) {
        return 0.999f;
    }
    
    float hopsize = static_cast<float>(fft_size) / static_cast<float>(overlap_factor);
    float update_rate_hz = sample_rate / hopsize;
    float num_frames = decay_time_sec * update_rate_hz;
    float decay_factor = std::pow(0.5f, 1.0f / num_frames);
    
    return decay_factor;
}

// Converte tempo de decaimento (em segundos para atingir floor_db) em fator de decay
static float decayfloor_to_decay_factor(float decay_time_sec, float floor_db, float sample_rate, size_t fft_size, int overlap_factor) {
    if (decay_time_sec <= 0.0f || sample_rate <= 0.0f || fft_size == 0 || overlap_factor <= 0) {
        return 0.0f; // Instant decay
    }
    
    // Se floor_db for muito alto ou positivo, assume um valor padrão razoável (-60dB)
    // para evitar comportamento estranho.
    float target_db = (floor_db >= -0.1f) ? -60.0f : floor_db;
    
    float hopsize = static_cast<float>(fft_size) / static_cast<float>(overlap_factor);
    float update_rate_hz = sample_rate / hopsize;
    float num_frames = decay_time_sec * update_rate_hz;
    
    // factor = 10^(target_db / (20 * num_frames))
    return std::pow(10.0f, target_db / (20.0f * num_frames));
}

// Converte tempo de ataque (em segundos) em fator de attack (incremento por frame)
static float attacktime_to_attack_factor(float attack_time_sec, float sample_rate, size_t fft_size, int overlap_factor) {
    if (attack_time_sec <= 0.0f || sample_rate <= 0.0f || fft_size == 0 || overlap_factor <= 0) {
        return 1.0f; // Instant attack
    }
    
    float hopsize = static_cast<float>(fft_size) / static_cast<float>(overlap_factor);
    float frame_duration = hopsize / sample_rate;
    
    // attack factor is added per frame to reach 1.0
    // num_frames = attack_time_sec / frame_duration
    // attack_factor = 1.0 / num_frames
    
    float num_frames = attack_time_sec / frame_duration;
    if (num_frames < 1.0f) return 1.0f;
    
    return 1.0f / num_frames;
}

static void torch_amb_spectrails_tilde_configure_processors(t_torch_amb_spectrails_tilde *x) {
    for (auto& proc : x->processors_) {
        if (proc) {
            proc->set_threshold(x->threshold_);
            proc->set_attack(x->attack_);
            proc->set_decay(x->decay_);
            proc->set_min_peak_distance_hz(x->min_peak_distance_hz_, x->sampling_rate_);
            proc->set_detection_mode(x->detection_mode_ == 0 ? 
                contorchionist::core::ap_spectrails::DetectionMode::SLOPE_BASED : 
                contorchionist::core::ap_spectrails::DetectionMode::PROMINENCE);
            proc->set_prominence_threshold(x->prominence_threshold_);
            proc->set_max_peaks(x->max_peaks_);
            
            // Converte floor_db para linear se definido
            if (x->floor_db_ > -140.0f) {
                proc->set_floor_threshold(std::pow(10.0f, x->floor_db_ / 20.0f));
            } else {
                proc->set_floor_threshold(-1.0f); // Sentinel
            }
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

static void torch_amb_spectrails_tilde_tick(t_torch_amb_spectrails_tilde *x) {
    if (x->peak_data_ready_) {
        const auto& peaks = x->peak_data_buffer_;
        
        for (const auto& peak : peaks) {
            t_atom argv[4];
            SETFLOAT(argv+0, static_cast<t_float>(peak.rank));
            
            // Freq / MIDI
            float freq_val = peak.freq_hz;
            if (x->midi_mode_) {
                // MIDI = 69 + 12 * log2(freq / 440)
                if (freq_val > 0) {
                    freq_val = 69.0f + 12.0f * std::log2(freq_val / 440.0f);
                } else {
                    freq_val = -1500.0f; // Or some low value
                }
            }
            SETFLOAT(argv+1, static_cast<t_float>(freq_val));
            
            // Mag / Velocity
            float mag_val = peak.mag_db;
            if (x->velocity_mode_ > 0) {
                // Velocity
                float velocity = 0.0f;
                if (x->velocity_mode_ == 1) { // Log
                    // V = 127 * 10^(dB/40)
                    velocity = 127.0f * std::pow(10.0f, mag_val / 40.0f);
                } else { // Linear
                    // V = 127 * (dB + 70) / 70
                    velocity = 127.0f * (mag_val + 70.0f) / 70.0f;
                }
                mag_val = velocity;
            }
            SETFLOAT(argv+2, static_cast<t_float>(mag_val));
            
            SETFLOAT(argv+3, static_cast<t_float>(peak.state));
            
            outlet_list(x->info_outlet_, &s_list, 4, argv);
        }
        
        x->peak_data_ready_ = false;
    }
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
        
        // Update peak info for control output
        // We only take info from channel 0 (W) as it drives the detection
        if (x->processors_[0]) {
            // Copy data to buffer (simple copy, assuming no race condition critical enough to crash)
            // In a strict real-time environment, we might want a lock-free queue, 
            // but for control data visualization, this is usually acceptable in PD externals
            // if the data size is small.
            x->peak_data_buffer_ = x->processors_[0]->get_latest_peaks();
            x->peak_data_ready_ = true;
            clock_delay(x->info_clock_, 0);
        }

    } catch (const std::exception &e) {
        pd_error(x, "torch.amb.spectrails~: %s", e.what());
        zero_outputs();
    }

    return w + offset;
}

static void torch_amb_spectrails_tilde_dsp(t_torch_amb_spectrails_tilde *x, t_signal **sp) {
    // Validação crítica
    if (!x || !sp || x->num_channels_ <= 0) {
        pd_error(x, "torch.amb.spectrails~: invalid state in dsp()");
        return;
    }
    
    bool block_size_changed = (x->current_block_size_ != sp[0]->s_n);
    bool sample_rate_changed = (x->sampling_rate_ != sys_getsr());

    if (block_size_changed) {
        x->current_block_size_ = sp[0]->s_n;
        size_t block_fft = static_cast<size_t>(sp[0]->s_n);
        if (block_fft != x->fft_size_) {
            torch_amb_spectrails_tilde_resize_processors(x, block_fft);
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
    
    dsp_addv(torch_amb_spectrails_tilde_perform, static_cast<int>(dsp_vec.size()), dsp_vec.data());
}

static void *torch_amb_spectrails_tilde_new(t_symbol *, int argc, t_atom *argv) {
    auto *x = reinterpret_cast<t_torch_amb_spectrails_tilde *>(pd_new(torch_amb_spectrails_tilde_class));
    if (!x) {
        pd_error(nullptr, "torch.amb.spectrails~: failed to allocate object");
        return nullptr;
    }
    
    // Valores padrão
    x->current_block_size_ = 0;
    x->sampling_rate_ = sys_getsr() > 0 ? sys_getsr() : 48000.0f;
    x->threshold_ = 0.01f;
    x->attack_ = 0.7f;
    x->decay_ = 0.999f;
    x->min_peak_distance_hz_ = 100.0f;
    x->overlap_factor_ = 4;
    x->ambi_order_ = 1;
    x->pregain_db_ = 0.0f;
    x->pregain_linear_ = 1.0f;
    x->detection_mode_ = 0; // 0=slope (padrão), 1=prominence
    x->prominence_threshold_ = 0.6f; // Sigmund~ usa 0.6 (PEAKTHRESHFACTOR)
    x->max_peaks_ = 0;
    x->floor_db_ = -150.0f; // Default to "unset" (below -140)
    x->peak_data_ready_ = false;

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
        x->ambi_order_ = 1;
    }
    
    x->num_channels_ = (x->ambi_order_ + 1) * (x->ambi_order_ + 1); // B = (N+1)^2
    
    x->threshold_ = parser.get_float("threshold thresh", 0.01f);
    x->attack_ = parser.get_float("attack att", 0.7f);
    x->overlap_factor_ = static_cast<int>(parser.get_float("overlap of", 4));
    
    float decaytime_sec = parser.get_float("decaytime decayt dtime", -1.0f);
    if (decaytime_sec > 0.0f) {
        size_t temp_fft_size = static_cast<size_t>(parser.get_float("fftsize fft n", 1024));
        x->decay_ = decay6db_to_decay_factor(decaytime_sec, x->sampling_rate_, temp_fft_size, x->overlap_factor_);
        post("torch.amb.spectrails~: warning: 'decaytime' is deprecated, use 'decay6dbs' or 'decays'");
    } else {
        // Check for new flags
        float decays = parser.get_float("decays", -1.0f);
        float decayms = parser.get_float("decayms", -1.0f);
        float decay6dbs = parser.get_float("decay6dbs", -1.0f);
        float decay6dbms = parser.get_float("decay6dbms", -1.0f);
        
        size_t temp_fft_size = static_cast<size_t>(parser.get_float("fftsize fft n", 1024));
        
        if (decays > 0.0f) {
            x->decay_ = decayfloor_to_decay_factor(decays, x->floor_db_, x->sampling_rate_, temp_fft_size, x->overlap_factor_);
        } else if (decayms > 0.0f) {
            x->decay_ = decayfloor_to_decay_factor(decayms / 1000.0f, x->floor_db_, x->sampling_rate_, temp_fft_size, x->overlap_factor_);
        } else if (decay6dbs > 0.0f) {
            x->decay_ = decay6db_to_decay_factor(decay6dbs, x->sampling_rate_, temp_fft_size, x->overlap_factor_);
        } else if (decay6dbms > 0.0f) {
            x->decay_ = decay6db_to_decay_factor(decay6dbms / 1000.0f, x->sampling_rate_, temp_fft_size, x->overlap_factor_);
        } else {
            x->decay_ = parser.get_float("decay dec", 0.999f);
        }
    }
    
    x->min_peak_distance_hz_ = parser.get_float("min_peak_distance mindist mpd", 100.0f);
    
    x->pregain_db_ = parser.get_float("pregaindb pregain", 0.0f);
    x->pregain_linear_ = std::pow(10.0f, x->pregain_db_ / 20.0f);
    
    x->detection_mode_ = static_cast<int>(parser.get_float("mode", 0.0f)); // 0=slope, 1=prominence
    x->prominence_threshold_ = parser.get_float("prominence prom", 0.6f);
    x->max_peaks_ = static_cast<int>(parser.get_float("max_peaks maxpeaks mp", 0.0f));
    x->floor_db_ = parser.get_float("floordb floor", -150.0f);
    
    x->midi_mode_ = parser.has_flag("midi m");
    
    x->velocity_mode_ = 0;
    if (parser.has_flag("velocity vel v")) {
        std::string vel_mode = parser.get_string("velocity vel v", "log");
        if (vel_mode == "linear" || vel_mode == "lin") {
            x->velocity_mode_ = 2;
        } else {
            x->velocity_mode_ = 1;
        }
    }
    
    float gain_db = parser.get_float("gaindb gain g", 0.0f);
    float gain_lin = std::pow(10.0f, gain_db / 20.0f);
    
    bool limiter_enable = true;
    if (parser.has_flag("nolimiter nolim")) {
        limiter_enable = false;
    }

    float limiter_thresh_db = parser.get_float("limiter lim l", 0.0f);
    float limiter_thresh_lin = std::pow(10.0f, limiter_thresh_db / 20.0f);
    
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
    try {
        for (int ch = 0; ch < x->num_channels_; ++ch) {
            x->processors_.push_back(std::make_unique<Processor>(x->num_bins_, x->device_));
        }
        
        // Apply initial gain and limiter settings
        for (auto& proc : x->processors_) {
            if (proc) {
                proc->set_output_gain(gain_lin);
                proc->set_limiter(limiter_enable, limiter_thresh_lin);
            }
        }
        
        torch_amb_spectrails_tilde_configure_processors(x);
    } catch (const std::exception &e) {
        pd_error(x, "torch.amb.spectrails~: failed to create processors: %s", e.what());
        return nullptr;
    }

    // Cria inlets: mag0 (main), phase0, mag1, phase1, ...
    // O primeiro mag já existe (CLASS_MAINSIGNALIN), criamos os demais
    // Total de inlets a criar: (num_channels * 2) - 1
    // Ordem: phase0, mag1, phase1, mag2, phase2, ...
    
    // Primeiro inlet adicional: phase0
    x->inlets_.push_back(inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal));
    
    // Demais canais: mag e phase
    for (int ch = 1; ch < x->num_channels_; ++ch) {
        x->inlets_.push_back(inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal)); // mag
        x->inlets_.push_back(inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal)); // phase
    }
    
    // Cria outlets: out_mag0, out_phase0, out_mag1, out_phase1, ...
    for (int ch = 0; ch < x->num_channels_; ++ch) {
        x->outlets_.push_back(outlet_new(&x->x_obj, &s_signal));
        x->outlets_.push_back(outlet_new(&x->x_obj, &s_signal));
    }
    
    // Extra outlet for control data (rightmost)
    x->info_outlet_ = outlet_new(&x->x_obj, &s_list);
    x->info_clock_ = clock_new(x, (t_method)torch_amb_spectrails_tilde_tick);

    post("torch.amb.spectrails~: order=%d channels=%d (%.1fHz thresh, %.3f att, %.4f dec, overlap=%d) [%s] - READY", 
         x->ambi_order_, x->num_channels_, x->min_peak_distance_hz_, x->attack_, x->decay_, x->overlap_factor_,
         pd_torch_device_friendly_name(x->device_).c_str());
    
    return x;
}

static void torch_amb_spectrails_tilde_free(t_torch_amb_spectrails_tilde *x) {
    if (x->info_clock_) clock_free(x->info_clock_);
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

static void torch_amb_spectrails_tilde_attackms(t_torch_amb_spectrails_tilde *x, t_floatarg f) {
    if (f >= 0.0f) {
        x->attack_ = attacktime_to_attack_factor(f / 1000.0f, x->sampling_rate_, x->fft_size_, x->overlap_factor_);
        torch_amb_spectrails_tilde_configure_processors(x);
    } else {
        pd_error(x, "torch.amb.spectrails~: attackms must be non-negative");
    }
}

static void torch_amb_spectrails_tilde_attacks(t_torch_amb_spectrails_tilde *x, t_floatarg f) {
    if (f >= 0.0f) {
        x->attack_ = attacktime_to_attack_factor(f, x->sampling_rate_, x->fft_size_, x->overlap_factor_);
        torch_amb_spectrails_tilde_configure_processors(x);
    } else {
        pd_error(x, "torch.amb.spectrails~: attacks must be non-negative");
    }
}

static void torch_amb_spectrails_tilde_decay(t_torch_amb_spectrails_tilde *x, t_floatarg f) {
    x->decay_ = f;
    torch_amb_spectrails_tilde_configure_processors(x);
}

static void torch_amb_spectrails_tilde_decayms(t_torch_amb_spectrails_tilde *x, t_floatarg f) {
    if (f > 0.0f) {
        x->decay_ = decayfloor_to_decay_factor(f / 1000.0f, x->floor_db_, x->sampling_rate_, x->fft_size_, x->overlap_factor_);
        torch_amb_spectrails_tilde_configure_processors(x);
    } else {
        pd_error(x, "torch.amb.spectrails~: decayms must be positive");
    }
}

static void torch_amb_spectrails_tilde_decays(t_torch_amb_spectrails_tilde *x, t_floatarg f) {
    if (f > 0.0f) {
        x->decay_ = decayfloor_to_decay_factor(f, x->floor_db_, x->sampling_rate_, x->fft_size_, x->overlap_factor_);
        torch_amb_spectrails_tilde_configure_processors(x);
    } else {
        pd_error(x, "torch.amb.spectrails~: decays must be positive");
    }
}

static void torch_amb_spectrails_tilde_decay6dbms(t_torch_amb_spectrails_tilde *x, t_floatarg f) {
    if (f > 0.0f) {
        x->decay_ = decay6db_to_decay_factor(f / 1000.0f, x->sampling_rate_, x->fft_size_, x->overlap_factor_);
        torch_amb_spectrails_tilde_configure_processors(x);
    } else {
        pd_error(x, "torch.amb.spectrails~: decay6dbms must be positive");
    }
}

static void torch_amb_spectrails_tilde_decay6dbs(t_torch_amb_spectrails_tilde *x, t_floatarg f) {
    if (f > 0.0f) {
        x->decay_ = decay6db_to_decay_factor(f, x->sampling_rate_, x->fft_size_, x->overlap_factor_);
        torch_amb_spectrails_tilde_configure_processors(x);
    } else {
        pd_error(x, "torch.amb.spectrails~: decay6dbs must be positive");
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
}

static void torch_amb_spectrails_tilde_mode(t_torch_amb_spectrails_tilde *x, t_floatarg f) {
    int mode = static_cast<int>(f);
    if (mode == 0 || mode == 1) {
        x->detection_mode_ = mode;
        torch_amb_spectrails_tilde_configure_processors(x);
    } else {
        pd_error(x, "torch.amb.spectrails~: mode must be 0 (slope) or 1 (prominence)");
    }
}

static void torch_amb_spectrails_tilde_prominence(t_torch_amb_spectrails_tilde *x, t_floatarg f) {
    x->prominence_threshold_ = std::clamp(f, 0.0f, 1.0f);
    torch_amb_spectrails_tilde_configure_processors(x);
}

static void torch_amb_spectrails_tilde_max_peaks(t_torch_amb_spectrails_tilde *x, t_floatarg f) {
    x->max_peaks_ = std::max(0, static_cast<int>(f));
    torch_amb_spectrails_tilde_configure_processors(x);
}

static void torch_amb_spectrails_tilde_floordb(t_torch_amb_spectrails_tilde *x, t_floatarg f) {
    x->floor_db_ = f;
    torch_amb_spectrails_tilde_configure_processors(x);
}

static void torch_amb_spectrails_tilde_gaindb(t_torch_amb_spectrails_tilde *x, t_floatarg f) {
    float gain_lin = std::pow(10.0f, f / 20.0f);
    for (auto& proc : x->processors_) {
        if (proc) proc->set_output_gain(gain_lin);
    }
}

static void torch_amb_spectrails_tilde_nolimiter(t_torch_amb_spectrails_tilde *x) {
    for (auto& proc : x->processors_) {
        if (proc) proc->set_limiter(false, 1.0f);
    }
}

static void torch_amb_spectrails_tilde_limiter(t_torch_amb_spectrails_tilde *x, t_symbol *s, int argc, t_atom *argv) {
    bool enable = true;
    float threshold_db = 0.0f;
    
    if (argc > 0) {
        if (argv[0].a_type == A_SYMBOL) {
            t_symbol* sym = atom_getsymbol(argv);
            if (sym == gensym("off") || sym == gensym("false") || sym == gensym("disable")) {
                enable = false;
            }
        } else {
            threshold_db = atom_getfloat(argv);
            enable = true;
        }
    }
    
    float threshold_lin = std::pow(10.0f, threshold_db / 20.0f);
    
    for (auto& proc : x->processors_) {
        if (proc) proc->set_limiter(enable, threshold_lin);
    }
}

static void torch_amb_spectrails_tilde_midi(t_torch_amb_spectrails_tilde *x, t_floatarg f) {
    x->midi_mode_ = (f != 0.0f);
}

static void torch_amb_spectrails_tilde_velocity(t_torch_amb_spectrails_tilde *x, t_symbol *s, int argc, t_atom *argv) {
    if (argc == 0) {
        x->velocity_mode_ = 1; // Default log
        return;
    }
    
    if (argv[0].a_type == A_SYMBOL) {
        t_symbol* sym = atom_getsymbol(argv);
        if (sym == gensym("linear") || sym == gensym("lin")) {
            x->velocity_mode_ = 2;
        } else if (sym == gensym("log") || sym == gensym("logarithmic")) {
            x->velocity_mode_ = 1;
        } else if (sym == gensym("off") || sym == gensym("none")) {
            x->velocity_mode_ = 0;
        }
    } else {
        float val = atom_getfloat(argv);
        x->velocity_mode_ = (val != 0.0f) ? 1 : 0;
    }
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
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_attackms),
                    gensym("attackms"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_attackms),
                    gensym("attms"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_attacks),
                    gensym("attacks"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_attacks),
                    gensym("atts"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_decay),
                    gensym("decay"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_decay),
                    gensym("dec"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_decayms),
                    gensym("decayms"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_decays),
                    gensym("decays"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_decay6dbms),
                    gensym("decay6dbms"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_decay6dbs),
                    gensym("decay6dbs"), A_FLOAT, 0);
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
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_mode),
                    gensym("mode"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_prominence),
                    gensym("prominence"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_max_peaks),
                    gensym("max_peaks"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_max_peaks),
                    gensym("maxpeaks"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_floordb),
                    gensym("floordb"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_floordb),
                    gensym("floor"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_gaindb),
                    gensym("gaindb"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_gaindb),
                    gensym("gain"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_limiter),
                    gensym("limiter"), A_GIMME, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_nolimiter),
                    gensym("nolimiter"), A_NULL, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_nolimiter),
                    gensym("nolim"), A_NULL, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_midi),
                    gensym("midi"), A_FLOAT, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_velocity),
                    gensym("velocity"), A_GIMME, 0);
    class_addmethod(torch_amb_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_amb_spectrails_tilde_velocity),
                    gensym("vel"), A_GIMME, 0);
    
    post("torch.amb.spectrails~: ambisonic spectral trails processor");
}

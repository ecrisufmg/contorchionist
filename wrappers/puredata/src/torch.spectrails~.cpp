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

typedef struct _torch_spectrails_tilde {
    t_object x_obj;
    t_sample x_f_dummy; // Dummy para CLASS_MAINSIGNALIN
    
    torch::Device device_; // Dispositivo de processamento (CPU/CUDA/MPS)
    std::unique_ptr<Processor> processor;

    size_t fft_size;
    size_t num_bins;

    float threshold;
    float attack;
    float decay;
    float min_peak_distance_hz;
    int overlap_factor_;
    float pregain_db_;      // Ganho prévio em dB
    float pregain_linear_;  // Ganho prévio convertido para linear
    int detection_mode_;    // 0=slope, 1=prominence
    float prominence_threshold_; // Para modo prominence (padrão 0.6)
    int max_peaks_;         // Máximo de picos simultâneos (0 = ilimitado)
    float floor_db_;        // Piso de ruído para release (-1 = usar threshold)
    
    bool midi_mode_;        // Output freq as MIDI note
    int velocity_mode_;     // 0=off (dB), 1=log, 2=linear
    
    // Outlets
    t_outlet *out_mag_;
    t_outlet *out_phase_;
    t_outlet *info_outlet_; // Outlet para dados de controle
    t_clock *info_clock_;   // Clock para enviar dados de controle
    
    // Buffer para dados de picos (thread-safe copy)
    std::vector<contorchionist::core::ap_spectrails::PeakInfo> peak_data_buffer_;
    bool peak_data_ready_;
    
    // Estado do ambiente Pd
    int current_block_size_;
    float sampling_rate_;

    // Stored time-based parameters for recalculation on resize
    float stored_attack_ms_;      // -1 if using factor directly
    float stored_decay_ms_;       // -1 if using factor directly (decay to floor)
    float stored_decay_6db_sec_;  // -1 if using factor directly (decay 6dB)
} t_torch_spectrails_tilde;

static t_class *torch_spectrails_tilde_class = nullptr;

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

static void torch_spectrails_tilde_configure_processor(t_torch_spectrails_tilde *x) {
    if (!x->processor) {
        return;
    }
    x->processor->set_threshold(x->threshold);
    x->processor->set_attack(x->attack);
    x->processor->set_decay(x->decay);
    x->processor->set_min_peak_distance_hz(x->min_peak_distance_hz, x->sampling_rate_);
    x->processor->set_detection_mode(x->detection_mode_ == 0 ? 
        contorchionist::core::ap_spectrails::DetectionMode::SLOPE_BASED : 
        contorchionist::core::ap_spectrails::DetectionMode::PROMINENCE);
    x->processor->set_prominence_threshold(x->prominence_threshold_);
    x->processor->set_max_peaks(x->max_peaks_);
    
    // Converte floor_db para linear se definido
    if (x->floor_db_ > -140.0f) {
        x->processor->set_floor_threshold(std::pow(10.0f, x->floor_db_ / 20.0f));
    } else {
        x->processor->set_floor_threshold(-1.0f); // Sentinel
    }
}

static void torch_spectrails_tilde_resize_processor(t_torch_spectrails_tilde *x, size_t fft_size) {
    fft_size = std::max<size_t>(2, fft_size);
    x->fft_size = fft_size;
    x->num_bins = x->fft_size / 2 + 1;

    // Recalculate time-based parameters if needed
    if (x->stored_attack_ms_ >= 0.0f) {
        x->attack = attacktime_to_attack_factor(x->stored_attack_ms_ / 1000.0f, x->sampling_rate_, x->fft_size, x->overlap_factor_);
    }

    if (x->stored_decay_ms_ > 0.0f) {
        x->decay = decayfloor_to_decay_factor(x->stored_decay_ms_ / 1000.0f, x->floor_db_, x->sampling_rate_, x->fft_size, x->overlap_factor_);
    } else if (x->stored_decay_6db_sec_ > 0.0f) {
        x->decay = decay6db_to_decay_factor(x->stored_decay_6db_sec_, x->sampling_rate_, x->fft_size, x->overlap_factor_);
    }

    if (x->processor) {
        x->processor->resize(x->num_bins);
        torch_spectrails_tilde_configure_processor(x);
    }
}

static void torch_spectrails_tilde_tick(t_torch_spectrails_tilde *x) {
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

static t_int *torch_spectrails_tilde_perform(t_int *w) {
    auto *x = reinterpret_cast<t_torch_spectrails_tilde *>(w[1]);
    t_sample *in_mag = reinterpret_cast<t_sample *>(w[2]);
    t_sample *in_phase = reinterpret_cast<t_sample *>(w[3]);
    t_sample *out_mag = reinterpret_cast<t_sample *>(w[4]);
    t_sample *out_phase = reinterpret_cast<t_sample *>(w[5]);
    int n = static_cast<int>(w[6]);

    if (!x->processor) {
        for (int i = 0; i < n; ++i) {
            out_mag[i] = 0.0f;
            out_phase[i] = 0.0f;
        }
        return w + 7;
    }

    size_t num_bins = static_cast<size_t>(n) / 2 + 1;
    if (num_bins != x->num_bins) {
        torch_spectrails_tilde_resize_processor(x, static_cast<size_t>(n));
        num_bins = x->num_bins;
    }

    try {
        auto mag_tensor = torch::from_blob(in_mag,
                                           {static_cast<long>(num_bins)},
                                           torch::TensorOptions().dtype(torch::kFloat32))
                               .clone();
        auto phase_tensor = torch::from_blob(in_phase,
                                             {static_cast<long>(num_bins)},
                                             torch::TensorOptions().dtype(torch::kFloat32))
                                 .clone();

        // Aplica pregain
        if (x->pregain_linear_ != 1.0f) {
            mag_tensor *= x->pregain_linear_;
        }

        auto outputs = x->processor->process_frame(mag_tensor, phase_tensor);
        auto processed_mag = outputs[0].to(torch::kCPU).contiguous();
        auto processed_phase = outputs[1].to(torch::kCPU).contiguous();

        std::memcpy(out_mag, processed_mag.data_ptr<float>(), num_bins * sizeof(float));
        std::memcpy(out_phase, processed_phase.data_ptr<float>(), num_bins * sizeof(float));

        // Zero bins above Nyquist
        for (size_t i = num_bins; i < static_cast<size_t>(n); ++i) {
            out_mag[i] = 0.0f;
            out_phase[i] = 0.0f;
        }

        // Update peak info for control output
        x->peak_data_buffer_ = x->processor->get_latest_peaks();
        x->peak_data_ready_ = true;
        clock_delay(x->info_clock_, 0);

    } catch (const std::exception &e) {
        pd_error(x, "torch.spectrails~: %s", e.what());
        for (int i = 0; i < n; ++i) {
            out_mag[i] = 0.0f;
            out_phase[i] = 0.0f;
        }
    }

    return w + 7;
}

static void torch_spectrails_tilde_dsp(t_torch_spectrails_tilde *x, t_signal **sp) {
    bool block_size_changed = (x->current_block_size_ != sp[0]->s_n);
    bool sample_rate_changed = (x->sampling_rate_ != sys_getsr());

    if (block_size_changed) {
        x->current_block_size_ = sp[0]->s_n;
        size_t block_fft = static_cast<size_t>(sp[0]->s_n);
        if (block_fft != x->fft_size) {
            torch_spectrails_tilde_resize_processor(x, block_fft);
        }
    }
    
    if (sample_rate_changed) {
        x->sampling_rate_ = sys_getsr();
        torch_spectrails_tilde_configure_processor(x);
    }

    dsp_add(torch_spectrails_tilde_perform, 6, x,
            sp[0]->s_vec,
            sp[1]->s_vec,
            sp[2]->s_vec,
            sp[3]->s_vec,
            sp[0]->s_n);
}

static void *torch_spectrails_tilde_new(t_symbol *, int argc, t_atom *argv) {
    auto *x = reinterpret_cast<t_torch_spectrails_tilde *>(pd_new(torch_spectrails_tilde_class));
    if (!x) {
        return nullptr;
    }

    // Valores padrão
    x->current_block_size_ = 0;
    x->sampling_rate_ = sys_getsr() > 0 ? sys_getsr() : 48000.0f;
    x->threshold = 0.01f;
    x->attack = 0.7f;
    x->decay = 0.999f;
    x->min_peak_distance_hz = 100.0f;
    x->overlap_factor_ = 4;
    x->pregain_db_ = 0.0f;
    x->pregain_linear_ = 1.0f;
    x->detection_mode_ = 0;
    x->prominence_threshold_ = 0.6f;
    x->max_peaks_ = 0;
    x->floor_db_ = -150.0f;
    x->peak_data_ready_ = false;
    
    x->stored_attack_ms_ = -1.0f;
    x->stored_decay_ms_ = -1.0f;
    x->stored_decay_6db_sec_ = -1.0f;

    // Parser de argumentos
    pd_utils::ArgParser parser(argc, argv, &x->x_obj);
    bool verbose_arg = parser.has_flag("verbose v");
    
    // Parse device
    bool device_flag_present = parser.has_flag("device d");
    std::string device_arg_str = parser.get_string("device d", "cpu");
    auto device_result = get_device_from_string(device_arg_str);
    torch::Device device = device_result.first;
    
    // Initial FFT size based on system block size (will be updated in DSP)
    x->fft_size = static_cast<size_t>(sys_getblksize());
    if (x->fft_size == 0) x->fft_size = 64; // Fallback
    x->num_bins = x->fft_size / 2 + 1;

    x->threshold = parser.get_float("threshold thresh t", 0.01f);
    
    // Attack parsing
    if (parser.has_flag("attackms attms")) {
        float attms = parser.get_float("attackms attms", 0.0f);
        x->stored_attack_ms_ = attms;
        x->attack = attacktime_to_attack_factor(attms / 1000.0f, x->sampling_rate_, x->fft_size, x->overlap_factor_);
    } else if (parser.has_flag("attacks atts")) {
        float atts = parser.get_float("attacks atts", 0.0f);
        x->stored_attack_ms_ = atts * 1000.0f;
        x->attack = attacktime_to_attack_factor(atts, x->sampling_rate_, x->fft_size, x->overlap_factor_);
    } else {
        x->attack = parser.get_float("attack att a", 0.7f);
    }

    x->overlap_factor_ = static_cast<int>(parser.get_float("overlap of", 4));
    
    float decaytime_sec = parser.get_float("decaytime decayt dtime dt", -1.0f);
    if (decaytime_sec > 0.0f) {
        // Legacy support
        x->stored_decay_6db_sec_ = decaytime_sec;
        x->decay = decay6db_to_decay_factor(decaytime_sec, x->sampling_rate_, x->fft_size, x->overlap_factor_);
        post("torch.spectrails~: warning: 'decaytime' is deprecated, use 'decay6dbs' or 'decays'");
    } else {
        // Check for new flags
        float decays = parser.get_float("decays", -1.0f);
        float decayms = parser.get_float("decayms", -1.0f);
        float decay6dbs = parser.get_float("decay6dbs", -1.0f);
        float decay6dbms = parser.get_float("decay6dbms", -1.0f);
        
        if (decays > 0.0f) {
            x->stored_decay_ms_ = decays * 1000.0f;
            x->decay = decayfloor_to_decay_factor(decays, x->floor_db_, x->sampling_rate_, x->fft_size, x->overlap_factor_);
        } else if (decayms > 0.0f) {
            x->stored_decay_ms_ = decayms;
            x->decay = decayfloor_to_decay_factor(decayms / 1000.0f, x->floor_db_, x->sampling_rate_, x->fft_size, x->overlap_factor_);
        } else if (decay6dbs > 0.0f) {
            x->stored_decay_6db_sec_ = decay6dbs;
            x->decay = decay6db_to_decay_factor(decay6dbs, x->sampling_rate_, x->fft_size, x->overlap_factor_);
        } else if (decay6dbms > 0.0f) {
            x->stored_decay_6db_sec_ = decay6dbms / 1000.0f;
            x->decay = decay6db_to_decay_factor(decay6dbms / 1000.0f, x->sampling_rate_, x->fft_size, x->overlap_factor_);
        } else {
            x->decay = parser.get_float("decay dec d", 0.999f);
        }
    }
    
    x->min_peak_distance_hz = parser.get_float("min_peak_distance mindist mpd", 100.0f);
    
    x->pregain_db_ = parser.get_float("pregaindb pregain", 0.0f);
    x->pregain_linear_ = std::pow(10.0f, x->pregain_db_ / 20.0f);
    
    x->detection_mode_ = static_cast<int>(parser.get_float("mode", 0.0f));
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

    // Initialize device
    x->device_ = device;
    torch::Device final_device = device;
    pd_parse_and_set_torch_device(&x->x_obj, final_device, device_arg_str, verbose_arg, "torch.spectrails~", device_flag_present);
    x->device_ = final_device;

    try {
        x->processor = std::make_unique<Processor>(x->num_bins, x->device_);
        
        if (x->processor) {
            x->processor->set_output_gain(gain_lin);
            x->processor->set_limiter(limiter_enable, limiter_thresh_lin);
        }
        
        torch_spectrails_tilde_configure_processor(x);
    } catch (const std::exception &e) {
        pd_error(x, "torch.spectrails~: failed to create processor: %s", e.what());
        return nullptr;
    }

    inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal); // Phase inlet
    x->out_mag_ = outlet_new(&x->x_obj, &s_signal);
    x->out_phase_ = outlet_new(&x->x_obj, &s_signal);
    x->info_outlet_ = outlet_new(&x->x_obj, &s_list);
    x->info_clock_ = clock_new(x, (t_method)torch_spectrails_tilde_tick);

    post("torch.spectrails~: threshold=%.4f attack=%.3f decay=%.4f min_peak_dist=%.1fHz overlap=%d [%s]", 
         x->threshold, x->attack, x->decay, x->min_peak_distance_hz, x->overlap_factor_,
         pd_torch_device_friendly_name(x->device_).c_str());
    return x;
}

static void torch_spectrails_tilde_free(t_torch_spectrails_tilde *x) {
    if (x->info_clock_) clock_free(x->info_clock_);
    x->processor.reset();
}

static void torch_spectrails_tilde_threshold(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->threshold = f;
    torch_spectrails_tilde_configure_processor(x);
}

static void torch_spectrails_tilde_threshold_db(t_torch_spectrails_tilde *x, t_floatarg f_db) {
    float db_clamped = std::max(f_db, -140.0f);
    x->threshold = std::pow(10.0f, db_clamped / 20.0f);
    torch_spectrails_tilde_configure_processor(x);
}

static void torch_spectrails_tilde_attack(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->attack = f;
    x->stored_attack_ms_ = -1.0f; // Reset time-based tracking
    torch_spectrails_tilde_configure_processor(x);
}

static void torch_spectrails_tilde_attackms(t_torch_spectrails_tilde *x, t_floatarg f) {
    if (f >= 0.0f) {
        x->stored_attack_ms_ = f;
        x->attack = attacktime_to_attack_factor(f / 1000.0f, x->sampling_rate_, x->fft_size, x->overlap_factor_);
        torch_spectrails_tilde_configure_processor(x);
    } else {
        pd_error(x, "torch.spectrails~: attackms must be non-negative");
    }
}

static void torch_spectrails_tilde_attacks(t_torch_spectrails_tilde *x, t_floatarg f) {
    if (f >= 0.0f) {
        x->stored_attack_ms_ = f * 1000.0f;
        x->attack = attacktime_to_attack_factor(f, x->sampling_rate_, x->fft_size, x->overlap_factor_);
        torch_spectrails_tilde_configure_processor(x);
    } else {
        pd_error(x, "torch.spectrails~: attacks must be non-negative");
    }
}

static void torch_spectrails_tilde_decay(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->decay = f;
    x->stored_decay_ms_ = -1.0f; // Reset time-based tracking
    x->stored_decay_6db_sec_ = -1.0f;
    torch_spectrails_tilde_configure_processor(x);
}

static void torch_spectrails_tilde_min_peak_distance(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->min_peak_distance_hz = f;
    torch_spectrails_tilde_configure_processor(x);
}

static void torch_spectrails_tilde_decayms(t_torch_spectrails_tilde *x, t_floatarg f) {
    if (f > 0.0f) {
        x->stored_decay_ms_ = f;
        x->stored_decay_6db_sec_ = -1.0f;
        x->decay = decayfloor_to_decay_factor(f / 1000.0f, x->floor_db_, x->sampling_rate_, x->fft_size, x->overlap_factor_);
        torch_spectrails_tilde_configure_processor(x);
    } else {
        pd_error(x, "torch.spectrails~: decayms must be positive");
    }
}

static void torch_spectrails_tilde_decays(t_torch_spectrails_tilde *x, t_floatarg f) {
    if (f > 0.0f) {
        x->stored_decay_ms_ = f * 1000.0f;
        x->stored_decay_6db_sec_ = -1.0f;
        x->decay = decayfloor_to_decay_factor(f, x->floor_db_, x->sampling_rate_, x->fft_size, x->overlap_factor_);
        torch_spectrails_tilde_configure_processor(x);
    } else {
        pd_error(x, "torch.spectrails~: decays must be positive");
    }
}

static void torch_spectrails_tilde_decay6dbms(t_torch_spectrails_tilde *x, t_floatarg f) {
    if (f > 0.0f) {
        x->stored_decay_6db_sec_ = f / 1000.0f;
        x->stored_decay_ms_ = -1.0f;
        x->decay = decay6db_to_decay_factor(f / 1000.0f, x->sampling_rate_, x->fft_size, x->overlap_factor_);
        torch_spectrails_tilde_configure_processor(x);
    } else {
        pd_error(x, "torch.spectrails~: decay6dbms must be positive");
    }
}

static void torch_spectrails_tilde_decay6dbs(t_torch_spectrails_tilde *x, t_floatarg f) {
    if (f > 0.0f) {
        x->stored_decay_6db_sec_ = f;
        x->stored_decay_ms_ = -1.0f;
        x->decay = decay6db_to_decay_factor(f, x->sampling_rate_, x->fft_size, x->overlap_factor_);
        torch_spectrails_tilde_configure_processor(x);
    } else {
        pd_error(x, "torch.spectrails~: decay6dbs must be positive");
    }
}

static void torch_spectrails_tilde_overlap(t_torch_spectrails_tilde *x, t_floatarg f) {
    int new_overlap = static_cast<int>(f);
    if (new_overlap > 0) {
        x->overlap_factor_ = new_overlap;
    } else {
        pd_error(x, "torch.spectrails~: overlap factor must be positive");
    }
}

static void torch_spectrails_tilde_reset(t_torch_spectrails_tilde *x) {
    if (x->processor) {
        x->processor->reset_memory();
    }
}

static void torch_spectrails_tilde_pregaindb(t_torch_spectrails_tilde *x, t_floatarg f_db) {
    x->pregain_db_ = f_db;
    x->pregain_linear_ = std::pow(10.0f, f_db / 20.0f);
}

static void torch_spectrails_tilde_mode(t_torch_spectrails_tilde *x, t_floatarg f) {
    int mode = static_cast<int>(f);
    if (mode == 0 || mode == 1) {
        x->detection_mode_ = mode;
        torch_spectrails_tilde_configure_processor(x);
    } else {
        pd_error(x, "torch.spectrails~: mode must be 0 (slope) or 1 (prominence)");
    }
}

static void torch_spectrails_tilde_prominence(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->prominence_threshold_ = std::clamp(f, 0.0f, 1.0f);
    torch_spectrails_tilde_configure_processor(x);
}

static void torch_spectrails_tilde_max_peaks(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->max_peaks_ = std::max(0, static_cast<int>(f));
    torch_spectrails_tilde_configure_processor(x);
}

static void torch_spectrails_tilde_floordb(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->floor_db_ = f;
    torch_spectrails_tilde_configure_processor(x);
}

static void torch_spectrails_tilde_gaindb(t_torch_spectrails_tilde *x, t_floatarg f) {
    float gain_lin = std::pow(10.0f, f / 20.0f);
    if (x->processor) x->processor->set_output_gain(gain_lin);
}

static void torch_spectrails_tilde_nolimiter(t_torch_spectrails_tilde *x) {
    if (x->processor) x->processor->set_limiter(false, 1.0f);
}

static void torch_spectrails_tilde_limiter(t_torch_spectrails_tilde *x, t_symbol *s, int argc, t_atom *argv) {
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
    if (x->processor) x->processor->set_limiter(enable, threshold_lin);
}

static void torch_spectrails_tilde_midi(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->midi_mode_ = (f != 0.0f);
}

static void torch_spectrails_tilde_velocity(t_torch_spectrails_tilde *x, t_symbol *s, int argc, t_atom *argv) {
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

extern "C" void setup_torch0x2espectrails_tilde(void) {
    torch_spectrails_tilde_class = class_new(gensym("torch.spectrails~"),
                                             reinterpret_cast<t_newmethod>(torch_spectrails_tilde_new),
                                             reinterpret_cast<t_method>(torch_spectrails_tilde_free),
                                             sizeof(t_torch_spectrails_tilde),
                                             CLASS_DEFAULT,
                                             A_GIMME, 0);

    CLASS_MAINSIGNALIN(torch_spectrails_tilde_class, t_torch_spectrails_tilde, x_f_dummy);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_dsp),
                    gensym("dsp"), A_CANT, 0);

    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_threshold),
                    gensym("threshold"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_threshold),
                    gensym("thresh"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_threshold_db),
                    gensym("thresholddb"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_threshold_db),
                    gensym("threshdb"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_attack),
                    gensym("attack"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_attack),
                    gensym("att"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_attackms),
                    gensym("attackms"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_attackms),
                    gensym("attms"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_attacks),
                    gensym("attacks"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_attacks),
                    gensym("atts"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_decay),
                    gensym("decay"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_decay),
                    gensym("dec"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_decayms),
                    gensym("decayms"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_decays),
                    gensym("decays"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_decay6dbms),
                    gensym("decay6dbms"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_decay6dbs),
                    gensym("decay6dbs"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_overlap),
                    gensym("overlap"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_min_peak_distance),
                    gensym("min_peak_distance"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_reset),
                    gensym("reset"), A_NULL, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_pregaindb),
                    gensym("pregaindb"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_pregaindb),
                    gensym("pregain"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_mode),
                    gensym("mode"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_prominence),
                    gensym("prominence"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_max_peaks),
                    gensym("max_peaks"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_max_peaks),
                    gensym("maxpeaks"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_floordb),
                    gensym("floordb"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_floordb),
                    gensym("floor"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_gaindb),
                    gensym("gaindb"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_gaindb),
                    gensym("gain"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_limiter),
                    gensym("limiter"), A_GIMME, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_nolimiter),
                    gensym("nolimiter"), A_NULL, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_nolimiter),
                    gensym("nolim"), A_NULL, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_midi),
                    gensym("midi"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_velocity),
                    gensym("velocity"), A_GIMME, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_velocity),
                    gensym("vel"), A_GIMME, 0);
    
    post("torch.spectrails~: spectral trails processor");
}

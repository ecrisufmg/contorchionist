#include "m_pd.h"
#include "core_ap_spectrails.h"
#include "../utils/include/pd_arg_parser.h"

#include <memory>
#include <vector>
#include <exception>
#include <cmath>

// Define the C-style struct for the Pd tilde object
typedef struct _torch_spectrails_tilde {
    t_object x_obj;
    t_float x_f; // Dummy float for signal inlet (magnitude)
    t_float x_f_phase; // Dummy float for phase inlet

    std::unique_ptr<contorchionist::core::ap_spectrails::SpectralTrailsProcessor<float>> processor;

    t_outlet* p_phase_outlet;

    // Parameters stored in the Pd object
    float p_threshold;
    float p_attack;
    float p_phase_attack;
    float p_decay;
    float p_decay6db;
    bool p_limiter_enabled;
    float p_max_value;
    float p_limiter_softness;
    float p_sample_rate;
    float p_hop_size;
    float p_attack_adapt;
    float p_attack_onset;
    float p_onset_floor;
    float p_onset_hysteresis_ms;
    int p_onset_hysteresis_frames;
    bool p_onset_hysteresis_use_time;
    float p_attack_onset_ramp_ms;
    int p_attack_onset_ramp_frames;
    bool p_attack_onset_ramp_use_time;
    int p_reset_frames;
    float p_reset_multiplier;
    bool p_use_decay6db;
    bool p_phase_attack_locked;
    size_t p_num_bins; // N/2 + 1
    size_t p_fft_size; // Full FFT size (N)

} t_torch_spectrails_tilde;

// Declare the class
static t_class *torch_spectrails_tilde_class;

// --- Method Prototypes ---
static void *torch_spectrails_tilde_new(t_symbol *s, int argc, t_atom *argv);
static void torch_spectrails_tilde_free(t_torch_spectrails_tilde *x);
static void torch_spectrails_tilde_dsp(t_torch_spectrails_tilde *x, t_signal **sp);
static t_int *torch_spectrails_tilde_perform(t_int *w);

// --- Setter Methods ---
static void torch_spectrails_tilde_threshold(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_attack(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_phaseattack(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_decay(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_decay6db(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_decay6b(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_limiter(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_limitersoft(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_maxvalue(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_reset(t_torch_spectrails_tilde *x);
static void torch_spectrails_tilde_hopsize(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_attackadapt(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_attackonset(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_onsetfloor(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_onsethyst(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_onsethystframes(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_attackonsetramp(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_attackonsetrampframes(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_resetframes(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_resetmult(t_torch_spectrails_tilde *x, t_floatarg f);
static void torch_spectrails_tilde_frametiming(t_torch_spectrails_tilde *x, t_floatarg sr, t_floatarg hop);
static void torch_spectrails_tilde_update_frame_timing(t_torch_spectrails_tilde *x);

static void torch_spectrails_tilde_update_frame_timing(t_torch_spectrails_tilde *x) {
    if (!x->processor || x->p_hop_size <= 0.0f) {
        return;
    }

    x->processor->set_frame_timing(x->p_sample_rate, x->p_hop_size);

    if (x->p_onset_hysteresis_use_time) {
        x->processor->set_onset_hysteresis_time(x->p_onset_hysteresis_ms * 0.001f);
    } else {
        x->processor->set_onset_hysteresis_frames(x->p_onset_hysteresis_frames);
    }

    if (x->p_attack_onset_ramp_use_time) {
        x->processor->set_attack_onset_ramp_time(x->p_attack_onset_ramp_ms * 0.001f);
    } else {
        x->processor->set_attack_onset_ramp_frames(x->p_attack_onset_ramp_frames);
    }
}


// --- DSP Routine ---
static t_int *torch_spectrails_tilde_perform(t_int *w) {
    t_torch_spectrails_tilde *x = (t_torch_spectrails_tilde *)(w[1]);
    t_sample *in_mag = (t_sample *)(w[2]);
    t_sample *in_phase = (t_sample *)(w[3]);
    t_sample *out_mag = (t_sample *)(w[4]);
    t_sample *out_phase = (t_sample *)(w[5]);
    int n = (int)(w[6]); // Block size (should be FFT size)

    if (!x->processor) {
        // If processor isn't ready, zero the outputs
        for (int i = 0; i < n; ++i) {
            out_mag[i] = 0.0f;
            out_phase[i] = 0.0f;
        }
        return (w + 7);
    }

    try {
        // Calculate number of useful bins (N/2 + 1)
        size_t num_bins = n / 2 + 1;

        // Create torch tensors from input buffers (CPU)
        auto mag_tensor = torch::from_blob(
            in_mag, 
            {static_cast<long>(num_bins)}, 
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone(); // Clone to avoid aliasing issues

        auto phase_tensor = torch::from_blob(
            in_phase, 
            {static_cast<long>(num_bins)}, 
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone();

        // Process through the spectral trails processor
        auto output_tensors = x->processor->process_frame(mag_tensor, phase_tensor);
        
        auto& processed_mag = output_tensors[0];
        auto& processed_phase = output_tensors[1];

        // Ensure outputs are on CPU
        processed_mag = processed_mag.to(torch::kCPU);
        processed_phase = processed_phase.to(torch::kCPU);

        // Copy processed bins to output buffers
        auto mag_accessor = processed_mag.accessor<float, 1>();
        auto phase_accessor = processed_phase.accessor<float, 1>();

        for (size_t i = 0; i < num_bins; ++i) {
            out_mag[i] = mag_accessor[i];
            out_phase[i] = phase_accessor[i];
        }

        // Zero out the rest of the output buffers (redundant bins)
        for (size_t i = num_bins; i < static_cast<size_t>(n); ++i) {
            out_mag[i] = 0.0f;
            out_phase[i] = 0.0f;
        }

    } catch (const std::exception& e) {
        pd_error(x, "torch.spectrails~: Exception in perform: %s", e.what());
        
        // On error, zero outputs
        for (int i = 0; i < n; ++i) {
            out_mag[i] = 0.0f;
            out_phase[i] = 0.0f;
        }
    }

    return (w + 7);
}


// --- DSP Setup ---
static void torch_spectrails_tilde_dsp(t_torch_spectrails_tilde *x, t_signal **sp) {
    // The block size inside pfft~ corresponds to the FFT size
    size_t pfft_size = sp[0]->s_n;

    // Check if the FFT size from pfft~ matches the configured size
    if (x->p_fft_size != pfft_size) {
        post("torch.spectrails~: FFT size mismatch! Object configured with fftsize %zu, but pfft~ is running at %zu.", 
             x->p_fft_size, pfft_size);
        post("torch.spectrails~: Resizing internal processor to match pfft~.");
        
        x->p_fft_size = pfft_size;
        x->p_num_bins = x->p_fft_size / 2 + 1;
        
        if (x->processor) {
            x->processor->resize(x->p_num_bins);
        }
    }

    bool sr_changed = false;

    float env_sr = sp[0]->s_sr > 0 ? sp[0]->s_sr : sys_getsr();
    if (env_sr > 0.0f && std::abs(env_sr - x->p_sample_rate) > 1e-3f) {
        x->p_sample_rate = env_sr;
        sr_changed = true;
    }

    bool hop_was_invalid = false;
    if (x->p_hop_size <= 0.0f) {
        x->p_hop_size = static_cast<float>(x->p_fft_size);
        hop_was_invalid = true;
    }

    if (x->p_use_decay6db && x->processor && (sr_changed || hop_was_invalid)) {
        x->processor->set_decay_time_s(x->p_decay6db, x->p_sample_rate, x->p_hop_size);
        x->p_decay = x->processor->get_decay();
        post("torch.spectrails~: decay factor recalculated from decay6db (timing update).");
    }

    torch_spectrails_tilde_update_frame_timing(x);

    dsp_add(torch_spectrails_tilde_perform, 6, x, 
            sp[0]->s_vec,  // in_mag
            sp[1]->s_vec,  // in_phase
            sp[2]->s_vec,  // out_mag
            sp[3]->s_vec,  // out_phase
            sp[0]->s_n);
}

// --- Object Lifecycle ---

static void torch_spectrails_tilde_free(t_torch_spectrails_tilde *x) {
    // unique_ptr handles deletion automatically
}

static void *torch_spectrails_tilde_new(t_symbol *s, int argc, t_atom *argv) {
    t_torch_spectrails_tilde *x = (t_torch_spectrails_tilde *)pd_new(torch_spectrails_tilde_class);
    if (!x) {
        return nullptr;
    }

    // Argument Parsing
    pd_utils::ArgParser parser(argc, argv, (t_object*)x);

    // Get FFT size (default 1024)
    x->p_fft_size = static_cast<size_t>(parser.get_float("fftsize", 1024.0f));
    x->p_num_bins = x->p_fft_size / 2 + 1;

    // Parse parameters with defaults
    x->p_threshold = parser.get_float("threshold", 0.01f);
    x->p_attack = parser.get_float("attack", 0.8f);
    x->p_phase_attack = parser.get_float("phaseattack phase_attack", x->p_attack);
    x->p_decay = parser.get_float("decay", 0.999f);
    x->p_decay6db = parser.get_float("decay6db", -1.0f);
    x->p_limiter_enabled = static_cast<bool>(parser.get_float("limiter", 0.0f));
    x->p_max_value = parser.get_float("maxvalue", 1.0f);
    x->p_limiter_softness = parser.get_float("limitersoft limitersmooth", 0.0f);
    x->p_hop_size = parser.get_float("hopsize hop", static_cast<float>(x->p_fft_size));
    if (x->p_hop_size <= 0.0f) {
        x->p_hop_size = static_cast<float>(x->p_fft_size);
    }

    if (x->p_attack < 0.0f) {
        x->p_attack = 0.0f;
    } else if (x->p_attack > 1.0f) {
        x->p_attack = 1.0f;
    }

    if (x->p_phase_attack < 0.0f) {
        x->p_phase_attack = 0.0f;
    } else if (x->p_phase_attack > 1.0f) {
        x->p_phase_attack = 1.0f;
    }

    if (x->p_limiter_softness < 0.0f) {
        x->p_limiter_softness = 0.0f;
    }

    x->p_sample_rate = sys_getsr();
    if (x->p_sample_rate <= 0.0f) {
        x->p_sample_rate = 48000.0f;
    }
    x->p_use_decay6db = false;
    x->p_attack_adapt = parser.get_float("attackadapt attackcurve", 0.0f);
    if (x->p_attack_adapt < 0.0f) {
        x->p_attack_adapt = 0.0f;
    }
    x->p_attack_onset = parser.get_float("attackonset attack_onset onsetattack", 0.95f);
    if (x->p_attack_onset >= 0.0f) {
        if (x->p_attack_onset > 1.0f) {
            x->p_attack_onset = 1.0f;
        }
    }
    x->p_onset_floor = parser.get_float("onsetfloor onset_floor", 1e-3f);
    if (x->p_onset_floor < 0.0f) {
        x->p_onset_floor = 0.0f;
    }
    x->p_onset_hysteresis_ms = parser.get_float("onsethyst onset_hyst", 0.0f);
    if (x->p_onset_hysteresis_ms < 0.0f) {
        x->p_onset_hysteresis_ms = 0.0f;
    }
    x->p_onset_hysteresis_frames = static_cast<int>(parser.get_float("onsethystframes onset_hyst_frames", 1.0f));
    if (x->p_onset_hysteresis_frames < 1) {
        x->p_onset_hysteresis_frames = 1;
    }
    x->p_onset_hysteresis_use_time = x->p_onset_hysteresis_ms > 0.0f;
    x->p_attack_onset_ramp_ms = parser.get_float("attackonsetramp attack_onset_ramp", 0.0f);
    if (x->p_attack_onset_ramp_ms < 0.0f) {
        x->p_attack_onset_ramp_ms = 0.0f;
    }
    x->p_attack_onset_ramp_frames = static_cast<int>(parser.get_float("attackonsetrampframes attack_onset_ramp_frames", 0.0f));
    if (x->p_attack_onset_ramp_frames < 0) {
        x->p_attack_onset_ramp_frames = 0;
    }
    x->p_attack_onset_ramp_use_time = x->p_attack_onset_ramp_ms > 0.0f;
    x->p_reset_frames = static_cast<int>(parser.get_float("resetframes reset_frames", 0.0f));
    if (x->p_reset_frames < 0) {
        x->p_reset_frames = 0;
    }
    x->p_reset_multiplier = parser.get_float("resetmult reset_multiplier", 0.5f);
    if (x->p_reset_multiplier < 0.0f) {
        x->p_reset_multiplier = 0.0f;
    } else if (x->p_reset_multiplier > 1.0f) {
        x->p_reset_multiplier = 1.0f;
    }
    x->p_phase_attack_locked = parser.has_flag("phaseattack phase_attack");

    // Instantiate the C++ processor
    try {
        x->processor = std::make_unique<contorchionist::core::ap_spectrails::SpectralTrailsProcessor<float>>(
            x->p_num_bins,
            torch::kCPU,
            false // verbose
        );

        // Configure processor with parsed parameters
        x->processor->set_threshold(x->p_threshold);
        x->processor->set_attack(x->p_attack);
        if (x->p_phase_attack_locked) {
            x->processor->set_phase_attack(x->p_phase_attack);
        }
        x->processor->set_decay(x->p_decay);
        x->processor->set_limiter_enabled(x->p_limiter_enabled);
        x->processor->set_max_value(x->p_max_value);
        x->processor->set_limiter_softness(x->p_limiter_softness);
        x->processor->set_attack_dynamic_rate(x->p_attack_adapt);
        x->processor->set_attack_onset(x->p_attack_onset);
        x->processor->set_onset_floor(x->p_onset_floor);
        x->processor->set_reset_frames(x->p_reset_frames);
        x->processor->set_reset_multiplier(x->p_reset_multiplier);

        torch_spectrails_tilde_update_frame_timing(x);

        if (x->p_decay6db > 0.0f) {
            x->processor->set_decay_time_s(x->p_decay6db, x->p_sample_rate, x->p_hop_size);
            x->p_decay = x->processor->get_decay();
            x->p_use_decay6db = true;
        }

    } catch (const std::exception& e) {
        pd_error(x, "torch.spectrails~: Exception during processor creation: %s", e.what());
        return nullptr;
    }

    // Create inlets: signal inlet for phase (magnitude is default)
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
    
    // Create outlets: magnitude and phase
    outlet_new(&x->x_obj, &s_signal);
    x->p_phase_outlet = outlet_new(&x->x_obj, &s_signal);

    // Log creation with all parameters
    post("torch.spectrails~: ===== Object Created =====");
    post("  FFT Size: %zu", x->p_fft_size);
    post("  Num Bins: %zu", x->p_num_bins);
    post("  Sample Rate: %.2f", x->p_sample_rate);
    post("  Hop Size: %.2f", x->p_hop_size);
    post("  Threshold: %.6f", x->p_threshold);
    post("  Attack: %.6f", x->p_attack);
    post("  Phase Attack: %.6f", x->p_phase_attack_locked ? x->p_phase_attack : x->p_attack);
    post("  Attack Adapt: %.6f", x->p_attack_adapt);
    if (x->p_attack_onset < 0.0f) {
        post("  Attack Onset: disabled");
    } else {
        post("  Attack Onset: %.6f", x->p_attack_onset);
    }
    if (x->p_onset_hysteresis_use_time) {
        post("  Onset Hysteresis: %.3f ms", x->p_onset_hysteresis_ms);
    } else {
        post("  Onset Hysteresis (frames): %d", x->p_onset_hysteresis_frames);
    }
    if (x->p_attack_onset_ramp_use_time) {
        post("  Attack Onset Ramp: %.3f ms", x->p_attack_onset_ramp_ms);
    } else {
        post("  Attack Onset Ramp (frames): %d", x->p_attack_onset_ramp_frames);
    }
    post("  Onset Floor: %.6f", x->p_onset_floor);
    post("  Reset Frames: %d", x->p_reset_frames);
    post("  Reset Multiplier: %.6f", x->p_reset_multiplier);
    post("  Decay: %.6f", x->p_decay);
    if (x->p_use_decay6db) {
        post("  Decay6dB: %.6f s", x->p_decay6db);
    }
    post("  Limiter: %s", x->p_limiter_enabled ? "ON" : "OFF");
    post("  Max Value: %.6f", x->p_max_value);
    post("  Limiter Softness: %.6f", x->p_limiter_softness);
    post("=====================================");

    return (void *)x;
}

// --- Setters ---

static void torch_spectrails_tilde_threshold(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->p_threshold = f;
    if (x->processor) {
        x->processor->set_threshold(x->p_threshold);
        post("torch.spectrails~: threshold set to %.6f", x->p_threshold);
    }
}

static void torch_spectrails_tilde_attack(t_torch_spectrails_tilde *x, t_floatarg f) {
    if (f < 0.0f) {
        f = 0.0f;
    } else if (f > 1.0f) {
        f = 1.0f;
    }
    x->p_attack = f;
    if (x->processor) {
        x->processor->set_attack(x->p_attack);
        if (!x->p_phase_attack_locked) {
            x->p_phase_attack = x->p_attack;
        }
        post("torch.spectrails~: attack set to %.6f", x->p_attack);
    }
}

static void torch_spectrails_tilde_decay(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->p_decay = f;
    if (x->processor) {
        x->processor->set_decay(x->p_decay);
        x->p_use_decay6db = false;
        x->p_decay6db = -1.0f;
        post("torch.spectrails~: decay set to %.6f", x->p_decay);
    }
}

static void torch_spectrails_tilde_decay6db(t_torch_spectrails_tilde *x, t_floatarg f) {
    if (f <= 0.0f) {
        pd_error(x, "torch.spectrails~: decay6db requires a positive time in seconds");
        return;
    }

    x->p_decay6db = f;
    x->p_use_decay6db = true;
    if (x->processor) {
        x->processor->set_decay_time_s(x->p_decay6db, x->p_sample_rate, x->p_hop_size);
        x->p_decay = x->processor->get_decay();
        post("torch.spectrails~: decay6db set to %.6f s (decay factor=%.6f)", x->p_decay6db, x->p_decay);
    }
}

static void torch_spectrails_tilde_decay6b(t_torch_spectrails_tilde *x, t_floatarg f) {
    post("torch.spectrails~: alias 'decay6b' treated as 'decay6db'");
    torch_spectrails_tilde_decay6db(x, f);
}

static void torch_spectrails_tilde_phaseattack(t_torch_spectrails_tilde *x, t_floatarg f) {
    if (f < 0.0f) {
        f = 0.0f;
    } else if (f > 1.0f) {
        f = 1.0f;
    }

    x->p_phase_attack = f;
    x->p_phase_attack_locked = true;
    if (x->processor) {
        x->processor->set_phase_attack(x->p_phase_attack);
        post("torch.spectrails~: phase attack set to %.6f", x->p_phase_attack);
    }
}

static void torch_spectrails_tilde_limiter(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->p_limiter_enabled = (f != 0.0f);
    if (x->processor) {
        x->processor->set_limiter_enabled(x->p_limiter_enabled);
        post("torch.spectrails~: limiter %s", x->p_limiter_enabled ? "enabled" : "disabled");
    }
}

static void torch_spectrails_tilde_limitersoft(t_torch_spectrails_tilde *x, t_floatarg f) {
    if (f < 0.0f) {
        f = 0.0f;
    }

    x->p_limiter_softness = f;
    if (x->processor) {
        x->processor->set_limiter_softness(x->p_limiter_softness);
        post("torch.spectrails~: limiter softness set to %.6f", x->p_limiter_softness);
    }
}

static void torch_spectrails_tilde_maxvalue(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->p_max_value = f;
    if (x->processor) {
        x->processor->set_max_value(x->p_max_value);
        post("torch.spectrails~: max value set to %.6f", x->p_max_value);
    }
}

static void torch_spectrails_tilde_reset(t_torch_spectrails_tilde *x) {
    if (x->processor) {
        x->processor->reset_memory();
        post("torch.spectrails~: memory reset");
    }
}

static void torch_spectrails_tilde_hopsize(t_torch_spectrails_tilde *x, t_floatarg f) {
    if (f <= 0.0f) {
        pd_error(x, "torch.spectrails~: hopsize requires a positive value");
        return;
    }

    x->p_hop_size = f;
    post("torch.spectrails~: hopsize set to %.2f", x->p_hop_size);
    if (x->p_use_decay6db && x->processor) {
        x->processor->set_decay_time_s(x->p_decay6db, x->p_sample_rate, x->p_hop_size);
        x->p_decay = x->processor->get_decay();
        post("torch.spectrails~: decay factor recalculated from decay6db after hopsize update (%.6f)", x->p_decay);
    }
    torch_spectrails_tilde_update_frame_timing(x);
}

static void torch_spectrails_tilde_attackadapt(t_torch_spectrails_tilde *x, t_floatarg f) {
    if (f < 0.0f) {
        f = 0.0f;
    }

    x->p_attack_adapt = f;
    if (x->processor) {
        x->processor->set_attack_dynamic_rate(x->p_attack_adapt);
        post("torch.spectrails~: attackadapt set to %.6f", x->p_attack_adapt);
    }
}

static void torch_spectrails_tilde_attackonset(t_torch_spectrails_tilde *x, t_floatarg f) {
    if (f < 0.0f) {
        x->p_attack_onset = -1.0f;
        if (x->processor) {
            x->processor->set_attack_onset(-1.0f);
            post("torch.spectrails~: attackonset disabled");
        }
        return;
    }

    if (f > 1.0f) {
        f = 1.0f;
    }

    x->p_attack_onset = f;
    if (x->processor) {
        x->processor->set_attack_onset(x->p_attack_onset);
        post("torch.spectrails~: attackonset set to %.6f", x->p_attack_onset);
    }
}

static void torch_spectrails_tilde_onsetfloor(t_torch_spectrails_tilde *x, t_floatarg f) {
    if (f < 0.0f) {
        f = 0.0f;
    }

    x->p_onset_floor = f;
    if (x->processor) {
        x->processor->set_onset_floor(x->p_onset_floor);
        post("torch.spectrails~: onsetfloor set to %.6f", x->p_onset_floor);
    }
}

static void torch_spectrails_tilde_onsethyst(t_torch_spectrails_tilde *x, t_floatarg f) {
    if (f <= 0.0f) {
        x->p_onset_hysteresis_ms = 0.0f;
        x->p_onset_hysteresis_use_time = false;
        x->p_onset_hysteresis_frames = 1;
        if (x->processor) {
            x->processor->set_onset_hysteresis_frames(x->p_onset_hysteresis_frames);
            post("torch.spectrails~: onsethyst disabled (falling back to 1 frame)");
        }
        return;
    }

    x->p_onset_hysteresis_ms = f;
    x->p_onset_hysteresis_use_time = true;
    if (x->processor) {
        x->processor->set_onset_hysteresis_time(x->p_onset_hysteresis_ms * 0.001f);
        post("torch.spectrails~: onsethyst set to %.3f ms", x->p_onset_hysteresis_ms);
    }
}

static void torch_spectrails_tilde_onsethystframes(t_torch_spectrails_tilde *x, t_floatarg f) {
    int frames = static_cast<int>(f);
    if (frames < 1) {
        frames = 1;
    }

    x->p_onset_hysteresis_frames = frames;
    x->p_onset_hysteresis_use_time = false;
    if (x->processor) {
        x->processor->set_onset_hysteresis_frames(x->p_onset_hysteresis_frames);
        post("torch.spectrails~: onsethystframes set to %d", x->p_onset_hysteresis_frames);
    }
}

static void torch_spectrails_tilde_attackonsetramp(t_torch_spectrails_tilde *x, t_floatarg f) {
    if (f <= 0.0f) {
        x->p_attack_onset_ramp_ms = 0.0f;
        x->p_attack_onset_ramp_use_time = false;
        x->p_attack_onset_ramp_frames = 0;
        if (x->processor) {
            x->processor->set_attack_onset_ramp_frames(0);
            post("torch.spectrails~: attackonsetramp disabled");
        }
        return;
    }

    x->p_attack_onset_ramp_ms = f;
    x->p_attack_onset_ramp_use_time = true;
    if (x->processor) {
        x->processor->set_attack_onset_ramp_time(x->p_attack_onset_ramp_ms * 0.001f);
        post("torch.spectrails~: attackonsetramp set to %.3f ms", x->p_attack_onset_ramp_ms);
    }
}

static void torch_spectrails_tilde_attackonsetrampframes(t_torch_spectrails_tilde *x, t_floatarg f) {
    int frames = static_cast<int>(f);
    if (frames < 0) {
        frames = 0;
    }

    x->p_attack_onset_ramp_frames = frames;
    x->p_attack_onset_ramp_use_time = false;
    if (x->processor) {
        x->processor->set_attack_onset_ramp_frames(x->p_attack_onset_ramp_frames);
        post("torch.spectrails~: attackonsetrampframes set to %d", x->p_attack_onset_ramp_frames);
    }
}

static void torch_spectrails_tilde_resetframes(t_torch_spectrails_tilde *x, t_floatarg f) {
    int frames = static_cast<int>(f);
    if (frames < 0) {
        frames = 0;
    }

    x->p_reset_frames = frames;
    if (x->processor) {
        x->processor->set_reset_frames(x->p_reset_frames);
        post("torch.spectrails~: resetframes set to %d", x->p_reset_frames);
    }
}

static void torch_spectrails_tilde_resetmult(t_torch_spectrails_tilde *x, t_floatarg f) {
    if (f < 0.0f) {
        f = 0.0f;
    } else if (f > 1.0f) {
        f = 1.0f;
    }

    x->p_reset_multiplier = f;
    if (x->processor) {
        x->processor->set_reset_multiplier(x->p_reset_multiplier);
        post("torch.spectrails~: resetmult set to %.6f", x->p_reset_multiplier);
    }
}

static void torch_spectrails_tilde_frametiming(t_torch_spectrails_tilde *x, t_floatarg sr, t_floatarg hop) {
    bool sr_changed = false;
    bool hop_changed = false;

    float requested_sr = static_cast<float>(sr);
    if (requested_sr > 0.0f && std::abs(requested_sr - x->p_sample_rate) > 1e-3f) {
        x->p_sample_rate = requested_sr;
        sr_changed = true;
    } else if (requested_sr <= 0.0f) {
        float env_sr = sys_getsr();
        if (env_sr > 0.0f && std::abs(env_sr - x->p_sample_rate) > 1e-3f) {
            x->p_sample_rate = env_sr;
            sr_changed = true;
        }
    }

    float requested_hop = static_cast<float>(hop);
    if (requested_hop > 0.0f) {
        x->p_hop_size = requested_hop;
        hop_changed = true;
    }

    if (x->p_hop_size <= 0.0f) {
        x->p_hop_size = static_cast<float>(x->p_fft_size);
        hop_changed = true;
    }

    if (x->p_use_decay6db && x->processor && (sr_changed || hop_changed)) {
        x->processor->set_decay_time_s(x->p_decay6db, x->p_sample_rate, x->p_hop_size);
        x->p_decay = x->processor->get_decay();
    }

    torch_spectrails_tilde_update_frame_timing(x);

    if (x->processor && (sr_changed || hop_changed)) {
        post("torch.spectrails~: frametiming updated (fs=%.2f Hz, hop=%.2f)", x->p_sample_rate, x->p_hop_size);
    }
}

// --- PD Class Setup --
extern "C" void setup_torch0x2espectrails_tilde(void) {
        torch_spectrails_tilde_class = class_new(gensym("torch.spectrails~"),
                                                 (t_newmethod)torch_spectrails_tilde_new,
                                                 (t_method)torch_spectrails_tilde_free,
                                                 sizeof(t_torch_spectrails_tilde),
                                                 CLASS_DEFAULT,
                                                 A_GIMME, 0);

        CLASS_MAINSIGNALIN(torch_spectrails_tilde_class, t_torch_spectrails_tilde, x_f);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_dsp, 
                       gensym("dsp"), A_CANT, 0);

        // Setter methods
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_threshold, 
                       gensym("threshold"), A_FLOAT, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_attack, 
                       gensym("attack"), A_FLOAT, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_phaseattack, 
                   gensym("phaseattack"), A_FLOAT, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_phaseattack, 
                   gensym("phase_attack"), A_FLOAT, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_decay, 
                       gensym("decay"), A_FLOAT, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_decay6db, 
                   gensym("decay6db"), A_FLOAT, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_decay6b, 
               gensym("decay6b"), A_FLOAT, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_limiter, 
                       gensym("limiter"), A_FLOAT, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_limitersoft, 
                   gensym("limitersoft"), A_FLOAT, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_limitersoft, 
                   gensym("limitersmooth"), A_FLOAT, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_maxvalue, 
                       gensym("maxvalue"), A_FLOAT, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_reset, 
                       gensym("reset"), A_NULL, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_hopsize, 
                   gensym("hopsize"), A_FLOAT, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_hopsize, 
               gensym("hop"), A_FLOAT, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_attackadapt, 
               gensym("attackadapt"), A_FLOAT, 0);
        class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_attackadapt, 
               gensym("attackcurve"), A_FLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_attackonset, 
             gensym("attackonset"), A_FLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_attackonset, 
             gensym("attack_onset"), A_FLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_attackonset, 
             gensym("onsetattack"), A_FLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_onsetfloor, 
             gensym("onsetfloor"), A_FLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_onsetfloor, 
             gensym("onset_floor"), A_FLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_onsethyst, 
             gensym("onsethyst"), A_FLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_onsethyst, 
             gensym("onset_hyst"), A_FLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_onsethystframes, 
             gensym("onsethystframes"), A_FLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_onsethystframes, 
             gensym("onset_hyst_frames"), A_FLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_attackonsetramp, 
             gensym("attackonsetramp"), A_FLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_attackonsetramp, 
             gensym("attack_onset_ramp"), A_FLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_attackonsetrampframes, 
             gensym("attackonsetrampframes"), A_FLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_attackonsetrampframes, 
             gensym("attack_onset_ramp_frames"), A_FLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_resetframes, 
             gensym("resetframes"), A_FLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_resetframes, 
             gensym("reset_frames"), A_FLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_resetmult, 
             gensym("resetmult"), A_FLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_resetmult, 
             gensym("resetmultiplier"), A_FLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_resetmult, 
             gensym("reset_multiplier"), A_FLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_frametiming, 
             gensym("frametiming"), A_DEFFLOAT, A_DEFFLOAT, 0);
         class_addmethod(torch_spectrails_tilde_class, (t_method)torch_spectrails_tilde_frametiming, 
             gensym("frame_timing"), A_DEFFLOAT, A_DEFFLOAT, 0);

        post("torch.spectrails~: Spectral Trails Processor v1.0");
        post("  Use with torch.rfft~ and torch.irfft~ inside pfft~");
        post("  Parameters: @threshold @attack @phaseattack @attackadapt @attackonset @attackonsetramp @onsetfloor @onsethyst @resetframes @resetmult @decay @decay6db @limiter @limitersoft @maxvalue @hopsize");
    
}

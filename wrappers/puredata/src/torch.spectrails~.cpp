#include "m_pd.h"
#include "core_ap_spectrails.h"

#include <algorithm>
#include <memory>
#include <torch/torch.h>

#include "../utils/include/pd_arg_parser.h"

using Processor = contorchionist::core::ap_spectrails::SpectralTrailsProcessor<float>;

typedef struct _torch_spectrails_tilde {
    t_object x_obj;
    t_sample x_f_dummy; // Dummy para CLASS_MAINSIGNALIN
    std::unique_ptr<Processor> processor;

    size_t fft_size;
    size_t num_bins;

    float threshold;
    float attack;
    float decay;
    float min_peak_distance_hz;
    int overlap_factor_; // Fator de overlap para cálculo de decaytime
    
    // Estado do ambiente Pd
    int current_block_size_;
    float sampling_rate_;
} t_torch_spectrails_tilde;

static t_class *torch_spectrails_tilde_class = nullptr;

// Converte tempo de decaimento (em segundos para -6dB) em fator de decay
// decay_time_sec: tempo em segundos para o sinal decair 6dB
// sample_rate: taxa de amostragem em Hz
// fft_size: tamanho da FFT
// overlap_factor: fator de overlap (hopsize = fft_size / overlap_factor)
// Retorna: fator de decay por frame (0-1)
static float decaytime_to_decay_factor(float decay_time_sec, float sample_rate, size_t fft_size, int overlap_factor) {
    if (decay_time_sec <= 0.0f || sample_rate <= 0.0f || fft_size == 0 || overlap_factor <= 0) {
        return 0.999f; // Valor padrão seguro
    }
    
    // Hopsize = número de amostras entre frames
    float hopsize = static_cast<float>(fft_size) / static_cast<float>(overlap_factor);
    
    // Taxa de atualização em Hz (frames por segundo)
    float update_rate_hz = sample_rate / hopsize;
    
    // Número de frames no tempo de decaimento
    float num_frames = decay_time_sec * update_rate_hz;
    
    // Para decair 6dB (amplitude *= 0.5), precisamos: amplitude(t) = amplitude(0) * decay^num_frames = 0.5
    // Portanto: decay = 0.5^(1/num_frames)
    float decay_factor = std::pow(0.5f, 1.0f / num_frames);
    
    return decay_factor;
}

static void torch_spectrails_tilde_configure_processor(t_torch_spectrails_tilde *x) {
    if (!x->processor) {
        return;
    }
    x->processor->set_threshold(x->threshold);
    x->processor->set_attack(x->attack);
    x->processor->set_decay(x->decay);
    x->processor->set_min_peak_distance_hz(x->min_peak_distance_hz, x->sampling_rate_);
}

static void torch_spectrails_tilde_resize_processor(t_torch_spectrails_tilde *x, size_t fft_size) {
    fft_size = std::max<size_t>(2, fft_size);
    x->fft_size = fft_size;
    x->num_bins = x->fft_size / 2 + 1;
    if (x->processor) {
        x->processor->resize(x->num_bins);
        torch_spectrails_tilde_configure_processor(x);
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
        // Automatically follow the current block size.
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

        auto outputs = x->processor->process_frame(mag_tensor, phase_tensor);
        auto processed_mag = outputs[0].to(torch::kCPU);
        auto processed_phase = outputs[1].to(torch::kCPU);

        auto mag_acc = processed_mag.accessor<float, 1>();
        auto phase_acc = processed_phase.accessor<float, 1>();
        for (size_t i = 0; i < num_bins; ++i) {
            out_mag[i] = mag_acc[i];
            out_phase[i] = phase_acc[i];
        }
        for (size_t i = num_bins; i < static_cast<size_t>(n); ++i) {
            out_mag[i] = 0.0f;
            out_phase[i] = 0.0f;
        }
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
            post("torch.spectrails~: resized to fftsize %zu", x->fft_size);
        }
    }
    
    if (sample_rate_changed) {
        x->sampling_rate_ = sys_getsr();
        torch_spectrails_tilde_configure_processor(x); // Atualiza min_peak_distance com nova taxa
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
    x->overlap_factor_ = 4; // Padrão comum

    // Parser de argumentos com flags
    pd_utils::ArgParser parser(argc, argv, &x->x_obj);
    
    x->threshold = parser.get_float("threshold thresh t", 0.01f);
    x->attack = parser.get_float("attack att a", 0.7f);
    x->overlap_factor_ = static_cast<int>(parser.get_float("overlap of", 4));
    
    // Se decaytime foi especificado, usa ele; senão usa decay direto
    float decaytime_sec = parser.get_float("decaytime decayt dtime dt", -1.0f);
    if (decaytime_sec > 0.0f) {
        // Usa fft_size temporário para cálculo inicial (será atualizado no DSP)
        size_t temp_fft_size = static_cast<size_t>(parser.get_float("fftsize fft n", 1024));
        x->decay = decaytime_to_decay_factor(decaytime_sec, x->sampling_rate_, temp_fft_size, x->overlap_factor_);
    } else {
        x->decay = parser.get_float("decay dec d", 0.999f);
    }
    
    x->min_peak_distance_hz = parser.get_float("min_peak_distance mindist mpd", 100.0f);

    // FFT size será determinado automaticamente pelo block size no DSP
    // mas permitimos override inicial se especificado
    x->fft_size = static_cast<size_t>(parser.get_float("fftsize fft n", 0));
    if (x->fft_size == 0) {
        x->fft_size = 1024; // Tamanho inicial padrão
    }
    
    x->num_bins = x->fft_size / 2 + 1;

    try {
        x->processor = std::make_unique<Processor>(x->num_bins, torch::kCPU);
        torch_spectrails_tilde_configure_processor(x);
    } catch (const std::exception &e) {
        pd_error(x, "torch.spectrails~: failed to create processor: %s", e.what());
        return nullptr;
    }

    inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
    outlet_new(&x->x_obj, &s_signal);
    outlet_new(&x->x_obj, &s_signal);

    post("torch.spectrails~: threshold=%.4f attack=%.3f decay=%.4f min_peak_dist=%.1fHz overlap=%d", 
         x->threshold, x->attack, x->decay, x->min_peak_distance_hz, x->overlap_factor_);
    return x;
}

static void torch_spectrails_tilde_free(t_torch_spectrails_tilde *x) {
    x->processor.reset();
}

static void torch_spectrails_tilde_threshold(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->threshold = f;
    torch_spectrails_tilde_configure_processor(x);
}

static void torch_spectrails_tilde_attack(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->attack = f;
    torch_spectrails_tilde_configure_processor(x);
}

static void torch_spectrails_tilde_decay(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->decay = f;
    torch_spectrails_tilde_configure_processor(x);
}

static void torch_spectrails_tilde_min_peak_distance(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->min_peak_distance_hz = f;
    torch_spectrails_tilde_configure_processor(x);
}

static void torch_spectrails_tilde_decaytime(t_torch_spectrails_tilde *x, t_floatarg f) {
    if (f > 0.0f) {
        x->decay = decaytime_to_decay_factor(f, x->sampling_rate_, x->fft_size, x->overlap_factor_);
        torch_spectrails_tilde_configure_processor(x);
    } else {
        pd_error(x, "torch.spectrails~: decaytime must be positive");
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
                    reinterpret_cast<t_method>(torch_spectrails_tilde_attack),
                    gensym("attack"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_decay),
                    gensym("decay"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_decaytime),
                    gensym("decaytime"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_decaytime),
                    gensym("decayt"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_decaytime),
                    gensym("dtime"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_decaytime),
                    gensym("dt"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_overlap),
                    gensym("overlap"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_min_peak_distance),
                    gensym("min_peak_distance"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_reset),
                    gensym("reset"), A_NULL, 0);
    
    post("torch.spectrails~: peak detection with slope trigger");
}

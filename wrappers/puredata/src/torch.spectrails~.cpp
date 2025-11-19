#include "m_pd.h"
#include "core_ap_spectrails.h"

#include <algorithm>
#include <memory>
#include <torch/torch.h>

using Processor = contorchionist::core::ap_spectrails::SpectralTrailsProcessor<float>;

typedef struct _torch_spectrails_tilde {
    t_object x_obj;
    std::unique_ptr<Processor> processor;

    size_t fft_size;
    size_t num_bins;

    float threshold;
    float attack;
    float phase_attack;
    float decay;
    int slope_frames;
} t_torch_spectrails_tilde;

static t_class *torch_spectrails_tilde_class = nullptr;

static void torch_spectrails_tilde_configure_processor(t_torch_spectrails_tilde *x) {
    if (!x->processor) {
        return;
    }
    x->processor->set_threshold(x->threshold);
    x->processor->set_attack(x->attack);
    x->processor->set_phase_attack(x->phase_attack);
    x->processor->set_decay(x->decay);
    x->processor->set_slope_hold_frames(x->slope_frames);
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
    size_t block_fft = static_cast<size_t>(sp[0]->s_n);
    if (block_fft != x->fft_size) {
        torch_spectrails_tilde_resize_processor(x, block_fft);
        post("torch.spectrails~: resized to fftsize %zu", x->fft_size);
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

    // Default parameters
    x->fft_size = 1024;
    x->threshold = 0.01f;
    x->attack = 0.8f;
    x->phase_attack = 0.8f;
    x->decay = 0.995f;
    x->slope_frames = 2;

    if (argc > 0 && argv[0].a_type == A_FLOAT && atom_getfloat(argv) > 0) {
        x->fft_size = static_cast<size_t>(atom_getfloat(argv));
    }
    if (argc > 1 && argv[1].a_type == A_FLOAT) {
        x->threshold = atom_getfloat(argv + 1);
    }
    if (argc > 2 && argv[2].a_type == A_FLOAT) {
        x->attack = atom_getfloat(argv + 2);
    }
    if (argc > 3 && argv[3].a_type == A_FLOAT) {
        x->phase_attack = atom_getfloat(argv + 3);
    }
    if (argc > 4 && argv[4].a_type == A_FLOAT) {
        x->decay = atom_getfloat(argv + 4);
    }
    if (argc > 5 && argv[5].a_type == A_FLOAT) {
        x->slope_frames = std::max(1, static_cast<int>(atom_getfloat(argv + 5)));
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

    post("torch.spectrails~: simple spectral trails (fftsize=%zu)", x->fft_size);
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

static void torch_spectrails_tilde_phaseattack(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->phase_attack = f;
    torch_spectrails_tilde_configure_processor(x);
}

static void torch_spectrails_tilde_decay(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->decay = f;
    torch_spectrails_tilde_configure_processor(x);
}

static void torch_spectrails_tilde_reset(t_torch_spectrails_tilde *x) {
    if (x->processor) {
        x->processor->reset_memory();
    }
}

static void torch_spectrails_tilde_slopeframes(t_torch_spectrails_tilde *x, t_floatarg f) {
    x->slope_frames = std::max(1, static_cast<int>(f));
    torch_spectrails_tilde_configure_processor(x);
}

extern "C" void setup_torch0x2espectrails_tilde(void) {
    torch_spectrails_tilde_class = class_new(gensym("torch.spectrails~"),
                                             reinterpret_cast<t_newmethod>(torch_spectrails_tilde_new),
                                             reinterpret_cast<t_method>(torch_spectrails_tilde_free),
                                             sizeof(t_torch_spectrails_tilde),
                                             CLASS_DEFAULT,
                                             A_GIMME, 0);

    CLASS_MAINSIGNALIN(torch_spectrails_tilde_class, t_torch_spectrails_tilde, threshold);
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
                    reinterpret_cast<t_method>(torch_spectrails_tilde_phaseattack),
                    gensym("phaseattack"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_phaseattack),
                    gensym("phase_attack"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_decay),
                    gensym("decay"), A_FLOAT, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_reset),
                    gensym("reset"), A_NULL, 0);
    class_addmethod(torch_spectrails_tilde_class,
                    reinterpret_cast<t_method>(torch_spectrails_tilde_slopeframes),
                    gensym("slopeframes"), A_FLOAT, 0);

    post("torch.spectrails~: simplified debug build");
}

#include "m_pd.h"
#include "core_ap_spectraltracker.hpp"
#include "../utils/include/pd_arg_parser.h"

#include <vector>
#include <memory>

typedef struct _spectraltracker_tilde {
    t_object x_obj;
    t_float x_f;

    std::unique_ptr<SpectralTrackerProcessor<double>> processor;

    double p_sample_rate;
    size_t p_fft_size;

    t_outlet* outlets[8];
} t_spectraltracker_tilde;

static t_class* spectraltracker_tilde_class;

static void* spectraltracker_tilde_new(t_symbol* s, int argc, t_atom* argv);
static void spectraltracker_tilde_free(t_spectraltracker_tilde* x);
static void spectraltracker_tilde_dsp(t_spectraltracker_tilde* x, t_signal** sp);
static t_int* spectraltracker_tilde_perform(t_int* w);

static t_int* spectraltracker_tilde_perform(t_int* w) {
    t_spectraltracker_tilde* x = (t_spectraltracker_tilde*)(w[1]);
    t_sample* in = (t_sample*)(w[2]);
    int n = (int)(w[3]);

    if (x->processor) {
        size_t num_bins = n / 2 + 1;
        std::vector<double> input_frame(num_bins);
        for (size_t i = 0; i < num_bins; ++i) {
            input_frame[i] = in[i];
        }

        x->processor->process_frame(input_frame);
        x->processor->find_and_assign_peaks(x->p_sample_rate);

        const auto& voices = x->processor->get_voices();
        for (size_t i = 0; i < 8; ++i) {
            if (i < voices.size()) {
                t_atom atom_list[3];
                SETFLOAT(&atom_list[0], voices[i].frequency);
                SETFLOAT(&atom_list[1], 10 * log10(voices[i].amplitude));
                SETFLOAT(&atom_list[2], i);
                outlet_list(x->outlets[i], &s_list, 3, atom_list);
            }
        }
    }

    return (w + 4);
}

static void spectraltracker_tilde_dsp(t_spectraltracker_tilde* x, t_signal** sp) {
    x->p_fft_size = sp[0]->s_n;
    x->p_sample_rate = sp[0]->s_sr;

    if (x->processor) {
        x->processor->resize(x->p_fft_size);
    }
    dsp_add(spectraltracker_tilde_perform, 3, x, sp[0]->s_vec, sp[0]->s_n);
}

static void spectraltracker_tilde_free(t_spectraltracker_tilde* x) {}

static void* spectraltracker_tilde_new(t_symbol* s, int argc, t_atom* argv) {
    t_spectraltracker_tilde* x = (t_spectraltracker_tilde*)pd_new(spectraltracker_tilde_class);
    if (!x) return nullptr;

    pd_utils::ArgParser parser(argc, argv, (t_object*)x);
    x->p_fft_size = static_cast<size_t>(parser.get_float("fftsize", 1024));

    try {
        x->processor = std::make_unique<SpectralTrackerProcessor<double>>(x->p_fft_size, 8);
    } catch (const std::exception& e) {
        pd_error(x, "spectraltracker~: EXCEPTION: %s", e.what());
        return nullptr;
    }

    for (int i = 0; i < 8; ++i) {
        x->outlets[i] = outlet_new(&x->x_obj, &s_list);
    }

    return (void*)x;
}

extern "C" {
    void spectraltracker_tilde_setup(void) {
        spectraltracker_tilde_class = class_new(gensym("spectraltracker~"),
            (t_newmethod)spectraltracker_tilde_new,
            (t_method)spectraltracker_tilde_free,
            sizeof(t_spectraltracker_tilde),
            CLASS_DEFAULT, A_GIMME, 0);

        CLASS_MAINSIGNALIN(spectraltracker_tilde_class, t_spectraltracker_tilde, x_f);
        class_addmethod(spectraltracker_tilde_class, (t_method)spectraltracker_tilde_dsp, gensym("dsp"), A_CANT, 0);
    }
}
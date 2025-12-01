#include "m_pd.h"
#include "../../../core/include/core_ap_compressor.h"
#include "../utils/include/pd_arg_parser.h"

using Compressor = contorchionist::core::ap_compressor::Compressor<float>;

static t_class *torch_compressor_tilde_class;

typedef struct _torch_compressor_tilde {
    t_object x_obj;
    t_float x_f;
    Compressor* compressor;
    
    // Parameters
    float thresh_db;
    float ratio;
    float curve_mode;
    float attack_ms;
    float release_ms;
    int window_size;
    float lookahead_ms;
    float makeup_db;
    float auto_makeup; // 0 or 1

} t_torch_compressor_tilde;

static t_int *torch_compressor_tilde_perform(t_int *w) {
    t_torch_compressor_tilde *x = (t_torch_compressor_tilde *)(w[1]);
    t_sample *in = (t_sample *)(w[2]);
    t_sample *out = (t_sample *)(w[3]);
    int n = (int)(w[4]);

    if (x->compressor) {
        x->compressor->process(in, out, n, 
            x->thresh_db, x->ratio, x->curve_mode, 
            x->attack_ms, x->release_ms);
    } else {
        for (int i = 0; i < n; i++) out[i] = 0;
    }

    return (w + 5);
}

static void torch_compressor_tilde_dsp(t_torch_compressor_tilde *x, t_signal **sp) {
    if (x->compressor) {
        x->compressor->setSampleRate(sp[0]->s_sr);
        // Re-apply lookahead as it depends on SR
        x->compressor->setLookahead(x->lookahead_ms);
    }
    dsp_add(torch_compressor_tilde_perform, 4, x, sp[0]->s_vec, sp[1]->s_vec, sp[0]->s_n);
}

static void torch_compressor_tilde_free(t_torch_compressor_tilde *x) {
    if (x->compressor) {
        delete x->compressor;
        x->compressor = nullptr;
    }
}

// Parameter methods
static void torch_compressor_tilde_thresh(t_torch_compressor_tilde *x, t_floatarg f) {
    x->thresh_db = f;
}

static void torch_compressor_tilde_ratio(t_torch_compressor_tilde *x, t_floatarg f) {
    x->ratio = (f < 1.0f) ? 1.0f : f;
}

static void torch_compressor_tilde_curve(t_torch_compressor_tilde *x, t_floatarg f) {
    x->curve_mode = f;
}

static void torch_compressor_tilde_attack(t_torch_compressor_tilde *x, t_floatarg f) {
    x->attack_ms = (f < 0.0f) ? 0.0f : f;
}

static void torch_compressor_tilde_release(t_torch_compressor_tilde *x, t_floatarg f) {
    x->release_ms = (f < 0.0f) ? 0.0f : f;
}

static void torch_compressor_tilde_window(t_torch_compressor_tilde *x, t_floatarg f) {
    int w = (int)f;
    if (w > 0) {
        x->window_size = w;
        if (x->compressor) x->compressor->setRMSWindowSize(w);
    }
}

static void torch_compressor_tilde_lookahead(t_torch_compressor_tilde *x, t_floatarg f) {
    x->lookahead_ms = (f < 0.0f) ? 0.0f : f;
    if (x->compressor) x->compressor->setLookahead(x->lookahead_ms);
}

static void torch_compressor_tilde_makeup(t_torch_compressor_tilde *x, t_floatarg f) {
    x->makeup_db = f;
    if (x->compressor) x->compressor->setMakeup(x->makeup_db);
}

static void torch_compressor_tilde_auto(t_torch_compressor_tilde *x, t_floatarg f) {
    x->auto_makeup = (f != 0.0f);
    if (x->compressor) x->compressor->setAutoMakeup(x->auto_makeup != 0.0f);
}

static void *torch_compressor_tilde_new(t_symbol *s, int argc, t_atom *argv) {
    t_torch_compressor_tilde *x = (t_torch_compressor_tilde *)pd_new(torch_compressor_tilde_class);
    
    // Defaults
    x->thresh_db = -60.0f;
    x->ratio = 2.0f;
    x->curve_mode = 0.0f;
    x->attack_ms = 10.0f;
    x->release_ms = 100.0f;
    x->window_size = 1024;
    x->lookahead_ms = 0.0f;
    x->makeup_db = 0.0f;
    x->auto_makeup = 0.0f;

    // Parse args
    pd_utils::ArgParser parser(argc, argv, (t_object*)x);
    x->thresh_db = parser.get_float("thresh threshold", x->thresh_db);
    x->ratio = parser.get_float("ratio", x->ratio);
    x->curve_mode = parser.get_float("curve mode", x->curve_mode);
    x->attack_ms = parser.get_float("attack att", x->attack_ms);
    x->release_ms = parser.get_float("release rel", x->release_ms);
    x->window_size = (int)parser.get_float("window win", (float)x->window_size);
    x->lookahead_ms = parser.get_float("lookahead look", x->lookahead_ms);
    x->makeup_db = parser.get_float("makeup gain", x->makeup_db);
    x->auto_makeup = parser.get_float("auto automakeup", x->auto_makeup);

    x->compressor = new Compressor(44100.0f, x->window_size);
    
    // Apply initial settings
    x->compressor->setLookahead(x->lookahead_ms);
    x->compressor->setMakeup(x->makeup_db);
    x->compressor->setAutoMakeup(x->auto_makeup != 0.0f);

    outlet_new(&x->x_obj, &s_signal);
    return (void *)x;
}

extern "C" void setup_torch0x2ecompressor_tilde(void) {
    torch_compressor_tilde_class = class_new(gensym("torch.compressor~"),
        (t_newmethod)torch_compressor_tilde_new,
        (t_method)torch_compressor_tilde_free,
        sizeof(t_torch_compressor_tilde),
        CLASS_DEFAULT,
        A_GIMME, 0);

    class_addmethod(torch_compressor_tilde_class, (t_method)torch_compressor_tilde_dsp, gensym("dsp"), A_CANT, 0);
    CLASS_MAINSIGNALIN(torch_compressor_tilde_class, t_torch_compressor_tilde, x_f);

    class_addmethod(torch_compressor_tilde_class, (t_method)torch_compressor_tilde_thresh, gensym("thresh"), A_FLOAT, 0);
    class_addmethod(torch_compressor_tilde_class, (t_method)torch_compressor_tilde_ratio, gensym("ratio"), A_FLOAT, 0);
    class_addmethod(torch_compressor_tilde_class, (t_method)torch_compressor_tilde_curve, gensym("curve"), A_FLOAT, 0);
    class_addmethod(torch_compressor_tilde_class, (t_method)torch_compressor_tilde_attack, gensym("attack"), A_FLOAT, 0);
    class_addmethod(torch_compressor_tilde_class, (t_method)torch_compressor_tilde_release, gensym("release"), A_FLOAT, 0);
    class_addmethod(torch_compressor_tilde_class, (t_method)torch_compressor_tilde_window, gensym("window"), A_FLOAT, 0);
    class_addmethod(torch_compressor_tilde_class, (t_method)torch_compressor_tilde_lookahead, gensym("lookahead"), A_FLOAT, 0);
    class_addmethod(torch_compressor_tilde_class, (t_method)torch_compressor_tilde_makeup, gensym("makeup"), A_FLOAT, 0);
    class_addmethod(torch_compressor_tilde_class, (t_method)torch_compressor_tilde_auto, gensym("auto"), A_FLOAT, 0);
}

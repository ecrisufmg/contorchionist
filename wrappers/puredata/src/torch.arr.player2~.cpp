#include "m_pd.h"
#include "pd_arg_parser.h"
#include <sndfile.h>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <unistd.h> // For close()

static t_class *torch_arr_player2_class;

struct t_torch_arr_player2 {
    t_object x_obj;
    t_float x_f; // Dummy for signal inlet

    // Outlets
    t_outlet *x_out_status;
    std::vector<t_outlet*> x_out_signals;

    // State
    std::string x_array_basename;
    int x_n_channels;
    double x_current_phase; // In samples
    double x_rate;
    bool x_playing;
    bool x_paused; // If true, we are paused (not playing, but phase is preserved)
    bool x_finished; // True if playback finished naturally
    
    // Fading
    double x_fade_ms;
    double x_fade_phase; // 0.0 to 1.0
    double x_fade_inc;   // Increment per sample
    int x_fade_state;    // 0: none, 1: fade-in, 2: fade-out
    int x_next_state;    // State to switch to after fade-out (0: stop, 1: pause, 2: seek)
    double x_seek_target; // Target phase for seek

    // Arrays
    std::vector<t_symbol*> x_array_names;
    std::vector<t_garray*> x_arrays;
    std::vector<t_word*> x_array_vecs;
    std::vector<int> x_array_sizes;

    // File info
    double x_file_sr;
    int x_file_frames;
    
    // Canvas for path resolution
    t_canvas *x_canvas;
    
    // Clock for status output
    t_clock *x_clock;

    // Control output state
    int x_ctl_mode; // 0: Manual, 1: Per Block (Default), 2: Time Interval
    double x_ctl_interval_ms;
    double x_samples_since_last_output;
    double x_sys_sr;

    // Change detection
    int x_last_sent_state;
    double x_last_sent_timesmp;
    double x_last_sent_rate;
    bool x_first_output;
    bool x_send_duration_on_tick; // Flag to send duration on next tick
};

// Helper to resolve path
static std::string resolve_path(t_torch_arr_player2 *x, const std::string &filename) {
    if (filename.empty()) return "";
    
    // Check if absolute path (simple check)
    if (filename[0] == '/' || filename[0] == '~' || (filename.length() > 1 && filename[1] == ':')) {
        return filename;
    }

    // Try to resolve relative to canvas
    if (x->x_canvas) {
        char buf[MAXPDSTRING], *bufptr;
        int fd = canvas_open(x->x_canvas, filename.c_str(), "", buf, &bufptr, MAXPDSTRING, 1);
        if (fd >= 0) {
            close(fd); // We just want the path
            return std::string(buf) + "/" + std::string(bufptr);
        }
    }
    
    return filename; // Fallback to as-is
}

// Helper to get array pointers
static bool check_arrays(t_torch_arr_player2 *x) {
    bool all_ok = true;
    for (int i = 0; i < x->x_n_channels; ++i) {
        int size;
        t_word *vec;
        x->x_arrays[i] = (t_garray *)pd_findbyclass(x->x_array_names[i], garray_class);
        if (!x->x_arrays[i] || !garray_getfloatwords(x->x_arrays[i], &size, &vec)) {
            all_ok = false;
            x->x_array_vecs[i] = nullptr;
            x->x_array_sizes[i] = 0;
        } else {
            x->x_array_vecs[i] = vec;
            x->x_array_sizes[i] = size;
        }
    }
    return all_ok;
}

// Cubic interpolation (from tabread4~)
static inline float cubic_interp(float *buf, int size, double findex) {
    int index = (int)findex;
    if (index < 1) index = 1;
    if (index > size - 3) index = size - 3;
    
    // Clamp again just in case size is very small
    if (index < 1) return 0.0f; 

    float frac = (float)(findex - index);
    float a = buf[index - 1];
    float b = buf[index];
    float c = buf[index + 1];
    float d = buf[index + 2];
    float cminusb = c - b;
    
    return b + frac * (
        cminusb - 0.1666667f * (1.0f - frac) * (
            (d - a - 3.0f * cminusb) * frac + (d + 2.0f * a - 3.0f * b)
        )
    );
}

// Helper to output status
static void torch_arr_player2_output_status(t_torch_arr_player2 *x) {
    int state = 0;
    if (x->x_finished) state = 3;
    else if (x->x_playing) state = 1;
    else if (x->x_paused) state = 2;
    else state = 0;

    double timesmp = x->x_current_phase;
    double timems = (x->x_file_sr > 0) ? (timesmp / x->x_file_sr * 1000.0) : 0.0;
    double times = (x->x_file_sr > 0) ? (timesmp / x->x_file_sr) : 0.0;
    double rate = x->x_rate;

    t_atom at;

    // Rate
    if (x->x_first_output || rate != x->x_last_sent_rate) {
        SETFLOAT(&at, rate);
        outlet_anything(x->x_out_status, gensym("rate"), 1, &at);
        x->x_last_sent_rate = rate;
    }

    // Time (smp, milis, sec)
    if (x->x_first_output || timesmp != x->x_last_sent_timesmp) {
        // sec (was tsec)
        SETFLOAT(&at, times);
        outlet_anything(x->x_out_status, gensym("sec"), 1, &at);

        // milis (was tmilis)
        SETFLOAT(&at, timems);
        outlet_anything(x->x_out_status, gensym("milis"), 1, &at);

        // smp (was tsmp)
        SETFLOAT(&at, timesmp);
        outlet_anything(x->x_out_status, gensym("smp"), 1, &at);
        
        x->x_last_sent_timesmp = timesmp;
    }

    // State
    if (x->x_first_output || state != x->x_last_sent_state) {
        SETFLOAT(&at, (t_float)state);
        outlet_anything(x->x_out_status, gensym("state"), 1, &at);
        x->x_last_sent_state = state;
    }
    
    x->x_first_output = false;
}

// Clock callback
static void torch_arr_player2_tick(t_torch_arr_player2 *x) {
    if (x->x_send_duration_on_tick) {
        if (x->x_file_sr > 0) {
            t_atom at;
            SETFLOAT(&at, (t_float)x->x_file_frames / (t_float)x->x_file_sr);
            outlet_anything(x->x_out_status, gensym("tsec"), 1, &at);
        }
        x->x_send_duration_on_tick = false;
    }
    torch_arr_player2_output_status(x);
}

// Perform routine
static t_int *torch_arr_player2_perform(t_int *w) {
    t_torch_arr_player2 *x = (t_torch_arr_player2 *)(w[1]);
    int n = (int)(w[2]);
    
    // Get output buffers
    std::vector<t_sample*> outs(x->x_n_channels);
    for (int i = 0; i < x->x_n_channels; ++i) {
        outs[i] = (t_sample*)(w[3 + i]);
    }

    // Check arrays
    if (!check_arrays(x)) {
        for (int i = 0; i < x->x_n_channels; ++i) {
            std::fill(outs[i], outs[i] + n, 0.0f);
        }
        return (w + 3 + x->x_n_channels);
    }

    double phase = x->x_current_phase;
    
    // Resampling compensation
    double resample_factor = 1.0;
    if (x->x_sys_sr > 0 && x->x_file_sr > 0) {
        resample_factor = x->x_file_sr / x->x_sys_sr;
    }
    double rate = x->x_rate * resample_factor;

    double fade_phase = x->x_fade_phase;
    double fade_inc = x->x_fade_inc;
    int fade_state = x->x_fade_state;

    for (int j = 0; j < n; ++j) {
        float output_gain = 1.0f;

        // Handle Fading
        if (fade_state != 0) {
            if (fade_state == 1) { // Fade In
                fade_phase += fade_inc;
                if (fade_phase >= 1.0) {
                    fade_phase = 1.0;
                    fade_state = 0; // Done
                }
            } else if (fade_state == 2) { // Fade Out
                fade_phase -= fade_inc;
                if (fade_phase <= 0.0) {
                    fade_phase = 0.0;
                    fade_state = 0;
                    
                    // Action after fade out
                    if (x->x_next_state == 0) { // Stop
                        x->x_playing = false;
                        x->x_paused = false;
                        phase = 0.0;
                    } else if (x->x_next_state == 1) { // Pause
                        x->x_playing = false;
                        x->x_paused = true;
                    } else if (x->x_next_state == 2) { // Seek
                        phase = x->x_seek_target;
                        x->x_playing = true; // Resume playing
                        x->x_paused = false;
                        fade_state = 1; // Start fade in
                        fade_phase = 0.0;
                    }
                }
            }
            output_gain = (float)fade_phase;
        } else {
            // Not fading
            if (!x->x_playing) output_gain = 0.0f;
        }

        // Read from arrays
        if (x->x_playing || (fade_state == 2)) { // Play if playing or fading out
            for (int i = 0; i < x->x_n_channels; ++i) {
                int index = (int)phase;
                int size = x->x_array_sizes[i];
                float val = 0.0f;

                if (size >= 4) {
                     if (index < 1) index = 1;
                     if (index > size - 3) index = size - 3;
                     
                     double frac = phase - index;
                     t_word *wp = x->x_array_vecs[i] + index;
                     
                     float a = wp[-1].w_float;
                     float b = wp[0].w_float;
                     float c = wp[1].w_float;
                     float d = wp[2].w_float;
                     float cminusb = c - b;
                     
                     val = b + frac * (
                        cminusb - 0.1666667f * (1.0f - frac) * (
                            (d - a - 3.0f * cminusb) * frac + (d + 2.0f * a - 3.0f * b)
                        )
                    );
                }
                
                outs[i][j] = val * output_gain;
            }
            
            // Increment phase
            phase += rate;
            
            // Loop or Stop?
            int min_size = x->x_array_sizes[0];
            for(int k=1; k<x->x_n_channels; ++k) if(x->x_array_sizes[k] < min_size) min_size = x->x_array_sizes[k];
            
            if (phase >= min_size - 2 || phase < 0) { // -2 for interpolation safety
                // Auto stop
                x->x_playing = false;
                x->x_paused = false;
                x->x_finished = true;
                phase = 0; // Reset
                clock_delay(x->x_clock, 0);
            }

        } else {
            for (int i = 0; i < x->x_n_channels; ++i) {
                outs[i][j] = 0.0f;
            }
        }
    }

    // Update state
    x->x_current_phase = phase;
    x->x_fade_phase = fade_phase;
    x->x_fade_state = fade_state;

    // Handle Control Output
    if (x->x_ctl_mode == 1) { // Per Block
        clock_delay(x->x_clock, 0);
    } else if (x->x_ctl_mode == 2) { // Time Interval
        x->x_samples_since_last_output += n;
        double thresh = (x->x_ctl_interval_ms * 0.001) * x->x_sys_sr;
        if (thresh > 0 && x->x_samples_since_last_output >= thresh) {
            clock_delay(x->x_clock, 0);
            x->x_samples_since_last_output -= thresh;
            // Prevent buildup if we are way behind
            if (x->x_samples_since_last_output > thresh) x->x_samples_since_last_output = 0;
        }
    }

    return (w + 3 + x->x_n_channels);
}

static void torch_arr_player2_dsp(t_torch_arr_player2 *x, t_signal **sp) {
    int n = sp[0]->s_n;
    
    // Calculate fade increment based on sample rate
    double sr = sp[0]->s_sr;
    x->x_sys_sr = sr; // Update system SR
    if (x->x_fade_ms > 0) {
        x->x_fade_inc = 1000.0 / (x->x_fade_ms * sr);
    } else {
        x->x_fade_inc = 1.0;
    }

    std::vector<t_int> args;
    args.push_back((t_int)x);
    args.push_back((t_int)n);
    for (int i = 0; i < x->x_n_channels; ++i) {
        args.push_back((t_int)sp[i]->s_vec); // Output vectors
    }
    
    dsp_addv(torch_arr_player2_perform, args.size(), args.data());
}

// Commands

static void torch_arr_player2_open(t_torch_arr_player2 *x, t_symbol *s) {
    std::string filename_raw = s->s_name;
    std::string filename = resolve_path(x, filename_raw);
    
    post("torch.arr.player2~: Attempting to open '%s'", filename.c_str());

    SF_INFO sfinfo;
    sfinfo.format = 0; // Must be zero for sf_open to read
    SNDFILE *sndfile = sf_open(filename.c_str(), SFM_READ, &sfinfo);

    if (!sndfile) {
        pd_error(x, "torch.arr.player2~: Failed to open file '%s'. Reason: %s", filename.c_str(), sf_strerror(NULL));
        return;
    }

    int file_chans = sfinfo.channels;
    int frames = (int)sfinfo.frames;
    
    post("torch.arr.player2~: Opened file. Channels: %d, Frames: %d, Rate: %d", file_chans, frames, sfinfo.samplerate);

    if (file_chans != x->x_n_channels) {
        pd_error(x, "torch.arr.player2~: Channel mismatch. Object has %d, file has %d", x->x_n_channels, file_chans);
    }
    
    int chans_to_load = std::min(x->x_n_channels, file_chans);
    
    // Check arrays to get new pointers
    if (!check_arrays(x)) {
        pd_error(x, "torch.arr.player2~: Could not find or access one or more arrays.");
        sf_close(sndfile);
        return;
    }

    // Zero out arrays (NO RESIZE)
    for (int i = 0; i < x->x_n_channels; ++i) {
        if (x->x_array_vecs[i] && x->x_array_sizes[i] > 0) {
            for(int k=0; k<x->x_array_sizes[i]; ++k) {
                x->x_array_vecs[i][k].w_float = 0.0f;
            }
        }
    }
    
    // Read data
    std::vector<float> buffer(frames * file_chans);
    sf_count_t read_count = sf_readf_float(sndfile, buffer.data(), frames);
    
    if (read_count != frames) {
        post("torch.arr.player2~: Warning - Expected %d frames, read %lld", frames, read_count);
    }
    
    // Deinterleave and copy to arrays
    for (int i = 0; i < read_count; ++i) {
        for (int c = 0; c < chans_to_load; ++c) {
            if (x->x_array_vecs[c]) {
                float sample = buffer[i * file_chans + c];
                int idx = i;
                // Check bounds
                if (idx < x->x_array_sizes[c]) {
                    x->x_array_vecs[c][idx].w_float = sample;
                }
            }
        }
    }
    
    sf_close(sndfile);
    
    // Redraw arrays
    for (int i = 0; i < chans_to_load; ++i) {
        if (x->x_arrays[i]) garray_redraw(x->x_arrays[i]);
    }
    
    x->x_file_frames = frames;
    x->x_file_sr = sfinfo.samplerate;
    
    post("torch.arr.player2~: Loaded %s (%d frames, %d Hz)", filename.c_str(), frames, (int)x->x_file_sr);

    if (x->x_file_frames > x->x_array_sizes[0]) {
         post("torch.arr.player2~: Warning - File is longer than array (%d vs %d). Truncated.", x->x_file_frames, x->x_array_sizes[0]);
    }

    if (x->x_file_sr > 0) {
        t_atom at;
        SETFLOAT(&at, (t_float)frames / (t_float)x->x_file_sr);
        outlet_anything(x->x_out_status, gensym("tsec"), 1, &at);
    }
}

static void torch_arr_player2_play(t_torch_arr_player2 *x) {
    if (x->x_playing && !x->x_paused && x->x_fade_state != 2) return; // Already playing
    
    x->x_playing = true;
    x->x_paused = false;
    x->x_finished = false;
    x->x_fade_state = 1; // Fade in
    x->x_fade_phase = 0.0;
}

static void torch_arr_player2_pause(t_torch_arr_player2 *x) {
    if (!x->x_playing || x->x_paused) return;
    
    x->x_fade_state = 2; // Fade out
    x->x_next_state = 1; // Pause
    x->x_finished = false;
}

static void torch_arr_player2_stop(t_torch_arr_player2 *x) {
    if (!x->x_playing && !x->x_paused) return;
    
    x->x_fade_state = 2; // Fade out
    x->x_next_state = 0; // Stop
    x->x_finished = false;
}

static void torch_arr_player2_seek(t_torch_arr_player2 *x, t_floatarg f) {
    double target = f;
    if (target < 0) target = 0;
    // Check max?
    
    if (x->x_playing && !x->x_paused) {
        x->x_seek_target = target;
        x->x_fade_state = 2; // Fade out
        x->x_next_state = 2; // Seek
    } else {
        x->x_current_phase = target;
    }
}

static void torch_arr_player2_rate(t_torch_arr_player2 *x, t_floatarg f) {
    x->x_rate = f;
}

static void torch_arr_player2_fade(t_torch_arr_player2 *x, t_floatarg f) {
    x->x_fade_ms = f;
    if (x->x_fade_ms < 0) x->x_fade_ms = 0;
}

static void torch_arr_player2_ctlrate(t_torch_arr_player2 *x, t_floatarg f) {
    if (f > 0) {
        x->x_ctl_mode = 2;
        x->x_ctl_interval_ms = 1000.0 / f;
    } else {
        x->x_ctl_mode = 0; // Manual
    }
}

static void torch_arr_player2_ctlms(t_torch_arr_player2 *x, t_floatarg f) {
    if (f > 0) {
        x->x_ctl_mode = 2;
        x->x_ctl_interval_ms = f;
    } else {
        x->x_ctl_mode = 0; // Manual
    }
}

// Unified message handler
static void torch_arr_player2_anything(t_torch_arr_player2 *x, t_symbol *s, int argc, t_atom *argv) {
    // Check for flags/commands directly without ArgParser for runtime efficiency
    
    // File
    if (s == gensym("open")) {
        if (argc > 0 && argv[0].a_type == A_SYMBOL) {
            torch_arr_player2_open(x, argv[0].a_w.w_symbol);
        } else {
            pd_error(x, "torch.arr.player2~: open requires a filename symbol");
        }
        return;
    }
    
    // Play/Start
    if (s == gensym("play") || s == gensym("start") || (s == gensym("float") && atom_getfloat(argv) == 1.0f)) {
        torch_arr_player2_play(x);
        return;
    }
    
    // Pause
    if (s == gensym("pause") || (s == gensym("float") && (atom_getfloat(argv) == 0.0f || atom_getfloat(argv) == 2.0f))) {
        torch_arr_player2_pause(x);
        return;
    }
    
    // Stop
    if (s == gensym("stop")) {
        torch_arr_player2_stop(x);
        return;
    }
    
    // Seek
    if (s == gensym("seek")) {
        float pos = 0;
        if (argc > 0) pos = atom_getfloat(argv);
        torch_arr_player2_seek(x, pos);
        return;
    }

    // smp (samples)
    if (s == gensym("smp")) {
        float pos = 0;
        if (argc > 0) pos = atom_getfloat(argv);
        torch_arr_player2_seek(x, pos);
        return;
    }

    // sec (seconds)
    if (s == gensym("sec")) {
        float sec = 0;
        if (argc > 0) sec = atom_getfloat(argv);
        if (x->x_file_sr > 0) {
            torch_arr_player2_seek(x, sec * x->x_file_sr);
        }
        return;
    }

    // milis (milliseconds)
    if (s == gensym("milis")) {
        float ms = 0;
        if (argc > 0) ms = atom_getfloat(argv);
        if (x->x_file_sr > 0) {
            torch_arr_player2_seek(x, (ms / 1000.0) * x->x_file_sr);
        }
        return;
    }
    
    // Rate
    if (s == gensym("rate")) {
        float r = 1.0;
        if (argc > 0) r = atom_getfloat(argv);
        torch_arr_player2_rate(x, r);
        return;
    }
    
    // Fade
    if (s == gensym("fade")) {
        float f = 10.0;
        if (argc > 0) f = atom_getfloat(argv);
        torch_arr_player2_fade(x, f);
        return;
    }

    // Control Rate
    if (s == gensym("ctlrate") || s == gensym("fps")) {
        float f = 0.0;
        if (argc > 0) f = atom_getfloat(argv);
        torch_arr_player2_ctlrate(x, f);
        return;
    }

    // Control Interval (ms)
    if (s == gensym("ctlms")) {
        float f = 0.0;
        if (argc > 0) f = atom_getfloat(argv);
        torch_arr_player2_ctlms(x, f);
        return;
    }
    
    // Bang to output status
    if (s == gensym("bang")) {
        torch_arr_player2_output_status(x);
        return;
    }
    
    // Set array basename
    if (s == gensym("array") || s == gensym("set")) {
        if (argc > 0 && argv[0].a_type == A_SYMBOL) {
            std::string new_base = argv[0].a_w.w_symbol->s_name;
            x->x_array_basename = new_base;
            
            // Update names
            for (int i = 0; i < x->x_n_channels; ++i) {
                std::string name = x->x_array_basename + "-" + std::to_string(i);
                x->x_array_names[i] = gensym(name.c_str());
            }
            
            // Check arrays to update info immediately
            check_arrays(x);
            
            // Update file frames estimate from array size (use first channel)
            if (x->x_n_channels > 0 && x->x_array_sizes[0] > 0) {
                x->x_file_frames = x->x_array_sizes[0];
            }
            
            post("torch.arr.player2~: Set array basename to '%s'", new_base.c_str());
        } else {
            pd_error(x, "torch.arr.player2~: array requires a symbol");
        }
        return;
    }

    // Set SR
    if (s == gensym("sr")) {
        if (argc > 0) {
            float sr = atom_getfloat(argv);
            if (sr > 0) {
                x->x_file_sr = sr;
            }
        }
        return;
    }
}

static void *torch_arr_player2_new(t_symbol *s, int argc, t_atom *argv) {
    t_torch_arr_player2 *x = (t_torch_arr_player2 *)pd_new(torch_arr_player2_class);
    
    post("torch.arr.player2~: Creating new instance. Args: %d", argc);

    pd_utils::ArgParser parser(argc, argv, (t_object*)x);
    
    x->x_n_channels = (int)parser.get_float("ch channel channels", 2.0f); // Default 2 channels
    x->x_array_basename = parser.get_string("array arr", "array");
    x->x_fade_ms = parser.get_float("fade fadems", 10.0f);
    x->x_rate = parser.get_float("rate r", 1.0f);
    
    post("torch.arr.player2~: Config - Channels: %d, Array Base: %s", x->x_n_channels, x->x_array_basename.c_str());

    // Setup outlets
    for (int i = 0; i < x->x_n_channels; ++i) {
        outlet_new(&x->x_obj, &s_signal);
    }
    x->x_out_status = outlet_new(&x->x_obj, &s_list);

    // Init state
    x->x_current_phase = 0;
    x->x_playing = false;
    x->x_paused = false;
    x->x_finished = false;
    x->x_fade_phase = 0.0;
    x->x_fade_state = 0;
    x->x_file_frames = 0;
    x->x_file_sr = 44100;
    
    // Init control output
    x->x_ctl_mode = 1; // Default: Per Block
    x->x_ctl_interval_ms = 0;
    x->x_samples_since_last_output = 0;
    x->x_sys_sr = sys_getsr();

    x->x_last_sent_state = -1;
    x->x_last_sent_timesmp = -1.0;
    x->x_last_sent_rate = 0.0;
    x->x_first_output = true;
    x->x_send_duration_on_tick = false;

    // Init arrays
    x->x_array_names.resize(x->x_n_channels);
    x->x_arrays.resize(x->x_n_channels, nullptr);
    x->x_array_vecs.resize(x->x_n_channels, nullptr);
    x->x_array_sizes.resize(x->x_n_channels, 0);

    for (int i = 0; i < x->x_n_channels; ++i) {
        std::string name = x->x_array_basename + "-" + std::to_string(i);
        x->x_array_names[i] = gensym(name.c_str());
    }

    // Capture canvas
    x->x_canvas = canvas_getcurrent();
    if (!x->x_canvas) post("torch.arr.player2~: Warning - Could not get current canvas.");

    // Init clock
    x->x_clock = clock_new(x, (t_method)torch_arr_player2_tick);

    // Load file if provided in args
    std::string file = parser.get_string("file f");
    post("torch.arr.player2~: -file argument: '%s'", file.c_str());

    if (!file.empty()) {
        torch_arr_player2_open(x, gensym(file.c_str()));
        x->x_send_duration_on_tick = true;
        clock_delay(x->x_clock, 0);
    }

    return (void *)x;
}

static void torch_arr_player2_free(t_torch_arr_player2 *x) {
    clock_free(x->x_clock);
}

extern "C" void setup_torch0x2earr0x2eplayer2_tilde(void) {
    torch_arr_player2_class = class_new(gensym("torch.arr.player2~"),
        (t_newmethod)torch_arr_player2_new,
        (t_method)torch_arr_player2_free,
        sizeof(t_torch_arr_player2),
        CLASS_DEFAULT,
        A_GIMME, 0);

    class_addmethod(torch_arr_player2_class, (t_method)torch_arr_player2_dsp, gensym("dsp"), A_CANT, 0);
    class_addanything(torch_arr_player2_class, torch_arr_player2_anything);
}

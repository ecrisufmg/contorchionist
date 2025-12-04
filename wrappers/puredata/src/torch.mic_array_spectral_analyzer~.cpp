#include "m_pd.h"
#include "core_ap_micarrayspecanalyzer.hpp"
#include "../utils/include/pd_arg_parser.h"
#include "../utils/include/pd_torch_device_adapter.h"

#include <vector>
#include <memory>
#include <string>

using Analyzer = contorchionist::core::ap_micarrayspecanalyzer::MicArraySpecAnalyzer;
using AnalysisResult = contorchionist::core::ap_micarrayspecanalyzer::AnalysisResult;
using BandConfig = contorchionist::core::ap_micarrayspecanalyzer::BandConfig;

typedef struct _torch_mic_array_spectral_analyzer_tilde {
    t_object x_obj;
    t_sample x_f_dummy;
    
    int num_mics_;
    std::unique_ptr<Analyzer> analyzer_;
    torch::Device device_;
    
    // Inlets/Outlets
    std::vector<t_inlet*> inlets_; // Additional inlets (first is default)
    t_outlet* out_control_;
    t_clock* clock_;
    
    // State
    float sample_rate_;
    int current_block_size_;
    
    // Data Exchange
    std::vector<AnalysisResult> results_buffer_;
    bool results_ready_;
    
} t_torch_mic_array_spectral_analyzer_tilde;

static t_class *torch_mic_array_spectral_analyzer_tilde_class = nullptr;

static void torch_mic_array_spectral_analyzer_tilde_tick(t_torch_mic_array_spectral_analyzer_tilde *x) {
    if (x->results_ready_) {
        for (const auto& res : x->results_buffer_) {
            t_atom argv[5];
            SETFLOAT(argv+0, static_cast<t_float>(res.band_index));
            SETFLOAT(argv+1, static_cast<t_float>(res.angle_deg));
            SETFLOAT(argv+2, static_cast<t_float>(res.strength));
            SETFLOAT(argv+3, static_cast<t_float>(res.overall_level));
            SETFLOAT(argv+4, static_cast<t_float>(res.dominant_freq_hz));
            
            outlet_list(x->out_control_, &s_list, 5, argv);
        }
        x->results_ready_ = false;
    }
}

static t_int *torch_mic_array_spectral_analyzer_tilde_perform(t_int *w) {
    // w[0] = dsp method
    // w[1] = x
    // w[2] = n
    // w[3...] = inputs
    
    auto *x = reinterpret_cast<t_torch_mic_array_spectral_analyzer_tilde *>(w[1]);
    int n = static_cast<int>(w[2]);
    int num_mics = x->num_mics_;
    
    if (!x->analyzer_) return w + 3 + num_mics;

    // Gather inputs into a tensor [num_mics, num_bins]
    // Inputs are full FFT frames? Or RFFT (N/2+1)?
    // The spec says "Input Domain: Frequency Domain ... provided by external RFFT objects".
    // Usually RFFT objects in PD output N/2+1 bins, but the block size N is the full FFT size?
    // Or is the block size N/2+1?
    // Standard PD convention: if block size is N, signal vector has N samples.
    // If it's RFFT data, it usually comes as 2 signals (Real, Imag) or Mag/Phase.
    // Here we only take Magnitude.
    // If the upstream object is `rfft~`, it outputs N/2 real values and N/2 imag values? 
    // Or does it output a block of size N where only N/2+1 are valid?
    // Let's assume the input signal vector size `n` corresponds to the number of bins we process.
    // If the user connects an `rfft~` with block size 1024, the signal vector size is usually 1024 (padded) or 513?
    // In PD, `rfft~` usually runs inside a subpatch with `block~`.
    // Let's assume `n` is the number of bins provided.
    
    // Wait, the analyzer was initialized with `fft_size`.
    // `num_bins` = fft_size / 2 + 1.
    // If `n` != `num_bins`, we might have a mismatch or we need to resize.
    // For now, let's assume `n` is sufficient to cover `num_bins`.
    
    // Actually, usually `n` (block size) = FFT Size.
    // And valid bins are 0 to N/2.
    // So we read N/2 + 1 values from the input vectors.
    
    int num_bins = n / 2 + 1;
    
    try {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(x->device_);
        // We need to copy data from N input vectors to a tensor.
        // This is the bottleneck.
        
        // Allocate tensor [num_mics, num_bins]
        // Ideally we should have a pre-allocated buffer in `x` to avoid allocation in DSP loop.
        // But for simplicity/safety with LibTorch (which manages memory well), let's create from blob if possible, 
        // but data is non-contiguous (separate pointers).
        // So we must copy.
        
        // Optimization: Use a persistent CPU buffer and copy to it, then to GPU if needed.
        // For now, let's create a CPU tensor and copy.
        
        auto input_tensor = torch::empty({num_mics, num_bins}, torch::kFloat32); // CPU
        auto acc = input_tensor.accessor<float, 2>();
        
        for (int i = 0; i < num_mics; ++i) {
            t_sample* in = reinterpret_cast<t_sample*>(w[3 + i]);
            for (int k = 0; k < num_bins; ++k) {
                acc[i][k] = static_cast<float>(in[k]);
            }
        }
        
        // Process
        auto results = x->analyzer_->process_bands(input_tensor);
        
        // Store results
        x->results_buffer_ = results;
        x->results_ready_ = true;
        clock_delay(x->clock_, 0);
        
    } catch (const std::exception& e) {
        pd_error(x, "torch.mic_array_spectral_analyzer~: %s", e.what());
    }
    
    return w + 3 + num_mics;
}

static void update_processor_settings(t_torch_mic_array_spectral_analyzer_tilde *x) {
    if (!x->analyzer_) return;
    if (x->current_block_size_ <= 0 || x->sample_rate_ <= 0) return;

    try {
        x->analyzer_->set_sample_rate(x->sample_rate_);
        x->analyzer_->resize(x->current_block_size_);
    } catch (const std::exception& e) {
        pd_error(x, "torch.mic_array_spectral_analyzer~: Error updating settings: %s", e.what());
    }
}

static void torch_mic_array_spectral_analyzer_tilde_dsp(t_torch_mic_array_spectral_analyzer_tilde *x, t_signal **sp) {
    int n = sp[0]->s_n;
    float sr = sys_getsr();
    
    bool block_size_changed = (x->current_block_size_ != n);
    bool sample_rate_changed = (x->sample_rate_ != sr);
    
    if (block_size_changed) x->current_block_size_ = n;
    if (sample_rate_changed) x->sample_rate_ = sr;
    
    if (block_size_changed || sample_rate_changed) {
        update_processor_settings(x);
    }

    std::vector<t_int> vec;
    vec.push_back(reinterpret_cast<t_int>(x));
    vec.push_back(static_cast<t_int>(n));
    
    for (int i = 0; i < x->num_mics_; ++i) {
        vec.push_back(reinterpret_cast<t_int>(sp[i]->s_vec));
    }
    
    dsp_addv(torch_mic_array_spectral_analyzer_tilde_perform, vec.size(), vec.data());
}

static void torch_mic_array_spectral_analyzer_tilde_mic(t_torch_mic_array_spectral_analyzer_tilde *x, t_floatarg id, t_floatarg angle, t_floatarg dist) {
    if (x->analyzer_) {
        x->analyzer_->set_mic_geometry(static_cast<int>(id), angle, dist);
    }
}

static void torch_mic_array_spectral_analyzer_tilde_speaker(t_torch_mic_array_spectral_analyzer_tilde *x, t_floatarg angle) {
    if (x->analyzer_) {
        x->analyzer_->register_speaker(angle);
    }
}

static void torch_mic_array_spectral_analyzer_tilde_calibrate(t_torch_mic_array_spectral_analyzer_tilde *x) {
    if (x->analyzer_) {
        x->analyzer_->calibrate();
    }
}

static void torch_mic_array_spectral_analyzer_tilde_bands(t_torch_mic_array_spectral_analyzer_tilde *x, t_symbol *s, int argc, t_atom *argv) {
    if (!x->analyzer_) return;
    
    if (argc % 2 != 0) {
        pd_error(x, "torch.mic_array_spectral_analyzer~: @bands requires pairs of frequencies (min max)");
        return;
    }
    
    std::vector<BandConfig> bands;
    for (int i = 0; i < argc; i += 2) {
        float min_hz = atom_getfloat(argv + i);
        float max_hz = atom_getfloat(argv + i + 1);
        bands.push_back({min_hz, max_hz});
    }
    x->analyzer_->set_bands(bands);
}

static void torch_mic_array_spectral_analyzer_tilde_overlap(t_torch_mic_array_spectral_analyzer_tilde *x, t_floatarg f) {
    if (x->analyzer_) {
        x->analyzer_->set_overlap_factor(f);
    }
}

static void torch_mic_array_spectral_analyzer_tilde_strength_mode(t_torch_mic_array_spectral_analyzer_tilde *x, t_symbol *s) {
    if (!x->analyzer_) return;
    std::string mode = s->s_name;
    if (mode == "mag" || mode == "magnitude") {
        x->analyzer_->set_strength_mode(contorchionist::core::ap_micarrayspecanalyzer::StrengthMode::MAGNITUDE);
    } else if (mode == "pow" || mode == "power") {
        x->analyzer_->set_strength_mode(contorchionist::core::ap_micarrayspecanalyzer::StrengthMode::POWER);
    } else {
        pd_error(x, "torch.mic_array_spectral_analyzer~: Invalid strength mode '%s'. Use 'mag' or 'pow'.", mode.c_str());
    }
}

static void torch_mic_array_spectral_analyzer_tilde_level_mode(t_torch_mic_array_spectral_analyzer_tilde *x, t_symbol *s) {
    if (!x->analyzer_) return;
    std::string mode = s->s_name;
    if (mode == "mag" || mode == "magnitude") {
        x->analyzer_->set_level_mode(contorchionist::core::ap_micarrayspecanalyzer::LevelMode::MAGNITUDE);
    } else if (mode == "pow" || mode == "power") {
        x->analyzer_->set_level_mode(contorchionist::core::ap_micarrayspecanalyzer::LevelMode::POWER);
    } else if (mode == "db") {
        x->analyzer_->set_level_mode(contorchionist::core::ap_micarrayspecanalyzer::LevelMode::DB);
    } else {
        pd_error(x, "torch.mic_array_spectral_analyzer~: Invalid level mode '%s'. Use 'mag', 'pow', or 'db'.", mode.c_str());
    }
}

static void *torch_mic_array_spectral_analyzer_tilde_new(t_symbol *s, int argc, t_atom *argv) {
    auto *x = reinterpret_cast<t_torch_mic_array_spectral_analyzer_tilde *>(pd_new(torch_mic_array_spectral_analyzer_tilde_class));
    if (!x) return nullptr;
    
    pd_utils::ArgParser parser(argc, argv, &x->x_obj);
    
    x->num_mics_ = static_cast<int>(parser.get_float("mics m", 4));
    if (x->num_mics_ < 1) x->num_mics_ = 1;
    
    // Device
    bool verbose = parser.has_flag("verbose v");
    std::string device_str = parser.get_string("device d", "cpu");
    auto dev_res = get_device_from_string(device_str);
    x->device_ = dev_res.first;
    pd_parse_and_set_torch_device(&x->x_obj, x->device_, device_str, verbose, "torch.mic_array_spectral_analyzer~", parser.has_flag("device d"));

    // FFT Size (default to block size, but we don't know it yet, assume 1024 or get from sys)
    int fft_size = sys_getblksize();
    if (fft_size == 0) fft_size = 1024;
    
    x->analyzer_ = std::make_unique<Analyzer>(x->num_mics_, fft_size, x->device_);
    
    // Strength Mode
    if (parser.has_flag("mag")) {
        x->analyzer_->set_strength_mode(contorchionist::core::ap_micarrayspecanalyzer::StrengthMode::MAGNITUDE);
    } else if (parser.has_flag("pow")) {
        x->analyzer_->set_strength_mode(contorchionist::core::ap_micarrayspecanalyzer::StrengthMode::POWER);
    } else {
        // Default to POWER as requested
        x->analyzer_->set_strength_mode(contorchionist::core::ap_micarrayspecanalyzer::StrengthMode::POWER);
    }

    // Level Mode
    std::string level_str = parser.get_string("level l", "");
    if (level_str.empty()) level_str = parser.get_string("rms", ""); // Alias
    
    if (!level_str.empty()) {
        if (level_str == "mag") x->analyzer_->set_level_mode(contorchionist::core::ap_micarrayspecanalyzer::LevelMode::MAGNITUDE);
        else if (level_str == "pow") x->analyzer_->set_level_mode(contorchionist::core::ap_micarrayspecanalyzer::LevelMode::POWER);
        else if (level_str == "db") x->analyzer_->set_level_mode(contorchionist::core::ap_micarrayspecanalyzer::LevelMode::DB);
    }

    // Default Geometry (Inverted Star) if num_mics == 4
    if (x->num_mics_ == 4) {
        // Mic 0: Angle -135°, Dist 0.2m
        x->analyzer_->set_mic_geometry(0, -135.0f, 0.2f);
        // Mic 1: Angle +135°, Dist 0.2m
        x->analyzer_->set_mic_geometry(1, 135.0f, 0.2f);
        // Mic 2: Angle -45°, Dist 0.2m
        x->analyzer_->set_mic_geometry(2, -45.0f, 0.2f);
        // Mic 3: Angle +45°, Dist 0.2m
        x->analyzer_->set_mic_geometry(3, 45.0f, 0.2f);
        
        // Apply immediately
        x->analyzer_->calibrate();
    }

    // Initial Bands
    std::vector<BandConfig> initial_bands;
    bool bands_found = false;
    
    // Manual scan for @bands since ArgParser might not handle variable lists easily
    for (int i = 0; i < argc; ++i) {
        if (argv[i].a_type == A_SYMBOL) {
            t_symbol* sym = atom_getsymbol(argv + i);
            if (sym == gensym("@bands")) {
                bands_found = true;
                // Parse subsequent floats until next symbol or end
                int j = i + 1;
                std::vector<float> band_freqs;
                while (j < argc && argv[j].a_type == A_FLOAT) {
                    band_freqs.push_back(atom_getfloat(argv + j));
                    j++;
                }
                
                if (band_freqs.size() % 2 != 0) {
                    pd_error(x, "torch.mic_array_spectral_analyzer~: @bands requires pairs of frequencies");
                } else {
                    for (size_t k = 0; k < band_freqs.size(); k += 2) {
                        initial_bands.push_back({band_freqs[k], band_freqs[k+1]});
                    }
                }
                break; // Assume only one @bands flag
            }
        }
    }
    
    if (!bands_found || initial_bands.empty()) {
        // Default full range
        initial_bands.push_back({20.0f, 20000.0f});
    }
    
    x->analyzer_->set_bands(initial_bands);
    
    // Overlap
    float overlap = parser.get_float("overlap of", 1.0f);
    x->analyzer_->set_overlap_factor(overlap);

    // Inlets
    // First inlet is x_obj (signal). We need num_mics - 1 more signal inlets.
    for (int i = 1; i < x->num_mics_; ++i) {
        x->inlets_.push_back(inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal));
    }
    
    // Outlet
    x->out_control_ = outlet_new(&x->x_obj, &s_list);
    x->clock_ = clock_new(x, (t_method)torch_mic_array_spectral_analyzer_tilde_tick);
    
    return x;
}

static void torch_mic_array_spectral_analyzer_tilde_free(t_torch_mic_array_spectral_analyzer_tilde *x) {
    if (x->clock_) clock_free(x->clock_);
    for (auto* in : x->inlets_) {
        inlet_free(in);
    }
    x->analyzer_.reset();
}

extern "C" void setup_torch0x2emic_array_spectral_analyzer_tilde(void) {
    torch_mic_array_spectral_analyzer_tilde_class = class_new(
        gensym("torch.mic_array_spectral_analyzer~"),
        reinterpret_cast<t_newmethod>(torch_mic_array_spectral_analyzer_tilde_new),
        reinterpret_cast<t_method>(torch_mic_array_spectral_analyzer_tilde_free),
        sizeof(t_torch_mic_array_spectral_analyzer_tilde),
        CLASS_DEFAULT,
        A_GIMME, 0);
        
    CLASS_MAINSIGNALIN(torch_mic_array_spectral_analyzer_tilde_class, t_torch_mic_array_spectral_analyzer_tilde, x_f_dummy);
    
    class_addmethod(torch_mic_array_spectral_analyzer_tilde_class,
        reinterpret_cast<t_method>(torch_mic_array_spectral_analyzer_tilde_dsp),
        gensym("dsp"), A_CANT, 0);
        
    class_addmethod(torch_mic_array_spectral_analyzer_tilde_class,
        reinterpret_cast<t_method>(torch_mic_array_spectral_analyzer_tilde_mic),
        gensym("mic"), A_FLOAT, A_FLOAT, A_FLOAT, 0);
        
    class_addmethod(torch_mic_array_spectral_analyzer_tilde_class,
        reinterpret_cast<t_method>(torch_mic_array_spectral_analyzer_tilde_speaker),
        gensym("speaker"), A_FLOAT, 0);
        
    class_addmethod(torch_mic_array_spectral_analyzer_tilde_class,
        reinterpret_cast<t_method>(torch_mic_array_spectral_analyzer_tilde_calibrate),
        gensym("calibrate"), A_NULL, 0);
        
    class_addmethod(torch_mic_array_spectral_analyzer_tilde_class,
        reinterpret_cast<t_method>(torch_mic_array_spectral_analyzer_tilde_bands),
        gensym("bands"), A_GIMME, 0);
        
    class_addmethod(torch_mic_array_spectral_analyzer_tilde_class,
        reinterpret_cast<t_method>(torch_mic_array_spectral_analyzer_tilde_overlap),
        gensym("overlap"), A_FLOAT, 0);
        
    class_addmethod(torch_mic_array_spectral_analyzer_tilde_class,
        reinterpret_cast<t_method>(torch_mic_array_spectral_analyzer_tilde_overlap),
        gensym("of"), A_FLOAT, 0);

    class_addmethod(torch_mic_array_spectral_analyzer_tilde_class,
        reinterpret_cast<t_method>(torch_mic_array_spectral_analyzer_tilde_strength_mode),
        gensym("strength_mode"), A_DEFSYMBOL, 0);

    class_addmethod(torch_mic_array_spectral_analyzer_tilde_class,
        reinterpret_cast<t_method>(torch_mic_array_spectral_analyzer_tilde_level_mode),
        gensym("level_mode"), A_DEFSYMBOL, 0);
}

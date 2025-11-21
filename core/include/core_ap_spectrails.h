#ifndef CORE_AP_SPECTRAILS_H
#define CORE_AP_SPECTRAILS_H

#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace contorchionist {
namespace core {
namespace ap_spectrails {

struct PeakInfo {
    int rank;
    float freq_hz;
    float mag_db;
    int state; // 1: new, 0: sustained, -1: decayed
    float mag_lin; // Used for sorting
};

/**
 * @brief Detection mode for spectral peaks
 */
enum class DetectionMode {
    SLOPE_BASED = 0,    // Original: negative slope after peak
    PROMINENCE = 1      // New: relative threshold + parabolic interpolation
};

/**
 * @brief Spectral trails processor with peak detection and slope-based triggering.
 *
 * This processor:
 * 1. Detects spectral peaks (local maxima)
 * 2. Tracks peak magnitudes and triggers write on negative slope
 * 3. Maintains a decay table with phase lock
 * 4. Enforces minimum frequency spacing between peaks
 *
 * Operates on RFFT half-spectra (N/2 + 1 bins).
 * DC (bin 0) and Nyquist (last bin) are forced to zero.
 */
template <typename T>
class SpectralTrailsProcessor {
private:
    // Declare private members FIRST so inline methods can use them
    size_t num_bins_;
    torch::Device device_;
    torch::Tensor output_magnitude_;
    torch::Tensor output_phase_;
    torch::Tensor target_magnitude_;
    torch::Tensor start_magnitude_;
    torch::Tensor target_phase_;
    torch::Tensor start_phase_;
    torch::Tensor envelope_position_;
    torch::Tensor previous_magnitude_;
    torch::Tensor peak_magnitude_;
    torch::Tensor last_written_peak_bin_;

    T threshold_;
    T attack_;
    T decay_;
    T min_peak_distance_bins_;
    T sample_rate_;
    T fft_size_;
    DetectionMode detection_mode_;
    T prominence_threshold_;
    bool use_parabolic_interp_;
    int max_peaks_;
    T floor_threshold_;
    T output_gain_;
    bool use_limiter_;
    T limiter_threshold_;
    
    std::vector<PeakInfo> latest_peaks_;

    void allocate_memory() {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
        output_magnitude_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
        output_phase_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
        target_magnitude_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
        start_magnitude_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
        target_phase_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
        start_phase_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
        envelope_position_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
        previous_magnitude_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
        peak_magnitude_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
        last_written_peak_bin_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
    }

public:
    explicit SpectralTrailsProcessor(size_t num_bins,
                                     torch::Device device = torch::kCPU)
        : num_bins_(num_bins),
          device_(device),
          threshold_(static_cast<T>(0.01)),
          attack_(static_cast<T>(0.7)),
          decay_(static_cast<T>(0.999)),
          min_peak_distance_bins_(static_cast<T>(2.0)),
          sample_rate_(static_cast<T>(44100.0)),
          fft_size_(static_cast<T>((num_bins - 1) * 2)),
          detection_mode_(DetectionMode::SLOPE_BASED),
          prominence_threshold_(static_cast<T>(0.6)),
          use_parabolic_interp_(true),
          max_peaks_(0),
          floor_threshold_(static_cast<T>(-1.0)),
          output_gain_(static_cast<T>(1.0)),
          use_limiter_(false),
          limiter_threshold_(static_cast<T>(1.0)) {
        if (num_bins_ == 0) {
            throw std::invalid_argument("SpectralTrailsProcessor: num_bins must be > 0");
        }
        allocate_memory();
    }

    void resize(size_t num_bins) {
        if (num_bins == 0) {
            throw std::invalid_argument("SpectralTrailsProcessor: num_bins must be > 0");
        }
        num_bins_ = num_bins;
        fft_size_ = static_cast<T>((num_bins - 1) * 2);
        allocate_memory();
    }

    void reset_memory() {
        output_magnitude_.zero_();
        output_phase_.zero_();
        target_magnitude_.zero_();
        start_magnitude_.zero_();
        target_phase_.zero_();
        start_phase_.zero_();
        envelope_position_.zero_();
        previous_magnitude_.zero_();
        peak_magnitude_.zero_();
        last_written_peak_bin_.zero_();
    }

    void set_threshold(T value) { threshold_ = std::max(static_cast<T>(0), value); }
    void set_attack(T value) { attack_ = std::clamp(value, static_cast<T>(0), static_cast<T>(1)); }
    void set_decay(T value) { decay_ = std::clamp(value, static_cast<T>(0), static_cast<T>(1)); }
    void set_min_peak_distance_hz(T value, T sample_rate) {
        sample_rate_ = sample_rate;
        min_peak_distance_bins_ = (value * fft_size_) / sample_rate_;
    }
    void set_detection_mode(DetectionMode mode) { detection_mode_ = mode; }
    void set_prominence_threshold(T value) { prominence_threshold_ = std::clamp(value, static_cast<T>(0), static_cast<T>(1)); }
    void set_parabolic_interpolation(bool enable) { use_parabolic_interp_ = enable; }
    void set_max_peaks(int max_peaks) { max_peaks_ = max_peaks; }
    void set_floor_threshold(T value) { floor_threshold_ = value; }
    void set_output_gain(T value) { output_gain_ = value; }
    void set_limiter(bool enable, T threshold = static_cast<T>(1.0)) {
        use_limiter_ = enable;
        limiter_threshold_ = threshold;
    }
    
    const std::vector<PeakInfo>& get_latest_peaks() const { return latest_peaks_; }

    // Get/set envelope positions for multi-channel synchronization
    torch::Tensor get_envelope_positions() const {
        return envelope_position_.clone();
    }

    void set_envelope_positions(const torch::Tensor& envelope) {
        if (envelope.size(0) != static_cast<long>(num_bins_)) {
            throw std::invalid_argument("SpectralTrailsProcessor: envelope size mismatch");
        }
        envelope_position_ = envelope.to(device_, torch::kFloat32);
    }

    // Force write specific bins (for ambisonic sync - triggered by W channel)
    void force_write_bins(const torch::Tensor& bins_to_write,
                         const torch::Tensor& magnitude_input,
                         const torch::Tensor& phase_input) {
        auto bins_acc = bins_to_write.template accessor<float, 1>();
        auto mag_acc = magnitude_input.template accessor<float, 1>();
        auto phase_acc = phase_input.template accessor<float, 1>();
        auto start_mag_acc = start_magnitude_.template accessor<float, 1>();
        auto target_mag_acc = target_magnitude_.template accessor<float, 1>();
        auto start_phase_acc = start_phase_.template accessor<float, 1>();
        auto target_phase_acc = target_phase_.template accessor<float, 1>();
        auto out_mag_acc = output_magnitude_.template accessor<float, 1>();
        auto out_phase_acc = output_phase_.template accessor<float, 1>();
        auto env_acc = envelope_position_.template accessor<float, 1>();
        auto peak_acc = peak_magnitude_.template accessor<float, 1>();

        for (long i = 0; i < bins_to_write.size(0); ++i) {
            if (bins_acc[i] > 0.5f) { // bin marked for write
                start_mag_acc[i] = out_mag_acc[i];
                target_mag_acc[i] = mag_acc[i];
                start_phase_acc[i] = out_phase_acc[i];
                target_phase_acc[i] = phase_acc[i];
                env_acc[i] = 0.0f; // Reset envelope
                peak_acc[i] = 0.0f; // Reset peak tracker
            }
        }
    }

    std::vector<torch::Tensor> process_frame(const torch::Tensor& magnitude_input,
                                             const torch::Tensor& phase_input) {
        if (magnitude_input.size(0) != static_cast<long>(num_bins_)) {
            throw std::invalid_argument("SpectralTrailsProcessor: magnitude_input size mismatch");
        }
        if (phase_input.size(0) != static_cast<long>(num_bins_)) {
            throw std::invalid_argument("SpectralTrailsProcessor: phase_input size mismatch");
        }

        auto mag = magnitude_input.to(device_, torch::kFloat32);
        auto phase = phase_input.to(device_, torch::kFloat32);
        
        latest_peaks_.clear();
        std::vector<int> new_peak_bins;

        // 1. Update envelope positions (move towards 1.0 using attack rate)
        envelope_position_ = torch::clamp(envelope_position_ + attack_, 0.0f, 1.0f);

        // 2. Interpolate from start to target using envelope position
        output_magnitude_ = start_magnitude_ + (target_magnitude_ - start_magnitude_) * envelope_position_;
        
        // Interpolate phase with proper wrapping (shortest path between angles)
        auto phase_diff = target_phase_ - start_phase_;
        // Wrap phase difference to [-pi, pi]
        phase_diff = torch::atan2(torch::sin(phase_diff), torch::cos(phase_diff));
        output_phase_ = start_phase_ + phase_diff * envelope_position_;

        // 3. Decay all bins
        output_magnitude_ *= decay_;
        target_magnitude_ *= decay_;
        start_magnitude_ *= decay_;

        // 4. Age out old peak markers (clear peaks that have decayed away)
        auto out_mag_check = output_magnitude_.template accessor<float, 1>();
        auto last_peak_check = last_written_peak_bin_.template accessor<float, 1>();
        
        // Determine release threshold: use floor_threshold_ if set.
        // Otherwise default to threshold - 20dB (0.1 * threshold).
        float release_level = (floor_threshold_ > static_cast<T>(0.0)) ? floor_threshold_ : (threshold_ * static_cast<T>(0.1));
        int active_peaks = 0;

        for (long j = 0; j < static_cast<long>(num_bins_); ++j) {
            if (last_peak_check[j] > 0) {
                if (out_mag_check[j] < release_level) {
                    // Peak decayed
                    float freq = static_cast<float>(j) * sample_rate_ / fft_size_;
                    float mag_lin = out_mag_check[j];
                    float mag_db = 20.0f * std::log10(std::max(mag_lin, 1e-10f));
                    latest_peaks_.push_back({0, freq, mag_db, -1, mag_lin});
                    
                    last_peak_check[j] = 0.0f; // Clear the marker
                } else {
                    active_peaks++;
                }
            }
        }

        // 5. Detect peaks and track slopes
        auto mag_acc = mag.template accessor<float, 1>();
        auto phase_acc = phase.template accessor<float, 1>();
        auto prev_acc = previous_magnitude_.template accessor<float, 1>();
        auto peak_acc = peak_magnitude_.template accessor<float, 1>();
        auto target_acc = target_magnitude_.template accessor<float, 1>();
        auto start_acc = start_magnitude_.template accessor<float, 1>();
        auto env_acc = envelope_position_.template accessor<float, 1>();
        auto out_mag_acc = output_magnitude_.template accessor<float, 1>();
        auto out_phase_acc = output_phase_.template accessor<float, 1>();
        auto last_peak_bin_acc = last_written_peak_bin_.template accessor<float, 1>();

        // Calculate percentile threshold if in prominence mode
        float percentile_threshold = 0.0f;
        if (detection_mode_ == DetectionMode::PROMINENCE && use_parabolic_interp_) {
            // PASSO 1: Detectar todos os picos locais
            // PASSO 2: Calcular magnitude/fase interpolada para cada pico
            // PASSO 3: Calcular percentil das magnitudes interpoladas
            std::vector<float> interpolated_peak_mags;
            
            for (long i = 1; i < static_cast<long>(num_bins_) - 1; ++i) {
                float curr_mag = mag_acc[i];
                float left_mag = mag_acc[i - 1];
                float right_mag = mag_acc[i + 1];
                bool is_local_peak = (curr_mag > left_mag) && (curr_mag > right_mag) && (curr_mag > threshold_);
                
                if (is_local_peak) {
                    // Parabolic interpolation para magnitude refinada
                    float left_sq = left_mag * left_mag;
                    float right_sq = right_mag * right_mag;
                    float windpower = (left_mag + right_mag - 2.0f * curr_mag);
                    
                    float detune = 0.0f;
                    if (std::abs(windpower) > 1e-10f) {
                        detune = (right_sq - left_sq) / (2.0f * windpower);
                        detune = std::clamp(detune, -0.5f, 0.5f);
                    }
                    
                    // Magnitude interpolada (correção por janelamento já foi aplicada antes)
                    // Aproximação: magnitude no pico real é ligeiramente maior
                    float pidetune = static_cast<float>(M_PI) * detune;
                    float ampcorrect = 1.0f / (1.0f - 0.5f * std::cos(pidetune)); // Hann window correction
                    float interpolated_mag = curr_mag * ampcorrect;
                    
                    interpolated_peak_mags.push_back(interpolated_mag);
                }
            }
            
            // Calcular percentil
            if (!interpolated_peak_mags.empty()) {
                std::sort(interpolated_peak_mags.begin(), interpolated_peak_mags.end());
                size_t percentile_idx = static_cast<size_t>(prominence_threshold_ * (interpolated_peak_mags.size() - 1));
                percentile_threshold = interpolated_peak_mags[percentile_idx];
            }
        }

        for (long i = 1; i < static_cast<long>(num_bins_) - 1; ++i) {
            float curr_mag = mag_acc[i];
            float prev_mag = prev_acc[i];
            float left_mag = mag_acc[i - 1];
            float right_mag = mag_acc[i + 1];

            // Check if this is a local peak
            bool is_peak = (curr_mag > left_mag) && (curr_mag > right_mag) && (curr_mag > threshold_);

            bool should_trigger = false;
            float detune = 0.0f;
            float interpolated_mag = curr_mag;
            float interpolated_phase = phase_acc[i];
            
            // PASSO 2: Calcular magnitude e fase interpoladas (parabolic interpolation)
            if (is_peak && use_parabolic_interp_) {
                float left_sq = left_mag * left_mag;
                float right_sq = right_mag * right_mag;
                float windpower = (left_mag + right_mag - 2.0f * curr_mag);
                
                if (std::abs(windpower) > 1e-10f) {
                    detune = (right_sq - left_sq) / (2.0f * windpower);
                    detune = std::clamp(detune, -0.5f, 0.5f);
                    
                    // Correção de amplitude por janelamento
                    float pidetune = static_cast<float>(M_PI) * detune;
                    float sinpidetune = std::sin(pidetune);
                    float cospidetune = std::cos(pidetune);
                    float ampcorrect = 1.0f / (1.0f - 0.5f * std::cos(pidetune));
                    
                    // Magnitude interpolada
                    interpolated_mag = curr_mag * ampcorrect;
                    
                    // Fase interpolada (rotação pelo detune)
                    float left_phase = (i > 0) ? phase_acc[i - 1] : 0.0f;
                    float right_phase = (i < static_cast<long>(num_bins_) - 1) ? phase_acc[i + 1] : 0.0f;
                    
                    // Interpolação linear de fase (poderia ser melhorada)
                    if (detune > 0) {
                        interpolated_phase = phase_acc[i] + detune * (right_phase - phase_acc[i]);
                    } else {
                        interpolated_phase = phase_acc[i] + (-detune) * (phase_acc[i] - left_phase);
                    }
                }
            }
            
            // PASSO 3: Aplicar critério de filtragem (slope ou percentil)
            if (detection_mode_ == DetectionMode::PROMINENCE) {
                // Prominence mode: slope + filtro por percentil de magnitude interpolada
                
                // Update peak tracker (usando magnitude interpolada)
                if (interpolated_mag > peak_acc[i]) {
                    peak_acc[i] = interpolated_mag;
                }

                // Detect negative slope (usando magnitude interpolada)
                bool negative_slope = interpolated_mag < prev_acc[i];
                bool was_at_peak = prev_acc[i] >= (peak_acc[i] * 0.95f);
                bool envelope_complete = env_acc[i] > 0.95f;
                bool significant_increase = interpolated_mag > (out_mag_acc[i] * 1.1f);
                
                // Filtro adicional: magnitude interpolada acima do percentil
                bool above_percentile = interpolated_mag > percentile_threshold;
                
                should_trigger = is_peak && negative_slope && was_at_peak && 
                                interpolated_mag > threshold_ && envelope_complete && 
                                significant_increase && above_percentile;
            } else {
                // Original slope-based mode (usando magnitude interpolada se disponível)
                // Update peak tracker
                if (interpolated_mag > peak_acc[i]) {
                    peak_acc[i] = interpolated_mag;
                }

                // Detect negative slope
                bool negative_slope = interpolated_mag < prev_acc[i];
                bool was_at_peak = prev_acc[i] >= (peak_acc[i] * 0.95f);
                bool envelope_complete = env_acc[i] > 0.95f;
                bool significant_increase = interpolated_mag > (out_mag_acc[i] * 1.1f);
                
                should_trigger = is_peak && negative_slope && was_at_peak && 
                                interpolated_mag > threshold_ && envelope_complete && 
                                significant_increase;
            }

            if (should_trigger) {
                // Check max peaks limit
                if (max_peaks_ > 0 && active_peaks >= max_peaks_) {
                    should_trigger = false;
                }
            }

            if (should_trigger) {
                // Check minimum distance from last written peak
                bool far_enough = true;
                for (long j = 0; j < static_cast<long>(num_bins_); ++j) {
                    if (last_peak_bin_acc[j] > 0 && std::abs(static_cast<float>(i) - last_peak_bin_acc[j]) < min_peak_distance_bins_) {
                        far_enough = false;
                        break;
                    }
                }

                if (far_enough) {
                    // PASSO 4: Redistribuir energia nos bins corretos baseado no detune
                    if (use_parabolic_interp_ && std::abs(detune) > 0.01f) {
                        // Energia distribuída proporcionalmente entre bins adjacentes
                        // Se detune > 0: pico está entre bin[i] e bin[i+1]
                        // Se detune < 0: pico está entre bin[i-1] e bin[i]
                        
                        float abs_detune = std::abs(detune);
                        long target_bin = i;
                        long adjacent_bin = (detune > 0) ? (i + 1) : (i - 1);
                        
                        // Proporção de energia: (1-abs_detune) no bin principal, abs_detune no adjacente
                        float main_weight = 1.0f - abs_detune;
                        float adjacent_weight = abs_detune;
                        
                        // Bin principal
                        start_acc[target_bin] = out_mag_acc[target_bin];
                        target_acc[target_bin] = interpolated_mag * main_weight;
                        
                        auto start_phase_acc = start_phase_.template accessor<float, 1>();
                        auto target_phase_acc = target_phase_.template accessor<float, 1>();
                        start_phase_acc[target_bin] = out_phase_acc[target_bin];
                        target_phase_acc[target_bin] = interpolated_phase;
                        
                        env_acc[target_bin] = 0.0f;
                        last_peak_bin_acc[target_bin] = static_cast<float>(target_bin);
                        peak_acc[target_bin] = 0.0f;
                        
                        // Bin adjacente (se válido)
                        if (adjacent_bin > 0 && adjacent_bin < static_cast<long>(num_bins_) - 1) {
                            start_acc[adjacent_bin] = out_mag_acc[adjacent_bin];
                            target_acc[adjacent_bin] = interpolated_mag * adjacent_weight;
                            start_phase_acc[adjacent_bin] = out_phase_acc[adjacent_bin];
                            target_phase_acc[adjacent_bin] = interpolated_phase;
                            env_acc[adjacent_bin] = 0.0f;
                        }
                    } else {
                        // Sem redistribuição: escreve apenas no bin central
                        start_acc[i] = out_mag_acc[i];
                        target_acc[i] = interpolated_mag;
                        
                        auto start_phase_acc = start_phase_.template accessor<float, 1>();
                        auto target_phase_acc = target_phase_.template accessor<float, 1>();
                        start_phase_acc[i] = out_phase_acc[i];
                        target_phase_acc[i] = interpolated_phase;
                        
                        env_acc[i] = 0.0f;
                        last_peak_bin_acc[i] = static_cast<float>(i);
                        peak_acc[i] = 0.0f;
                    }
                    
                    // Record new peak bin
                    new_peak_bins.push_back(i);
                    active_peaks++;
                }
            }

            // Decay peak tracker slowly
            if (!is_peak) {
                peak_acc[i] *= 0.99f;
            }
        }

        // Store current magnitude for next frame
        previous_magnitude_ = mag.clone();

        // Force DC and Nyquist to zero
        auto out_mag = output_magnitude_.clone();
        auto out_phase = output_phase_.clone();
        
        if (num_bins_ > 0) {
            out_mag[0] = 0.0f;
            out_phase[0] = 0.0f;
        }
        if (num_bins_ > 1) {
            out_mag[static_cast<long>(num_bins_) - 1] = 0.0f;
            out_phase[static_cast<long>(num_bins_) - 1] = 0.0f;
        }

        // Apply output gain
        if (output_gain_ != static_cast<T>(1.0)) {
            out_mag *= output_gain_;
        }

        // Apply limiter
        if (use_limiter_) {
            out_mag = torch::clamp(out_mag, static_cast<T>(0.0), limiter_threshold_);
        }

        // Collect active peaks (new and sustained)
        // We use the processed out_mag (with gain and limiter) for reporting
        auto final_mag_acc = out_mag.template accessor<float, 1>();

        for (long j = 0; j < static_cast<long>(num_bins_); ++j) {
            if (last_peak_check[j] > 0) {
                // Check if it's a new peak
                bool is_new = false;
                for (int new_bin : new_peak_bins) {
                    if (new_bin == j) {
                        is_new = true;
                        break;
                    }
                }
                
                float freq = static_cast<float>(j) * sample_rate_ / fft_size_;
                float mag_lin = final_mag_acc[j]; // Use final magnitude
                float mag_db = 20.0f * std::log10(std::max(mag_lin, 1e-10f));
                
                latest_peaks_.push_back({0, freq, mag_db, is_new ? 1 : 0, mag_lin});
            }
        }
        
        // Sort peaks by magnitude (descending)
        std::sort(latest_peaks_.begin(), latest_peaks_.end(), 
            [](const PeakInfo& a, const PeakInfo& b) {
                return a.mag_lin > b.mag_lin;
            });
            
        // Assign ranks
        for (size_t i = 0; i < latest_peaks_.size(); ++i) {
            latest_peaks_[i].rank = static_cast<int>(i);
        }

        return {out_mag, out_phase};
    }
};

}  // namespace ap_spectrails
}  // namespace core
}  // namespace contorchionist

#endif  // CORE_AP_SPECTRAILS_H

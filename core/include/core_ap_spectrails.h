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
          use_parabolic_interp_(true) {
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
        for (long j = 0; j < static_cast<long>(num_bins_); ++j) {
            if (last_peak_check[j] > 0 && out_mag_check[j] < threshold_ * 0.1f) {
                last_peak_check[j] = 0.0f; // Clear the marker when magnitude has decayed
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

        for (long i = 1; i < static_cast<long>(num_bins_) - 1; ++i) {
            float curr_mag = mag_acc[i];
            float prev_mag = prev_acc[i];
            float left_mag = mag_acc[i - 1];
            float right_mag = mag_acc[i + 1];

            // Check if this is a local peak
            bool is_peak = (curr_mag > left_mag) && (curr_mag > right_mag) && (curr_mag > threshold_);

            bool should_trigger = false;
            
            if (detection_mode_ == DetectionMode::PROMINENCE) {
                // Prominence mode: relative threshold + parabolic interpolation
                // Compare with bins at distance ±2 (like sigmund~)
                float left2_mag = (i >= 2) ? mag_acc[i - 2] : 0.0f;
                float right2_mag = (i < static_cast<long>(num_bins_) - 2) ? mag_acc[i + 2] : 0.0f;
                
                // Relative threshold: peak must exceed prominence_threshold * (neighbor peaks)
                float neighbor_power = left2_mag + right2_mag;
                bool prominent_peak = is_peak && (curr_mag > prominence_threshold_ * neighbor_power);
                
                // Check envelope state
                bool envelope_complete = env_acc[i] > 0.95f;
                bool significant_increase = curr_mag > (out_mag_acc[i] * 1.1f);
                
                should_trigger = prominent_peak && envelope_complete && significant_increase;
                
                // Parabolic interpolation for refined frequency (if enabled)
                if (should_trigger && use_parabolic_interp_) {
                    // Calculate detune using quadratic interpolation
                    // detune = ((right² - left²)) / (2 * (2*center - left - right))
                    float left_sq = left_mag * left_mag;
                    float right_sq = right_mag * right_mag;
                    float windpower = (left_mag + right_mag - 2.0f * curr_mag);
                    
                    if (std::abs(windpower) > 1e-10f) {
                        float detune = (right_sq - left_sq) / (2.0f * windpower);
                        detune = std::clamp(detune, -0.5f, 0.5f);
                        // Store detune for potential future use (frequency refinement)
                        // For now, we just detect at bin centers
                    }
                }
            } else {
                // Original slope-based mode
                // Update peak tracker
                if (curr_mag > peak_acc[i]) {
                    peak_acc[i] = curr_mag;
                }

                // Detect negative slope (started decaying from peak)
                bool negative_slope = curr_mag < prev_mag;
                bool was_at_peak = prev_mag >= (peak_acc[i] * 0.95f); // Within 5% of tracked peak

                // Trigger write: local peak, negative slope, was near maximum
                // CRITICAL: Only trigger if envelope is nearly complete (>0.95) to avoid re-triggering during attack
                bool envelope_complete = env_acc[i] > 0.95f;
                bool significant_increase = curr_mag > (out_mag_acc[i] * 1.1f);
                
                should_trigger = is_peak && negative_slope && was_at_peak && curr_mag > threshold_ && envelope_complete && significant_increase;
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
                    // Start new envelope: save current position as start, set new target
                    start_acc[i] = out_mag_acc[i];
                    target_acc[i] = curr_mag;
                    
                    // Phase envelope
                    auto start_phase_acc = start_phase_.template accessor<float, 1>();
                    auto target_phase_acc = target_phase_.template accessor<float, 1>();
                    start_phase_acc[i] = out_phase_acc[i];
                    target_phase_acc[i] = phase_acc[i];
                    
                    env_acc[i] = 0.0f; // Reset envelope to beginning
                    last_peak_bin_acc[i] = static_cast<float>(i);
                    peak_acc[i] = 0.0f; // Reset peak tracker for this bin
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

        return {out_mag, out_phase};
    }

private:
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
};

}  // namespace ap_spectrails
}  // namespace core
}  // namespace contorchionist

#endif  // CORE_AP_SPECTRAILS_H

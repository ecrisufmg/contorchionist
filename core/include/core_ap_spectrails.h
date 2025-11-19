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
          fft_size_(static_cast<T>((num_bins - 1) * 2)) {
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

        // 1. Apply attack ramp to reach target magnitudes
        auto distance_to_target = target_magnitude_ - output_magnitude_;
        output_magnitude_ += distance_to_target * attack_;

        // 2. Decay all bins
        output_magnitude_ *= decay_;
        target_magnitude_ *= decay_;

        // 3. Age out old peak markers (clear peaks that have decayed away)
        auto out_mag_check = output_magnitude_.template accessor<float, 1>();
        auto last_peak_check = last_written_peak_bin_.template accessor<float, 1>();
        for (long j = 0; j < static_cast<long>(num_bins_); ++j) {
            if (last_peak_check[j] > 0 && out_mag_check[j] < threshold_ * 0.1f) {
                last_peak_check[j] = 0.0f; // Clear the marker when magnitude has decayed
            }
        }

        // 4. Detect peaks and track slopes
        auto mag_acc = mag.template accessor<float, 1>();
        auto phase_acc = phase.template accessor<float, 1>();
        auto prev_acc = previous_magnitude_.template accessor<float, 1>();
        auto peak_acc = peak_magnitude_.template accessor<float, 1>();
        auto target_acc = target_magnitude_.template accessor<float, 1>();
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

            // Update peak tracker
            if (curr_mag > peak_acc[i]) {
                peak_acc[i] = curr_mag;
            }

            // Detect negative slope (started decaying from peak)
            bool negative_slope = curr_mag < prev_mag;
            bool was_at_peak = prev_mag >= (peak_acc[i] * 0.95f); // Within 5% of tracked peak

            // Trigger write: local peak, negative slope, was near maximum
            if (is_peak && negative_slope && was_at_peak && curr_mag > threshold_) {
                // Check minimum distance from last written peak
                bool far_enough = true;
                for (long j = 0; j < static_cast<long>(num_bins_); ++j) {
                    if (last_peak_bin_acc[j] > 0 && std::abs(static_cast<float>(i) - last_peak_bin_acc[j]) < min_peak_distance_bins_) {
                        far_enough = false;
                        break;
                    }
                }

                if (far_enough) {
                    // Set target magnitude (not immediate) and lock phase
                    target_acc[i] = curr_mag;
                    out_phase_acc[i] = phase_acc[i];
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
        previous_magnitude_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
        peak_magnitude_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
        last_written_peak_bin_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
    }

    size_t num_bins_;
    torch::Device device_;
    torch::Tensor output_magnitude_;
    torch::Tensor output_phase_;
    torch::Tensor target_magnitude_;
    torch::Tensor previous_magnitude_;
    torch::Tensor peak_magnitude_;
    torch::Tensor last_written_peak_bin_;

    T threshold_;
    T attack_;
    T decay_;
    T min_peak_distance_bins_;
    T sample_rate_;
    T fft_size_;
};

}  // namespace ap_spectrails
}  // namespace core
}  // namespace contorchionist

#endif  // CORE_AP_SPECTRAILS_H
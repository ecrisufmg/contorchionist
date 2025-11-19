#ifndef CORE_AP_SPECTRAILS_H
#define CORE_AP_SPECTRAILS_H

#include <torch/torch.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace contorchionist {
namespace core {
namespace ap_spectrails {

/**
 * @brief Minimal spectral trails processor used for debugging.
 *
 * The processor holds a copy of the previous spectral frame (magnitude/phase)
 * and applies:
 *  - reinforcement whenever the current magnitude is above the threshold
 *  - exponential decay whenever the magnitude falls below the threshold
 *
 * All computations are frame-by-frame and operate on RFFT half-spectra
 * (N/2 + 1 bins). DC (bin 0) and Nyquist (last bin) phases are forced to 0.
 */
template <typename T>
class SpectralTrailsProcessor {
public:
        explicit SpectralTrailsProcessor(size_t num_bins,
                                                                         torch::Device device = torch::kCPU)
                : num_bins_(num_bins),
                    device_(device),
                    threshold_(static_cast<T>(0.01)),
                    attack_(static_cast<T>(0.8)),
                    phase_attack_(static_cast<T>(0.8)),
                    decay_(static_cast<T>(0.995)),
                    phase_attack_locked_(false),
                    slope_hold_frames_(2) {
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
        allocate_memory();
    }

    void reset_memory() {
        memory_magnitude_.zero_();
        memory_phase_.zero_();
        previous_magnitude_.zero_();
        slope_counter_.zero_();
    }

    void set_threshold(T value) {
        threshold_ = std::max(static_cast<T>(0), value);
    }

    void set_attack(T value) {
        attack_ = std::clamp(value, static_cast<T>(0), static_cast<T>(1));
        if (!phase_attack_locked_) {
            phase_attack_ = attack_;
        }
    }

    void set_phase_attack(T value) {
        phase_attack_ = std::clamp(value, static_cast<T>(0), static_cast<T>(1));
        phase_attack_locked_ = true;
    }

    void set_decay(T value) {
        decay_ = std::clamp(value, static_cast<T>(0), static_cast<T>(1));
    }

    void set_slope_hold_frames(int frames) {
        slope_hold_frames_ = std::max(1, frames);
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

        auto valid_mask = valid_mask_bool_;

        auto decreasing = (mag - previous_magnitude_) < 0.0f;
        auto gated_bins = valid_mask & (mag >= threshold_);
        auto slope_reset = torch::zeros_like(slope_counter_);
        slope_counter_ = torch::where(decreasing & gated_bins,
                                      slope_counter_ + 1.0f,
                                      slope_reset);

        auto slope_frames_tensor = torch::full_like(slope_counter_,
                                                    static_cast<float>(slope_hold_frames_));
        auto slope_event = slope_counter_.eq(slope_frames_tensor);
        auto above_memory = mag > memory_magnitude_;
        auto trigger = slope_event & gated_bins & above_memory;

        auto reinforced_mag = torch::lerp(memory_magnitude_, mag, attack_);
        auto decayed = memory_magnitude_ * decay_;
        memory_magnitude_ = torch::where(trigger, reinforced_mag, memory_magnitude_);
        auto decay_mask = (~trigger) & valid_mask;
        memory_magnitude_ = torch::where(decay_mask, decayed, memory_magnitude_);

        auto phase_diff = torch::atan2(torch::sin(phase - memory_phase_),
                                       torch::cos(phase - memory_phase_));
        auto reinforced_phase = memory_phase_ + phase_attack_ * phase_diff;
        memory_phase_ = torch::where(trigger, reinforced_phase, memory_phase_);

        // Force DC and Nyquist phases to zero for stability
        if (num_bins_ > 0) {
            memory_phase_[0] = 0.0f;
        }
        if (num_bins_ > 1) {
            memory_phase_[static_cast<long>(num_bins_) - 1] = 0.0f;
        }

        previous_magnitude_ = mag;
        auto out_mag = (memory_magnitude_ * valid_mask_float_).clone();
        auto out_phase = memory_phase_.clone();
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
        memory_magnitude_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
        memory_phase_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
        previous_magnitude_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
        slope_counter_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
        valid_mask_bool_ = torch::ones({static_cast<long>(num_bins_)},
                                       torch::TensorOptions().dtype(torch::kBool).device(device_));
        valid_mask_float_ = torch::ones({static_cast<long>(num_bins_)}, opts);
        if (num_bins_ > 0) {
            valid_mask_bool_[0] = false;
            valid_mask_float_[0] = 0.0f;
        }
        if (num_bins_ > 1) {
            valid_mask_bool_[static_cast<long>(num_bins_) - 1] = false;
            valid_mask_float_[static_cast<long>(num_bins_) - 1] = 0.0f;
        }
    }

    size_t num_bins_;
    torch::Device device_;
    torch::Tensor memory_magnitude_;
    torch::Tensor memory_phase_;
    torch::Tensor previous_magnitude_;
    torch::Tensor slope_counter_;
    torch::Tensor valid_mask_bool_;
    torch::Tensor valid_mask_float_;

    T threshold_;
    T attack_;
    T phase_attack_;
    T decay_;
    bool phase_attack_locked_;
    int slope_hold_frames_;
};

}  // namespace ap_spectrails
}  // namespace core
}  // namespace contorchionist

#endif  // CORE_AP_SPECTRAILS_H

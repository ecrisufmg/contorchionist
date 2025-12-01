#ifndef CORE_AP_COMPRESSOR_H
#define CORE_AP_COMPRESSOR_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <torch/torch.h>

namespace contorchionist {
namespace core {
namespace ap_compressor {

template<typename T = float>
class Compressor {
public:
    Compressor(float sampleRate = 44100.0f, int rmsWindowSize = 1024)
        : sampleRate_(sampleRate), rmsWindowSize_(rmsWindowSize),
          sum_rms_(0.0), read_pos_(0), current_gain_(1.0) {
        resize_rms_buffer();
    }

    void setSampleRate(float sr) {
        if (sr > 0) {
            sampleRate_ = sr;
        }
    }

    void setRMSWindowSize(int size) {
        if (size != rmsWindowSize_ && size > 0) {
            rmsWindowSize_ = size;
            resize_rms_buffer();
        }
    }

    void reset() {
        if (rms_buffer_tensor_.defined()) {
            rms_buffer_tensor_.zero_();
        }
        sum_rms_ = 0.0;
        read_pos_ = 0;
        current_gain_ = 1.0f;
    }

    // Main process function
    // Assumes parameters are constant for the block (control rate)
    void process(const T* input, T* output, int blockSize,
                 float thresh_db, float ratio, float curve_mode,
                 float attack_ms, float release_ms) {
        
        // Ensure buffer is on CPU for direct access
        // In a full GPU implementation, we would keep it on GPU and use a kernel.
        // Here we use the tensor as storage.
        float* rms_data = rms_buffer_tensor_.data_ptr<float>();

        // Calculate coefficients
        // coeff = 1 - exp(-1 / (time_sec * sr))
        // If time is 0, coeff is 1 (instant)
        float attack_coeff = (attack_ms <= 0) ? 1.0f : 1.0f - std::exp(-1.0f / ((attack_ms / 1000.0f) * sampleRate_));
        float release_coeff = (release_ms <= 0) ? 1.0f : 1.0f - std::exp(-1.0f / ((release_ms / 1000.0f) * sampleRate_));

        for (int i = 0; i < blockSize; ++i) {
            T in_sample = input[i];
            
            // 1. RMS Detection (Sliding Window)
            T val_sq = in_sample * in_sample;
            T old_val = rms_data[read_pos_];
            
            // Update sum
            sum_rms_ -= old_val;
            sum_rms_ += val_sq;
            
            // Update buffer
            rms_data[read_pos_] = val_sq;
            
            // Advance pointer
            read_pos_++;
            if (read_pos_ >= rmsWindowSize_) read_pos_ = 0;

            // Calculate RMS
            // Ensure sum is non-negative (floating point drift protection)
            if (sum_rms_ < 0) sum_rms_ = 0;
            
            T mean_sq = sum_rms_ / static_cast<double>(rmsWindowSize_);
            T env_lin = std::sqrt(mean_sq);

            // 2. dB Conversion
            T level_db = (env_lin < 1e-7f) ? -100.0f : 20.0f * std::log10(env_lin);

            // 3. Gain Calculation
            T target_gain = 1.0f;
            
            // If signal is above threshold
            if (level_db > thresh_db) {
                // A. Normalization (0 to 1)
                // t = 0 at threshold, t = 1 at 0dB
                T range_in = 0.0f - thresh_db;
                T t = (range_in > 1e-9f) ? (level_db - thresh_db) / range_in : 0.0f;
                
                // Clamp t
                t = std::max(T(0), std::min(T(1), t));

                // B. Curve
                T t_curved = calculate_curve(t, curve_mode);

                // C. Map to Output
                // Standard Downward Compression Logic
                // Min output (at threshold) = thresh_db
                // Max output (at 0dB input) = thresh_db + (range_in / ratio)
                T out_min = thresh_db;
                T out_max = thresh_db + (range_in / ratio);
                T range_out = out_max - out_min;
                
                T db_desired = out_min + (t_curved * range_out);
                
                // D. Gain Reduction
                T diff_db = db_desired - level_db;
                
                target_gain = std::pow(10.0f, diff_db / 20.0f);
            }

            // 4. Ballistics (Attack/Release)
            if (target_gain < current_gain_) {
                // Attack (Gain is reducing)
                current_gain_ += (target_gain - current_gain_) * attack_coeff;
            } else {
                // Release (Gain is increasing back to 1.0)
                current_gain_ += (target_gain - current_gain_) * release_coeff;
            }

            // 5. Output
            output[i] = in_sample * current_gain_;
        }
        
        // Periodic drift correction for sum_rms_ could be added here if needed,
        // but for typical block sizes and float precision, it might be okay.
        // To be safe, we could re-sum the buffer every N blocks.
    }

    // Helper for curve
    T calculate_curve(T t, float mode) {
        if (std::abs(mode) < 0.001f) return t; // Linear
        
        if (mode > 0) {
            // Ease In (Exponential)
            return std::pow(t, mode + 1.0f); // mode 1 -> power 2? Pseudocode says "t elevated a mode". 
            // If mode is 1.0, it should be linear? No, usually mode 1 means power 1 (linear).
            // Pseudocode: "SE modo > 0 ENTÃO: RETORNAR t elevado a modo"
            // If mode is 2.0 -> t^2.
            // If mode is 1.0 -> t^1 (linear).
            // But the check "SE modo == 0" handles linear.
            // So if mode is 2.0, it's t^2.
            return std::pow(t, mode);
        } else {
            // Ease Out (Inverse)
            // RETORNAR 1 - ((1 - t) elevado a abs(modo))
            return 1.0f - std::pow(1.0f - t, std::abs(mode));
        }
    }

private:
    float sampleRate_;
    int rmsWindowSize_;
    torch::Tensor rms_buffer_tensor_;
    double sum_rms_; // Use double for accumulation to reduce drift
    int read_pos_;
    T current_gain_;

    void resize_rms_buffer() {
        rms_buffer_tensor_ = torch::zeros({rmsWindowSize_}, torch::kFloat32);
        sum_rms_ = 0.0;
        read_pos_ = 0;
    }
};

} // namespace ap_compressor
} // namespace core
} // namespace contorchionist

#endif // CORE_AP_COMPRESSOR_H

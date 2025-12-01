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
          sum_rms_(0.0), read_pos_(0), current_gain_(1.0),
          lookahead_samples_(0), delay_write_pos_(0),
          makeup_gain_db_(0.0f), auto_makeup_(false) {
        resize_rms_buffer();
        resize_delay_buffer();
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

    void setLookahead(float ms) {
        int samples = static_cast<int>((ms / 1000.0f) * sampleRate_);
        if (samples < 0) samples = 0;
        if (samples != lookahead_samples_) {
            lookahead_samples_ = samples;
            resize_delay_buffer();
        }
    }

    void setMakeup(float db) {
        makeup_gain_db_ = db;
    }

    void setAutoMakeup(bool enable) {
        auto_makeup_ = enable;
    }

    void reset() {
        if (rms_buffer_tensor_.defined()) {
            rms_buffer_tensor_.zero_();
        }
        std::fill(delay_buffer_.begin(), delay_buffer_.end(), 0.0f);
        sum_rms_ = 0.0;
        read_pos_ = 0;
        delay_write_pos_ = 0;
        current_gain_ = 1.0f;
    }

    // Main process function
    // Assumes parameters are constant for the block (control rate)
    void process(const T* input, T* output, int blockSize,
                 float thresh_db, float ratio, float curve_mode,
                 float attack_ms, float release_ms) {
        
        // Ensure buffer is on CPU for direct access
        float* rms_data = rms_buffer_tensor_.data_ptr<float>();

        // Calculate coefficients
        float attack_coeff = (attack_ms <= 0) ? 1.0f : 1.0f - std::exp(-1.0f / ((attack_ms / 1000.0f) * sampleRate_));
        float release_coeff = (release_ms <= 0) ? 1.0f : 1.0f - std::exp(-1.0f / ((release_ms / 1000.0f) * sampleRate_));

        // Calculate Makeup Gain
        float final_makeup_linear = 1.0f;
        if (auto_makeup_) {
            // Auto Makeup: Compensate for the loss at 0dB input
            // Loss = 0 - (Thresh + (0-Thresh)/Ratio) = -Thresh * (1 - 1/Ratio)
            // Gain = -Loss
            if (ratio > 0.0001f) {
                float loss_db = -thresh_db * (1.0f - 1.0f / ratio);
                final_makeup_linear = std::pow(10.0f, loss_db / 20.0f);
            }
        } else {
            final_makeup_linear = std::pow(10.0f, makeup_gain_db_ / 20.0f);
        }

        for (int i = 0; i < blockSize; ++i) {
            T in_sample = input[i];
            
            // --- Sidechain Path (RMS & Gain Calc) ---

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
            if (sum_rms_ < 0) sum_rms_ = 0;
            T mean_sq = sum_rms_ / static_cast<double>(rmsWindowSize_);
            T env_lin = std::sqrt(mean_sq);

            // 2. dB Conversion
            T level_db = (env_lin < 1e-7f) ? -100.0f : 20.0f * std::log10(env_lin);

            // 3. Gain Calculation
            T target_gain = 1.0f;
            
            if (level_db > thresh_db) {
                T range_in = 0.0f - thresh_db;
                T t = (range_in > 1e-9f) ? (level_db - thresh_db) / range_in : 0.0f;
                t = std::max(T(0), std::min(T(1), t));

                T t_curved = calculate_curve(t, curve_mode);

                // Standard Downward Compression
                T out_min = thresh_db;
                T out_max = thresh_db + (range_in / ratio);
                T range_out = out_max - out_min;
                
                T db_desired = out_min + (t_curved * range_out);
                T diff_db = db_desired - level_db;
                
                target_gain = std::pow(10.0f, diff_db / 20.0f);
            }

            // 4. Ballistics
            if (target_gain < current_gain_) {
                current_gain_ += (target_gain - current_gain_) * attack_coeff;
            } else {
                current_gain_ += (target_gain - current_gain_) * release_coeff;
            }

            // --- Audio Path (Delay & Apply Gain) ---
            
            T sample_to_process = in_sample;

            if (lookahead_samples_ > 0) {
                // Write to delay buffer
                delay_buffer_[delay_write_pos_] = in_sample;
                
                // Read from delay buffer
                // read_ptr = (write_ptr + capacity - lookahead) % capacity
                int delay_read_pos = (delay_write_pos_ + delay_buffer_.size() - lookahead_samples_) % delay_buffer_.size();
                sample_to_process = delay_buffer_[delay_read_pos];

                // Advance write pointer
                delay_write_pos_++;
                if (delay_write_pos_ >= delay_buffer_.size()) delay_write_pos_ = 0;
            }

            // 5. Output
            output[i] = sample_to_process * current_gain_ * final_makeup_linear;
        }
    }

    // Helper for curve
    T calculate_curve(T t, float mode) {
        if (std::abs(mode) < 0.001f) return t; // Linear
        
        if (mode > 0) {
            // Ease In (Exponential)
            return std::pow(t, mode + 1.0f);
        } else {
            // Ease Out (Inverse)
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

    // Lookahead & Makeup
    std::vector<T> delay_buffer_;
    int delay_write_pos_;
    int lookahead_samples_;
    float makeup_gain_db_;
    bool auto_makeup_;

    void resize_rms_buffer() {
        rms_buffer_tensor_ = torch::zeros({rmsWindowSize_}, torch::kFloat32);
        sum_rms_ = 0.0;
        read_pos_ = 0;
    }

    void resize_delay_buffer() {
        int size = (lookahead_samples_ > 0) ? lookahead_samples_ + 1 : 1;
        if (delay_buffer_.size() < static_cast<size_t>(size)) {
            delay_buffer_.resize(size, 0.0f);
        }
        delay_write_pos_ = 0;
    }
};

} // namespace ap_compressor
} // namespace core
} // namespace contorchionist

#endif // CORE_AP_COMPRESSOR_H

#ifndef CORE_AP_MICARRAYSPECANALYZER_HPP
#define CORE_AP_MICARRAYSPECANALYZER_HPP

#include <torch/torch.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace contorchionist {
namespace core {
namespace ap_micarrayspecanalyzer {

struct AnalysisResult {
    int band_index;
    float angle_deg;
    float strength;
    float dominant_freq_hz;
};

struct BandConfig {
    float min_hz;
    float max_hz;
};

struct MicInfo {
    int id;
    float physical_angle_deg;
    float distance;
    float pickup_angle_deg; // Calculated after calibration
};

class MicArraySpecAnalyzer {
private:
    int num_mics_;
    int fft_size_;
    int num_bins_;
    torch::Device device_;
    
    // State
    std::vector<MicInfo> mics_;
    std::vector<float> speakers_;
    std::vector<BandConfig> bands_;
    
    // Tensors
    torch::Tensor projection_matrix_; // [num_mics, 2]
    torch::Tensor band_masks_;        // [num_bands, num_bins]
    
    // DSP Parameters
    float sample_rate_;
    float overlap_factor_;

    // Helper: Normalize angle to 0-360
    float normalize_angle(float angle) {
        angle = std::fmod(angle, 360.0f);
        if (angle < 0) angle += 360.0f;
        return angle;
    }

    // Helper: Find nearest speaker angle
    float find_nearest_speaker(float mic_angle) {
        if (speakers_.empty()) return mic_angle + 180.0f; // Default to opposite if no speakers
        
        float min_diff = 360.0f;
        float best_speaker = 0.0f;
        
        for (float spk : speakers_) {
            float diff = std::abs(normalize_angle(mic_angle) - normalize_angle(spk));
            if (diff > 180.0f) diff = 360.0f - diff;
            
            if (diff < min_diff) {
                min_diff = diff;
                best_speaker = spk;
            }
        }
        return best_speaker;
    }

public:
    MicArraySpecAnalyzer(int num_mics, int fft_size, torch::Device device = torch::kCPU)
        : num_mics_(num_mics), fft_size_(fft_size), device_(device),
          sample_rate_(48000.0f), overlap_factor_(1.0f) {
        
        num_bins_ = fft_size_ / 2 + 1;
        mics_.resize(num_mics_);
        
        // Initialize mics with default values
        for (int i = 0; i < num_mics_; ++i) {
            mics_[i] = {i, 0.0f, 1.0f, 0.0f};
        }
        
        // Initialize tensors
        projection_matrix_ = torch::zeros({num_mics_, 2}, torch::TensorOptions().device(device_));
        update_projection_matrix();
    }

    void set_sample_rate(float sr) { 
        if (sample_rate_ != sr) {
            sample_rate_ = sr;
            recompute_masks();
        }
    }
    void set_overlap_factor(float of) { overlap_factor_ = of; }

    void resize(int fft_size) {
        if (fft_size_ != fft_size) {
            fft_size_ = fft_size;
            num_bins_ = fft_size_ / 2 + 1;
            recompute_masks();
        }
    }

    void set_mic_geometry(int id, float angle, float dist) {
        if (id >= 0 && id < num_mics_) {
            mics_[id].id = id;
            mics_[id].physical_angle_deg = angle;
            mics_[id].distance = dist;
            // Default pickup is opposite to physical (cardioid tail pointing away)
            // This will be overwritten by calibrate()
            mics_[id].pickup_angle_deg = angle; 
        }
    }

    void register_speaker(float angle) {
        speakers_.push_back(angle);
    }
    
    void clear_speakers() {
        speakers_.clear();
    }

    void set_bands(const std::vector<BandConfig>& new_bands) {
        bands_ = new_bands;
        recompute_masks();
    }

    // Null-Point Steering Calibration
    void calibrate() {
        if (speakers_.empty()) {
            // If no speakers registered, assume standard outward facing array
            // Pickup direction = Physical direction
            for (auto& mic : mics_) {
                mic.pickup_angle_deg = mic.physical_angle_deg;
            }
        } else {
            // Null-Point Steering
            for (auto& mic : mics_) {
                // Find nearest speaker (where the tail points)
                float nearest_spk = find_nearest_speaker(mic.physical_angle_deg);
                
                // If tail points to speaker, capsule points opposite
                // BUT, the user provides Physical Angle as "where capsule points".
                // The logic in MD says: "If the user points the tail at a speaker, the Physical_Angle (capsule) is naturally opposite."
                // And: "Mathematically, the software considers the 'Pickup Direction' as the capsule direction."
                
                // Wait, if the user ALREADY points the tail at the speaker, then the Physical Angle they input 
                // (assuming they input where the capsule points) is ALREADY the pickup direction.
                // The calibration logic in the MD says: "Logic: Match mic tails to nearest speakers to determine the Pickup_Angle."
                
                // Let's interpret: 
                // We want to optimize the vector math. 
                // If we are doing "Null Point Steering", we are physically orienting the mic.
                // The software just needs to know the resulting pickup direction.
                // If the user inputs the physical angle of the capsule, that IS the pickup direction.
                
                // However, the MD says: "Null-Point Steering (Auto-Calibration): Logic: Match mic tails to nearest speakers..."
                // This implies the software might ADJUST the angle if the user was imprecise?
                // OR, it implies we calculate the pickup angle BASED on the speaker location, assuming the user did their job.
                
                // Let's assume the latter: The user placed the mic roughly. We snap the "tail" to the exact speaker location.
                // Tail = Nearest Speaker.
                // Pickup (Capsule) = Tail + 180.
                
                mic.pickup_angle_deg = nearest_spk + 180.0f;
            }
        }
        update_projection_matrix();
    }

    void update_projection_matrix() {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
        torch::Tensor new_proj = torch::zeros({num_mics_, 2}, opts);
        auto acc = new_proj.accessor<float, 2>();
        
        float deg2rad = M_PI / 180.0f;
        
        for (int i = 0; i < num_mics_; ++i) {
            float angle_rad = mics_[i].pickup_angle_deg * deg2rad;
            // Weight is 1.0 as per updated spec
            float weight = 1.0f;
            
            // Col 0: X = sin(theta) (Standard audio polar: 0deg is North/Front/Y? No, usually 0 is East in math, but in audio 0 is often Front)
            // MD says: Col 0 = sin(theta), Col 1 = cos(theta).
            // If theta=0 (Front), sin=0, cos=1 -> (0, 1) -> Y axis. Correct.
            // If theta=90 (Right), sin=1, cos=0 -> (1, 0) -> X axis. Correct.
            
            acc[i][0] = weight * std::sin(angle_rad);
            acc[i][1] = weight * std::cos(angle_rad);
        }
        
        projection_matrix_ = new_proj;
    }

    void recompute_masks() {
        if (bands_.empty()) return;
        
        int num_bands = bands_.size();
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
        band_masks_ = torch::zeros({num_bands, num_bins_}, opts);
        auto acc = band_masks_.accessor<float, 2>();
        
        float bin_width_hz = sample_rate_ / fft_size_;
        
        for (int b = 0; b < num_bands; ++b) {
            float min_hz = bands_[b].min_hz;
            float max_hz = bands_[b].max_hz;
            
            // Convert Hz to fractional bins
            float b_start = min_hz / bin_width_hz;
            float b_end = max_hz / bin_width_hz;
            
            for (int k = 0; k < num_bins_; ++k) {
                float k_min = (float)k - 0.5f;
                float k_max = (float)k + 0.5f;
                
                float overlap_start = std::max(k_min, b_start);
                float overlap_end = std::min(k_max, b_end);
                
                float overlap = std::max(0.0f, overlap_end - overlap_start);
                acc[b][k] = overlap;
            }
        }
    }

    std::vector<AnalysisResult> process_bands(const torch::Tensor& magnitude_input) {
        // magnitude_input shape: [num_mics, num_bins]
        // We need to process all bands.
        
        std::vector<AnalysisResult> results;
        if (bands_.empty() || !band_masks_.defined()) return results;
        
        // Ensure input is on correct device
        auto mag = magnitude_input.to(device_);
        
        // Normalize by overlap factor if needed (usually RFFT is unnormalized)
        if (overlap_factor_ > 1.0f) {
            mag = mag / overlap_factor_;
        }

        // 1. Apply Masks
        // We want to compute RMS for each band for each mic.
        // RMS = sqrt( sum(mag^2 * mask) / sum(mask) ) ? 
        // Or just sum(mag * mask) for "Strength"? 
        // MD says: "Batch RMS: Calculate RMS of the masked band".
        // Let's do weighted RMS.
        
        // band_masks_: [num_bands, num_bins]
        // mag: [num_mics, num_bins]
        
        // We can use matrix multiplication or broadcasting.
        // Let's iterate for simplicity and clarity first, or use batched ops if possible.
        // Since num_bands is likely small, iteration is fine.
        
        // Pre-calculate squared magnitude for RMS
        auto mag_sq = mag.pow(2);
        
        for (int b = 0; b < bands_.size(); ++b) {
            auto mask = band_masks_[b]; // [num_bins]
            
            // Weighted Sum of Squares: sum(mag^2 * mask)
            // Shape: [num_mics, num_bins] * [num_bins] (broadcast) -> sum dim 1 -> [num_mics]
            auto weighted_sq = mag_sq * mask; 
            auto sum_sq = weighted_sq.sum(1); // [num_mics]
            
            // Normalize by mask sum (effective bandwidth in bins)
            float mask_sum = mask.sum().item<float>();
            if (mask_sum < 0.0001f) mask_sum = 1.0f;
            
            auto rms_vec = torch::sqrt(sum_sq / mask_sum); // [num_mics]
            
            // 2. Projection
            // rms_vec: [num_mics]
            // projection_matrix_: [num_mics, 2]
            // Result = rms_vec @ projection_matrix_ -> [2]
            
            auto proj = torch::matmul(rms_vec, projection_matrix_); // [2]
            float x = proj[0].item<float>();
            float y = proj[1].item<float>();
            
            // 3. Polar Conversion
            float strength = std::sqrt(x*x + y*y);
            float angle_rad = std::atan2(x, y); // atan2(x, y) gives angle from Y axis (North) if x=sin, y=cos
            float angle_deg = angle_rad * 180.0f / M_PI;
            angle_deg = normalize_angle(angle_deg);
            
            // 4. Parabolic Interpolation for Frequency
            // We need the global peak within the band.
            // Sum magnitudes across all mics: [num_bins]
            auto global_mag = mag.sum(0); // [num_bins]
            auto masked_global_mag = global_mag * mask;
            
            // Find max bin
            int peak_bin = masked_global_mag.argmax().item<int>();
            
            // Detune
            float freq_hz = 0.0f;
            if (peak_bin > 0 && peak_bin < num_bins_ - 1) {
                float m_i = masked_global_mag[peak_bin].item<float>();
                float m_left = masked_global_mag[peak_bin - 1].item<float>();
                float m_right = masked_global_mag[peak_bin + 1].item<float>();
                
                float denom = 2.0f * (m_left + m_right - 2.0f * m_i);
                float delta = 0.0f;
                if (std::abs(denom) > 1e-9f) {
                    delta = (m_right*m_right - m_left*m_left) / denom; // Using squared version from MD
                    // Wait, MD formula: (M_{i+1}^2 - M_{i-1}^2) / (2 * (M_{i-1} + M_{i+1} - 2 M_i))
                    // Note: The denominator usually uses linear magnitudes for standard parabolic, 
                    // but the MD specifies this exact formula. I will stick to the MD.
                }
                
                freq_hz = (float(peak_bin) + delta) * sample_rate_ / fft_size_;
            } else {
                freq_hz = float(peak_bin) * sample_rate_ / fft_size_;
            }
            
            results.push_back({b, angle_deg, strength, freq_hz});
        }
        
        return results;
    }
};

} // namespace ap_micarrayspecanalyzer
} // namespace core
} // namespace contorchionist

#endif // CORE_AP_MICARRAYSPECANALYZER_HPP

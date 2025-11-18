#ifndef CORE_AP_SPECTRAILS_H
#define CORE_AP_SPECTRAILS_H

#include <torch/torch.h>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <cmath>

namespace contorchionist {
    namespace core {
        namespace ap_spectrails {

/**
 * @brief SpectralTrailsProcessor
 * 
 * Implements a spectral memory/trail effect with threshold-based reinforcement,
 * attack smoothing, and decay over time. Works with torch tensors for GPU support.
 * 
 * Input: Magnitude/Power/dB bins (N/2+1 for RFFT) and Phase bins
 * Output: Processed Magnitude/Power/dB bins and Phase bins
 * 
 * The processor maintains a memory of the spectrum and applies:
 * 1. Universal decay on all bins
 * 2. Selective reinforcement (attack) for bins above threshold
 * 3. Optional limiting to prevent clipping
 * 4. Phase locking when bins are reinforced
 */
template<typename T>
class SpectralTrailsProcessor {
public:
    /**
     * @brief Constructor
     * @param num_bins Number of spectral bins (typically fft_size/2 + 1)
     * @param device Torch device (CPU, CUDA, etc.)
     * @param verbose Enable verbose logging
     */
    explicit SpectralTrailsProcessor(
        size_t num_bins,
        torch::Device device = torch::kCPU,
        bool verbose = false
    );

    ~SpectralTrailsProcessor();

    // --- Processing ---
    
    /**
     * @brief Process a frame of spectral data
     * @param magnitude_input Magnitude, power, or dB values (N/2+1 bins)
     * @param phase_input Phase values (N/2+1 bins)
     * @return Vector containing [processed_magnitude, processed_phase]
     */
    std::vector<torch::Tensor> process_frame(
        const torch::Tensor& magnitude_input,
        const torch::Tensor& phase_input
    );

    // --- Configuration Setters ---

    /**
     * @brief Set the threshold for reinforcement
     * @param threshold Threshold value (interpretation depends on input format)
     */
    void set_threshold(T threshold);

    /**
     * @brief Set the attack coefficient (0.0 to 1.0)
     * Higher values = faster attack, more of the new signal
     * @param attack Attack coefficient
     */
    void set_attack(T attack);

    /**
     * @brief Set the attack coefficient for phase interpolation.
     *        Defaults to the same value as magnitude attack.
     * @param attack_phase Phase attack coefficient
     */
    void set_phase_attack(T attack_phase);

    /**
     * @brief Set the decay factor (0.0 to 1.0)
     * Higher values = slower decay (closer to 1.0 means longer trails)
     * @param decay Decay factor per frame
     */
    void set_decay(T decay);

    /**
     * @brief Sets the decay factor using a time in seconds for a -6 dB drop.
     * @param time_s Time, in seconds, for the magnitude to reach half its value.
     * @param sample_rate The current sample rate.
     * @param hop_size The hop size between frames.
     */
    void set_decay_time_s(T time_s, T sample_rate, T hop_size);

    /**
     * @brief Enable or disable the limiter
     * When enabled, prevents bins from exceeding max_value
     * @param enabled Enable limiter
     */
    void set_limiter_enabled(bool enabled);

    /**
     * @brief Set the maximum value for the limiter
     * @param max_value Maximum allowed value
     */
    void set_max_value(T max_value);

    /**
     * @brief Configure how strongly the attack adapts per-bin based on the difference between input and memory.
     *        A value of 0 retains the global attack, higher values move the attack towards 1.0 more aggressively.
     * @param rate Non-negative adaptation rate.
     */
    void set_attack_dynamic_rate(T rate);

    /**
     * @brief Set the attack coefficient used for onset bins (memory below onset floor).
     * @param attack_onset Attack value applied to new/onset bins.
     */
    void set_attack_onset(T attack_onset);

    /**
     * @brief Set the magnitude floor under which a bin is considered a new onset.
     * @param floor Non-negative magnitude floor.
     */
    void set_onset_floor(T floor);

    /**
     * @brief Configure the number of consecutive frames below threshold before accelerated decay kicks in.
     * @param frames Non-negative frame count (0 disables the feature).
     */
    void set_reset_frames(int frames);

    /**
     * @brief Configure the multiplier applied when reset_frames is exceeded (0-1, lower values reset faster).
     * @param multiplier Clamp between 0 and 1.
     */
    void set_reset_multiplier(T multiplier);

    /**
     * @brief Control the softness of the limiter knee.
     *        0 keeps the legacy hard clamp; higher values progressively smooth the limiting curve.
     * @param softness Non-negative softness value.
     */
    void set_limiter_softness(T softness);

    /**
     * @brief Provide timing context so time-based parameters convert to frame counts.
     * @param sample_rate Current sample rate (Hz).
     * @param hop_size Hop size in samples per frame.
     */
    void set_frame_timing(T sample_rate, T hop_size);

    /**
     * @brief Configure onset hysteresis in frames (minimum consecutive frames above threshold).
     */
    void set_onset_hysteresis_frames(int frames);

    /**
     * @brief Configure onset hysteresis using a time in seconds (converted using frame timing).
     */
    void set_onset_hysteresis_time(T time_s);

    /**
     * @brief Configure the attack onset ramp duration in frames.
     */
    void set_attack_onset_ramp_frames(int frames);

    /**
     * @brief Configure the attack onset ramp duration using a time in seconds.
     */
    void set_attack_onset_ramp_time(T time_s);

    /**
     * @brief Resize the processor for a different FFT size
     * @param num_bins New number of bins
     */
    void resize(size_t num_bins);

    /**
     * @brief Reset the memory spectrum and phase to zero
     */
    void reset_memory();

    // --- Getters ---

    T get_threshold() const { return threshold_; }
    T get_attack() const { return attack_; }
    T get_phase_attack() const { return attack_phase_; }
    T get_decay() const { return decay_; }
    T get_attack_dynamic_rate() const { return attack_dynamic_rate_; }
    T get_attack_onset() const { return attack_onset_; }
    T get_onset_floor() const { return onset_floor_; }
    bool is_limiter_enabled() const { return limiter_enabled_; }
    T get_max_value() const { return max_value_; }
    int get_reset_frames() const { return reset_frames_; }
    T get_reset_multiplier() const { return reset_multiplier_; }
    T get_limiter_softness() const { return limiter_softness_; }
    torch::Device get_device() const { return device_; }
    size_t get_num_bins() const { return num_bins_; }

    // --- Device Management ---
    
    void set_device(torch::Device device);

private:
    // Configuration
    size_t num_bins_;
    torch::Device device_;
    bool verbose_;

    // Parameters
    T threshold_;
    T attack_;
    T attack_phase_;
    T decay_;
    T attack_dynamic_rate_;
    T attack_onset_;
    T onset_floor_;
    bool limiter_enabled_;
    T max_value_;
    T limiter_softness_;
    int reset_frames_;
    T reset_multiplier_;
    bool phase_attack_overridden_;

    // Timing context
    T sample_rate_;
    T hop_size_;
    T frames_per_second_;

    // Onset hysteresis configuration
    int onset_hysteresis_frames_;
    T onset_hysteresis_time_s_;
    bool onset_hysteresis_time_mode_;

    // Onset ramp configuration
    int attack_onset_ramp_frames_;
    T attack_onset_ramp_time_s_;
    bool attack_onset_ramp_time_mode_;

    // Tracking tensors
    torch::Tensor below_threshold_counts_;
    torch::Tensor above_threshold_counts_;
    torch::Tensor onset_ramp_progress_;
    torch::Tensor onset_ramp_active_;

    // Memory tensors
    torch::Tensor memory_magnitude_;
    torch::Tensor memory_phase_;

    // Helper methods
    void log(const std::string& message) const;
    void recompute_onset_hysteresis_frames();
    void recompute_attack_onset_ramp_frames();
    void ensure_device(torch::Tensor& tensor);
};

// ============================================================================
// Implementation
// ============================================================================

template<typename T>
SpectralTrailsProcessor<T>::SpectralTrailsProcessor(
    size_t num_bins,
    torch::Device device,
    bool verbose
)
        : num_bins_(num_bins),
            device_(device),
            verbose_(verbose),
            threshold_(static_cast<T>(0.01)),
            attack_(static_cast<T>(0.8)),
            attack_phase_(static_cast<T>(0.8)),
            decay_(static_cast<T>(0.999)),
            attack_dynamic_rate_(static_cast<T>(0.0)),
            attack_onset_(static_cast<T>(0.95)),
            onset_floor_(static_cast<T>(1e-3)),
            limiter_enabled_(false),
            max_value_(static_cast<T>(1.0)),
            limiter_softness_(static_cast<T>(0.0)),
            reset_frames_(0),
            reset_multiplier_(static_cast<T>(0.5)),
            phase_attack_overridden_(false),
            sample_rate_(static_cast<T>(0.0)),
            hop_size_(static_cast<T>(0.0)),
            frames_per_second_(static_cast<T>(0.0)),
            onset_hysteresis_frames_(1),
            onset_hysteresis_time_s_(static_cast<T>(0.0)),
            onset_hysteresis_time_mode_(false),
            attack_onset_ramp_frames_(0),
            attack_onset_ramp_time_s_(static_cast<T>(0.0)),
            attack_onset_ramp_time_mode_(false),
            below_threshold_counts_(),
            above_threshold_counts_(),
            onset_ramp_progress_(),
            onset_ramp_active_()
{
    if (num_bins_ == 0) {
        throw std::invalid_argument("SpectralTrailsProcessor: num_bins must be > 0");
    }

    // Initialize memory tensors on the specified device
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    memory_magnitude_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
    memory_phase_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
    below_threshold_counts_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
    above_threshold_counts_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
    onset_ramp_progress_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
    onset_ramp_active_ = torch::zeros({static_cast<long>(num_bins_)}, opts);

    log("SpectralTrailsProcessor initialized with " + std::to_string(num_bins_) + " bins on device: " + device_.str());
}

template<typename T>
SpectralTrailsProcessor<T>::~SpectralTrailsProcessor() {
    log("SpectralTrailsProcessor destroyed.");
}

template<typename T>
void SpectralTrailsProcessor<T>::set_device(torch::Device device) {
    if (device_.type() == device.type() && device_.index() == device.index()) {
        return; // No change needed
    }

    device_ = device;
    
    // Move memory tensors to new device
    if (memory_magnitude_.defined()) {
        memory_magnitude_ = memory_magnitude_.to(device_);
    }
    if (memory_phase_.defined()) {
        memory_phase_ = memory_phase_.to(device_);
    }
    if (below_threshold_counts_.defined()) {
        below_threshold_counts_ = below_threshold_counts_.to(device_);
    }
    if (above_threshold_counts_.defined()) {
        above_threshold_counts_ = above_threshold_counts_.to(device_);
    }
    if (onset_ramp_progress_.defined()) {
        onset_ramp_progress_ = onset_ramp_progress_.to(device_);
    }
    if (onset_ramp_active_.defined()) {
        onset_ramp_active_ = onset_ramp_active_.to(device_);
    }

    log("Device changed to: " + device_.str());
}

template<typename T>
void SpectralTrailsProcessor<T>::ensure_device(torch::Tensor& tensor) {
    if (tensor.device().type() != device_.type() || tensor.device().index() != device_.index()) {
        tensor = tensor.to(device_);
    }
}

template<typename T>
std::vector<torch::Tensor> SpectralTrailsProcessor<T>::process_frame(
    const torch::Tensor& magnitude_input,
    const torch::Tensor& phase_input
) {
    // Validate input sizes
    if (magnitude_input.size(0) != static_cast<long>(num_bins_)) {
        throw std::invalid_argument("SpectralTrailsProcessor: magnitude_input size mismatch. Expected " + 
                                  std::to_string(num_bins_) + ", got " + std::to_string(magnitude_input.size(0)));
    }
    if (phase_input.size(0) != static_cast<long>(num_bins_)) {
        throw std::invalid_argument("SpectralTrailsProcessor: phase_input size mismatch. Expected " + 
                                  std::to_string(num_bins_) + ", got " + std::to_string(phase_input.size(0)));
    }

    // Ensure inputs are on the correct device
    torch::Tensor mag_in = magnitude_input.to(device_);
    torch::Tensor phase_in = phase_input.to(device_);

    // 1. Apply universal decay to memory
    memory_magnitude_ = memory_magnitude_ * decay_;

    auto ensure_tracking_tensor = [&](torch::Tensor& tensor) {
        if (!tensor.defined() || tensor.sizes() != memory_magnitude_.sizes()) {
            tensor = torch::zeros_like(memory_magnitude_);
        }
    };

    // Track bins that remain below/above threshold to optionally accelerate decay or trigger hysteresis
    ensure_tracking_tensor(below_threshold_counts_);
    ensure_tracking_tensor(above_threshold_counts_);
    ensure_tracking_tensor(onset_ramp_progress_);
    ensure_tracking_tensor(onset_ramp_active_);

    torch::Tensor below_threshold_mask = mag_in <= threshold_;

    if (reset_frames_ > 0) {
        auto zeros = torch::zeros_like(below_threshold_counts_);
        auto incremented = below_threshold_counts_ + 1.0f;
        below_threshold_counts_ = torch::where(below_threshold_mask, incremented, zeros);

        if (reset_multiplier_ >= static_cast<T>(0.0) && reset_multiplier_ < static_cast<T>(1.0)) {
            auto reset_mask = below_threshold_counts_ >= static_cast<float>(reset_frames_);
            auto reset_scale = torch::full_like(memory_magnitude_, reset_multiplier_);
            memory_magnitude_ = torch::where(reset_mask, memory_magnitude_ * reset_scale, memory_magnitude_);
            memory_phase_ = torch::where(reset_mask, torch::zeros_like(memory_phase_), memory_phase_);
        }
    } else {
        below_threshold_counts_.zero_();
    }

    // 2. Track consecutive frames above threshold for onset hysteresis and ramp handling
    torch::Tensor above_threshold = mag_in > threshold_;
    auto zeros_like_counts = torch::zeros_like(above_threshold_counts_);
    auto incremented_above = above_threshold_counts_ + 1.0f;
    above_threshold_counts_ = torch::where(above_threshold, incremented_above, zeros_like_counts);

    auto zeros_like_active = torch::zeros_like(onset_ramp_active_);
    onset_ramp_active_ = torch::where(above_threshold, onset_ramp_active_, zeros_like_active);

    auto zeros_like_progress = torch::zeros_like(onset_ramp_progress_);
    onset_ramp_progress_ = torch::where(above_threshold, onset_ramp_progress_, zeros_like_progress);

    // 3. Create reinforcement mask (above threshold and increasing)
    torch::Tensor above_memory = mag_in > memory_magnitude_;
    torch::Tensor reinforce_mask = above_threshold & above_memory;

    // 4. Prepare adaptive attack baseline
    torch::Tensor mag_attack_tensor = torch::full_like(mag_in, attack_);
    torch::Tensor adaptive_factor;
    bool use_adaptive_attack = attack_dynamic_rate_ > static_cast<T>(0.0);

    if (use_adaptive_attack) {
        adaptive_factor = 1.0f - torch::exp(-torch::abs(mag_in - memory_magnitude_) * attack_dynamic_rate_);
        mag_attack_tensor = mag_attack_tensor + (1.0f - mag_attack_tensor) * adaptive_factor;
    }

    mag_attack_tensor = torch::clamp(mag_attack_tensor, 0.0f, 1.0f);

    // 5. Onset-specific attack ramp with hysteresis
    T onset_attack_clamped = static_cast<T>(0.0);
    torch::Tensor onset_weight = torch::zeros_like(mag_in);
    if (attack_onset_ >= static_cast<T>(0.0)) {
        onset_attack_clamped = std::clamp(attack_onset_, static_cast<T>(0.0), static_cast<T>(1.0));
        int required_frames = std::max(1, onset_hysteresis_frames_);
        torch::Tensor hysteresis_mask = above_threshold_counts_ >= static_cast<float>(required_frames);
        torch::Tensor onset_candidate_mask = hysteresis_mask & above_threshold & (memory_magnitude_ <= onset_floor_);

        auto zeros_mask_like = torch::zeros_like(onset_ramp_active_);
        auto ones_mask_like = torch::ones_like(onset_ramp_active_);

        onset_ramp_active_ = torch::where(onset_candidate_mask, ones_mask_like, onset_ramp_active_);
        onset_ramp_progress_ = torch::where(onset_candidate_mask, torch::zeros_like(onset_ramp_progress_), onset_ramp_progress_);

        if (attack_onset_ramp_frames_ > 0) {
            auto active_mask = onset_ramp_active_ > 0.5f;
            float ramp_frames = static_cast<float>(std::max(attack_onset_ramp_frames_, 1));

            auto progress = torch::where(active_mask, onset_ramp_progress_, torch::zeros_like(onset_ramp_progress_));
            auto progress_norm = torch::clamp(progress / ramp_frames, 0.0f, 1.0f);

            auto eased = progress_norm * progress_norm * (3.0f - 2.0f * progress_norm);

            onset_weight = torch::where(active_mask, eased, torch::zeros_like(eased));

            auto increment = torch::where(active_mask, progress + 1.0f, onset_ramp_progress_);
            onset_ramp_progress_ = torch::where(active_mask, torch::clamp(increment, 0.0f, ramp_frames), onset_ramp_progress_);

            auto done_mask = onset_ramp_progress_ >= ramp_frames;
            auto deactivate_mask = done_mask | torch::logical_not(above_threshold);
            onset_ramp_active_ = torch::where(deactivate_mask, zeros_mask_like, onset_ramp_active_);
            onset_ramp_progress_ = torch::where(onset_ramp_active_ > 0.5f, onset_ramp_progress_, torch::zeros_like(onset_ramp_progress_));

            auto onset_attack_tensor = torch::full_like(mag_in, onset_attack_clamped);
            mag_attack_tensor = mag_attack_tensor + onset_weight * (onset_attack_tensor - mag_attack_tensor);
        } else {
            auto onset_attack_tensor = torch::full_like(mag_in, onset_attack_clamped);
            mag_attack_tensor = torch::where(onset_candidate_mask, onset_attack_tensor, mag_attack_tensor);
            onset_weight = onset_candidate_mask.to(torch::kFloat32);
        }
    } else {
        onset_ramp_active_.zero_();
        onset_ramp_progress_.zero_();
    }

    torch::Tensor attack_update = mag_attack_tensor * mag_in + (1.0f - mag_attack_tensor) * memory_magnitude_;

    torch::Tensor ramp_factor = torch::ones_like(mag_in);
    torch::Tensor ramp_factor_phase = torch::ones_like(mag_in);
    if (attack_onset_ >= static_cast<T>(0.0) && attack_onset_ramp_frames_ > 0) {
        auto active_mask = onset_ramp_active_ > 0.5f;
        ramp_factor = torch::where(active_mask, onset_weight, ramp_factor);
        ramp_factor_phase = ramp_factor;
    }

    auto blended_update = memory_magnitude_ + ramp_factor * (attack_update - memory_magnitude_);
    memory_magnitude_ = torch::where(reinforce_mask, blended_update, memory_magnitude_);

    // 4. Apply limiter if enabled
    if (limiter_enabled_) {
        if (limiter_softness_ <= static_cast<T>(0.0)) {
            memory_magnitude_ = torch::clamp(memory_magnitude_, 0.0f, max_value_);
        } else {
            auto over = torch::relu(memory_magnitude_ - max_value_);
            auto softened = max_value_ + over / (1.0f + limiter_softness_ * over);
            memory_magnitude_ = torch::where(memory_magnitude_ > max_value_, softened, memory_magnitude_);
            memory_magnitude_ = torch::clamp(memory_magnitude_, 0.0f, std::numeric_limits<float>::max());
        }
    }

    // 5. Smooth phase interpolation for reinforced bins to avoid clicks
    // Instead of abrupt phase replacement, interpolate between current and new phase
    // Normalize phase difference to [-pi, pi] range
    auto phase_diff = phase_in - memory_phase_;
    
    // Wrap phase difference to [-pi, pi] to find shortest rotation
    phase_diff = torch::atan2(torch::sin(phase_diff), torch::cos(phase_diff));
    
    // Apply attack smoothing to phase as well (using same attack coefficient)
    torch::Tensor phase_attack_tensor = torch::full_like(phase_diff, attack_phase_);

    if (use_adaptive_attack) {
        phase_attack_tensor = phase_attack_tensor + (1.0f - phase_attack_tensor) * adaptive_factor;
    }

    phase_attack_tensor = torch::clamp(phase_attack_tensor, 0.0f, 1.0f);

    if (attack_onset_ >= static_cast<T>(0.0)) {
        auto onset_phase_tensor = torch::full_like(phase_diff, onset_attack_clamped);
        phase_attack_tensor = phase_attack_tensor + onset_weight * (onset_phase_tensor - phase_attack_tensor);
    }

    auto phase_update = memory_phase_ + ramp_factor_phase * phase_attack_tensor * phase_diff;
    
    // Update phase only where bins are being reinforced
    memory_phase_ = torch::where(reinforce_mask, phase_update, memory_phase_);

    // 6. Force DC and Nyquist phase to zero
    if (num_bins_ > 0) {
        memory_phase_[0] = 0.0f; // DC component
        if (num_bins_ > 1) {
            memory_phase_[static_cast<long>(num_bins_) - 1] = 0.0f; // Nyquist component
        }
    }

    // Return processed magnitude and phase
    return {memory_magnitude_.clone(), memory_phase_.clone()};
}

template<typename T>
void SpectralTrailsProcessor<T>::set_threshold(T threshold) {
    threshold_ = std::max(static_cast<T>(0.0), threshold);
    log("Threshold set to: " + std::to_string(threshold_));
}

template<typename T>
void SpectralTrailsProcessor<T>::set_attack(T attack) {
    attack_ = std::clamp(attack, static_cast<T>(0.0), static_cast<T>(1.0));
    if (!phase_attack_overridden_) {
        attack_phase_ = attack_;
    }
    log("Attack set to: " + std::to_string(attack_));
}

template<typename T>
void SpectralTrailsProcessor<T>::set_phase_attack(T attack_phase) {
    attack_phase_ = std::clamp(attack_phase, static_cast<T>(0.0), static_cast<T>(1.0));
    phase_attack_overridden_ = true;
    log("Phase attack set to: " + std::to_string(attack_phase_));
}

template<typename T>
void SpectralTrailsProcessor<T>::set_decay(T decay) {
    decay_ = std::clamp(decay, static_cast<T>(0.0), static_cast<T>(1.0));
    log("Decay set to: " + std::to_string(decay_));
}

template<typename T>
void SpectralTrailsProcessor<T>::set_decay_time_s(T time_s, T sample_rate, T hop_size) {
    if (time_s <= static_cast<T>(0.0) || sample_rate <= static_cast<T>(0.0) || hop_size <= static_cast<T>(0.0)) {
        log("set_decay_time_s ignored: invalid parameters");
        return;
    }

    T frames_per_second = sample_rate / hop_size;
    if (frames_per_second <= static_cast<T>(0.0)) {
        log("set_decay_time_s ignored: frames_per_second <= 0");
        return;
    }

    T total_frames = time_s * frames_per_second;
    if (total_frames <= static_cast<T>(0.0)) {
        log("set_decay_time_s ignored: total_frames <= 0");
        return;
    }

    decay_ = std::pow(static_cast<T>(0.5), static_cast<T>(1.0) / total_frames);
    log("Decay set from decay6db: " + std::to_string(decay_) + " (time_s=" + std::to_string(time_s) + ", frames_per_second=" + std::to_string(frames_per_second) + ")");
}

template<typename T>
void SpectralTrailsProcessor<T>::set_limiter_enabled(bool enabled) {
    limiter_enabled_ = enabled;
    log(std::string("Limiter ") + (enabled ? "enabled" : "disabled"));
}

template<typename T>
void SpectralTrailsProcessor<T>::set_max_value(T max_value) {
    max_value_ = std::max(static_cast<T>(0.0), max_value);
    log("Max value set to: " + std::to_string(max_value_));
}

template<typename T>
void SpectralTrailsProcessor<T>::set_attack_dynamic_rate(T rate) {
    attack_dynamic_rate_ = std::max(static_cast<T>(0.0), rate);
    log("Attack dynamic rate set to: " + std::to_string(attack_dynamic_rate_));
}

template<typename T>
void SpectralTrailsProcessor<T>::set_attack_onset(T attack_onset) {
    if (attack_onset < static_cast<T>(0.0)) {
        attack_onset_ = static_cast<T>(-1.0);
        log("Attack onset disabled");
        return;
    }

    attack_onset_ = std::clamp(attack_onset, static_cast<T>(0.0), static_cast<T>(1.0));
    log("Attack onset set to: " + std::to_string(attack_onset_));
}

template<typename T>
void SpectralTrailsProcessor<T>::set_onset_floor(T floor) {
    onset_floor_ = std::max(static_cast<T>(0.0), floor);
    log("Onset floor set to: " + std::to_string(onset_floor_));
}

template<typename T>
void SpectralTrailsProcessor<T>::set_reset_frames(int frames) {
    reset_frames_ = std::max(0, frames);
    if (reset_frames_ == 0 && below_threshold_counts_.defined()) {
        below_threshold_counts_.zero_();
    }
    log("Reset frames set to: " + std::to_string(reset_frames_));
}

template<typename T>
void SpectralTrailsProcessor<T>::set_reset_multiplier(T multiplier) {
    reset_multiplier_ = std::clamp(multiplier, static_cast<T>(0.0), static_cast<T>(1.0));
    log("Reset multiplier set to: " + std::to_string(reset_multiplier_));
}

template<typename T>
void SpectralTrailsProcessor<T>::set_limiter_softness(T softness) {
    limiter_softness_ = std::max(static_cast<T>(0.0), softness);
    log("Limiter softness set to: " + std::to_string(limiter_softness_));
}

template<typename T>
void SpectralTrailsProcessor<T>::set_frame_timing(T sample_rate, T hop_size) {
    if (sample_rate <= static_cast<T>(0.0) || hop_size <= static_cast<T>(0.0)) {
        sample_rate_ = static_cast<T>(0.0);
        hop_size_ = static_cast<T>(0.0);
        frames_per_second_ = static_cast<T>(0.0);
        log("Frame timing disabled (invalid sample rate or hop size)");
        return;
    }

    sample_rate_ = sample_rate;
    hop_size_ = hop_size;
    frames_per_second_ = sample_rate_ / hop_size_;

    if (onset_hysteresis_time_mode_) {
        recompute_onset_hysteresis_frames();
    }
    if (attack_onset_ramp_time_mode_) {
        recompute_attack_onset_ramp_frames();
    }

    log("Frame timing updated: sample_rate=" + std::to_string(sample_rate_) +
        ", hop_size=" + std::to_string(hop_size_) +
        ", frames_per_second=" + std::to_string(frames_per_second_));
}

template<typename T>
void SpectralTrailsProcessor<T>::set_onset_hysteresis_frames(int frames) {
    onset_hysteresis_frames_ = std::max(1, frames);
    onset_hysteresis_time_s_ = static_cast<T>(0.0);
    onset_hysteresis_time_mode_ = false;
    log("Onset hysteresis (frames) set to: " + std::to_string(onset_hysteresis_frames_));
}

template<typename T>
void SpectralTrailsProcessor<T>::set_onset_hysteresis_time(T time_s) {
    time_s = std::max(static_cast<T>(0.0), time_s);
    onset_hysteresis_time_s_ = time_s;
    onset_hysteresis_time_mode_ = time_s > static_cast<T>(0.0);
    if (onset_hysteresis_time_mode_) {
        recompute_onset_hysteresis_frames();
    } else {
        onset_hysteresis_frames_ = 1;
    }
    log("Onset hysteresis (time) set to: " + std::to_string(onset_hysteresis_time_s_) + " s");
}

template<typename T>
void SpectralTrailsProcessor<T>::set_attack_onset_ramp_frames(int frames) {
    attack_onset_ramp_frames_ = std::max(0, frames);
    attack_onset_ramp_time_s_ = static_cast<T>(0.0);
    attack_onset_ramp_time_mode_ = false;
    log("Attack onset ramp (frames) set to: " + std::to_string(attack_onset_ramp_frames_));
}

template<typename T>
void SpectralTrailsProcessor<T>::set_attack_onset_ramp_time(T time_s) {
    time_s = std::max(static_cast<T>(0.0), time_s);
    attack_onset_ramp_time_s_ = time_s;
    attack_onset_ramp_time_mode_ = time_s > static_cast<T>(0.0);
    if (attack_onset_ramp_time_mode_) {
        recompute_attack_onset_ramp_frames();
    } else {
        attack_onset_ramp_frames_ = 0;
    }
    log("Attack onset ramp (time) set to: " + std::to_string(attack_onset_ramp_time_s_) + " s");
}

template<typename T>
void SpectralTrailsProcessor<T>::recompute_onset_hysteresis_frames() {
    if (!onset_hysteresis_time_mode_ || onset_hysteresis_time_s_ <= static_cast<T>(0.0)) {
        onset_hysteresis_frames_ = std::max(1, onset_hysteresis_frames_);
        return;
    }

    if (frames_per_second_ <= static_cast<T>(0.0)) {
        // Defer computation until timing is available
        return;
    }

    int frames = static_cast<int>(std::round(onset_hysteresis_time_s_ * frames_per_second_));
    onset_hysteresis_frames_ = std::max(1, frames);
}

template<typename T>
void SpectralTrailsProcessor<T>::recompute_attack_onset_ramp_frames() {
    if (!attack_onset_ramp_time_mode_ || attack_onset_ramp_time_s_ <= static_cast<T>(0.0)) {
        attack_onset_ramp_frames_ = std::max(0, attack_onset_ramp_frames_);
        return;
    }

    if (frames_per_second_ <= static_cast<T>(0.0)) {
        // Defer computation until timing is available
        return;
    }

    int frames = static_cast<int>(std::round(attack_onset_ramp_time_s_ * frames_per_second_));
    attack_onset_ramp_frames_ = std::max(1, frames);
}

template<typename T>
void SpectralTrailsProcessor<T>::resize(size_t num_bins) {
    if (num_bins == 0) {
        throw std::invalid_argument("SpectralTrailsProcessor: num_bins must be > 0");
    }

    num_bins_ = num_bins;
    
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    memory_magnitude_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
    memory_phase_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
    below_threshold_counts_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
    above_threshold_counts_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
    onset_ramp_progress_ = torch::zeros({static_cast<long>(num_bins_)}, opts);
    onset_ramp_active_ = torch::zeros({static_cast<long>(num_bins_)}, opts);

    log("Resized to " + std::to_string(num_bins_) + " bins");
}

template<typename T>
void SpectralTrailsProcessor<T>::reset_memory() {
    memory_magnitude_.zero_();
    memory_phase_.zero_();
    if (below_threshold_counts_.defined()) {
        below_threshold_counts_.zero_();
    }
    if (above_threshold_counts_.defined()) {
        above_threshold_counts_.zero_();
    }
    if (onset_ramp_progress_.defined()) {
        onset_ramp_progress_.zero_();
    }
    if (onset_ramp_active_.defined()) {
        onset_ramp_active_.zero_();
    }
    log("Memory reset to zero");
}

template<typename T>
void SpectralTrailsProcessor<T>::log(const std::string& message) const {
    if (verbose_) {
        std::cout << "[SpectralTrailsProcessor]: " << message << std::endl;
    }
}

        } // namespace ap_spectrails
    } // namespace core
} // namespace contorchionist

#endif // CORE_AP_SPECTRAILS_H

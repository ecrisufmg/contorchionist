#ifndef CORE_AP_SPECLEAKRFFT_HPP
#define CORE_AP_SPECLEAKRFFT_HPP

#include <vector>
#include <cmath>
#include <algorithm> // For std::max

template <typename T>
class SpectralLeakRFFTProcessor {
public:
    /**
     * @brief Constructor for the SpectralLeakRFFTProcessor.
     * @param fft_size The size of the FFT. The internal memory spectrum will be sized N/2 + 1.
     */
    explicit SpectralLeakRFFTProcessor(size_t fft_size)
        : m_threshold(static_cast<T>(0.01)),
          m_attack_alpha(static_cast<T>(0.8)),
          m_decay_factor(static_cast<T>(0.999)),
          m_decay_time_s(static_cast<T>(4.0)),
          m_sample_rate(static_cast<T>(44100.0)),
          m_hop_size(static_cast<T>(512.0)) {
        if (fft_size > 0) {
            size_t num_bins = fft_size / 2 + 1;
            m_memory_spectrum.resize(num_bins, static_cast<T>(0.0));
            m_memory_phase.resize(num_bins, static_cast<T>(0.0));
        }
        calculate_decay_factor();
    }

    /**
     * @brief Processes a single frame of spectral magnitude and phase data.
     * @param magnitude_frame A vector of spectral magnitudes.
     * @param phase_frame A vector of spectral phases.
     */
    void process_frame(const std::vector<T>& magnitude_frame, const std::vector<T>& phase_frame) {
        if (magnitude_frame.size() != m_memory_spectrum.size() || phase_frame.size() != m_memory_phase.size()) {
            return;
        }

        // 1. Universal Decay
        for (T& bin : m_memory_spectrum) {
            bin *= m_decay_factor;
        }

        // 2. Selective Reinforcement with phase locking
        for (size_t i = 0; i < magnitude_frame.size(); ++i) {
            if (magnitude_frame[i] > m_memory_spectrum[i] && magnitude_frame[i] > m_threshold) {
                // Apply EMA for the attack on magnitude
                m_memory_spectrum[i] = (m_attack_alpha * magnitude_frame[i]) +
                                       ((static_cast<T>(1.0) - m_attack_alpha) * m_memory_spectrum[i]);
                // Lock the phase
                m_memory_phase[i] = phase_frame[i];
            }
        }

        // 3. Force DC and Nyquist phase to zero
        if (m_memory_phase.size() > 0) {
            m_memory_phase[0] = static_cast<T>(0.0); // DC component
            m_memory_phase.back() = static_cast<T>(0.0); // Nyquist component
        }
    }

    /**
     * @brief Retrieves the current state of the magnitude memory spectrum.
     * @return A constant reference to the internal magnitude spectrum vector.
     */
    const std::vector<T>& get_processed_frame() const {
        return m_memory_spectrum;
    }

    /**
     * @brief Retrieves the current state of the phase memory spectrum.
     * @return A constant reference to the internal phase spectrum vector.
     */
    const std::vector<T>& get_processed_phase_frame() const {
        return m_memory_phase;
    }

    // --- Configuration Setters ---

    /**
     * @brief Sets the magnitude threshold for reinforcement.
     * @param threshold The linear amplitude threshold.
     */
    void set_threshold(T threshold) {
        m_threshold = std::max(static_cast<T>(0.0), threshold);
    }

    /**
     * @brief Sets the attack alpha for the EMA filter.
     * @param alpha A value between 0.0 and 1.0. Higher values mean a faster attack.
     */
    void set_attack_alpha(T alpha) {
        if (alpha >= static_cast<T>(0.0) && alpha <= static_cast<T>(1.0)) {
            m_attack_alpha = alpha;
        }
    }

    /**
     * @brief Sets the decay time and recalculates the per-frame decay factor.
     * @param time_s The time in seconds for a bin's amplitude to decay by -3dB (to 50%).
     * @param sample_rate The audio sample rate.
     * @param hop_size The FFT hop size.
     */
    void set_decay_time_s(T time_s, T sample_rate, T hop_size) {
        m_decay_time_s = std::max(static_cast<T>(0.001), time_s); // Avoid zero or negative time
        if (sample_rate > 0) m_sample_rate = sample_rate;
        if (hop_size > 0) m_hop_size = hop_size;
        calculate_decay_factor();
    }

    /**
     * @brief Resizes the internal memory spectrum.
     * This is useful if the FFT size changes dynamically.
     * @param new_fft_size The new FFT size.
     */
    void resize(size_t new_fft_size) {
        if (new_fft_size > 0) {
            size_t num_bins = new_fft_size / 2 + 1;
            m_memory_spectrum.assign(num_bins, static_cast<T>(0.0));
            m_memory_phase.assign(num_bins, static_cast<T>(0.0));
        }
    }

protected:
    // --- Member Variables ---
    std::vector<T> m_memory_spectrum;
    std::vector<T> m_memory_phase;
private:
    T m_threshold;
    T m_attack_alpha;
    T m_decay_factor;
    T m_decay_time_s;
    T m_sample_rate;
    T m_hop_size;

    /**
     * @brief Calculates the per-frame decay factor based on decay time, sample rate, and hop size.
     * The factor is calculated so that a value decays to 50% (-3dB) over the specified time.
     */
    void calculate_decay_factor() {
        if (m_decay_time_s <= 0 || m_sample_rate <= 0 || m_hop_size <= 0) {
            m_decay_factor = static_cast<T>(1.0); // No decay
            return;
        }

        // Calculate the number of FFT frames within the decay time
        T num_frames = (m_decay_time_s * m_sample_rate) / m_hop_size;

        if (num_frames <= 0) {
            m_decay_factor = static_cast<T>(1.0); // No decay
            return;
        }

        // Calculate the decay factor for -3dB over num_frames
        // y = x * factor^n  =>  0.5 = 1 * factor^num_frames  =>  factor = 0.5^(1/num_frames)
        m_decay_factor = static_cast<T>(std::pow(0.5, 1.0 / num_frames));
    }
};

#endif // CORE_AP_SPECLEAKRFFT_HPP

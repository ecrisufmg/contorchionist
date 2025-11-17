#ifndef CORE_AP_SPECTRALTRACKER_HPP
#define CORE_AP_SPECTRALTRACKER_HPP

#include "core_ap_specleakrfft.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

template <typename T>
class SpectralTrackerProcessor : public SpectralLeakRFFTProcessor<T> {
public:
    struct Peak {
        size_t index;
        T amplitude;
        T frequency;
        int voice_id = -1;
    };

    explicit SpectralTrackerProcessor(size_t fft_size, size_t num_voices = 8)
        : SpectralLeakRFFTProcessor<T>(fft_size), m_num_voices(num_voices) {
        m_voices.resize(num_voices);
    }

    void find_and_assign_peaks(T sample_rate) {
        const std::vector<T>& spectrum = this->get_processed_frame();
        std::vector<Peak> current_peaks;

        for (size_t i = 1; i < spectrum.size() - 1; ++i) {
            if (spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1]) {
                T freq = static_cast<T>(i) * static_cast<T>(sample_rate) / static_cast<T>(this->m_memory_spectrum.size() * 2);
                current_peaks.push_back({i, spectrum[i], freq});
            }
        }

        std::sort(current_peaks.begin(), current_peaks.end(), [](const Peak& a, const Peak& b) {
            return a.amplitude > b.amplitude;
        });

        if (current_peaks.size() > m_num_voices) {
            current_peaks.resize(m_num_voices);
        }

        assign_voices(current_peaks);
    }

    const std::vector<Peak>& get_voices() const {
        return m_voices;
    }

private:
    void assign_voices(const std::vector<Peak>& peaks) {
        std::vector<bool> peak_assigned(peaks.size(), false);
        std::vector<bool> voice_assigned(m_num_voices, false);

        for (size_t i = 0; i < m_voices.size(); ++i) {
            if (m_voices[i].amplitude > 0) {
                int best_peak_idx = -1;
                T min_freq_dist = 1e18;

                for (size_t j = 0; j < peaks.size(); ++j) {
                    if (!peak_assigned[j]) {
                        T freq_dist = std::abs(m_voices[i].frequency - peaks[j].frequency);
                        if (freq_dist < min_freq_dist) {
                            min_freq_dist = freq_dist;
                            best_peak_idx = j;
                        }
                    }
                }

                if (best_peak_idx != -1) {
                    m_voices[i] = peaks[best_peak_idx];
                    m_voices[i].voice_id = i;
                    peak_assigned[best_peak_idx] = true;
                    voice_assigned[i] = true;
                }
            }
        }

        for (size_t i = 0; i < peaks.size(); ++i) {
            if (!peak_assigned[i]) {
                for (size_t j = 0; j < m_voices.size(); ++j) {
                    if (!voice_assigned[j]) {
                        m_voices[j] = peaks[i];
                        m_voices[j].voice_id = j;
                        voice_assigned[j] = true;
                        break;
                    }
                }
            }
        }
    }

    size_t m_num_voices;
    std::vector<Peak> m_voices;
};

#endif // CORE_AP_SPECTRALTRACKER_HPP
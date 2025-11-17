#include "core_util_conversions.h"
#include "contorchionist_core/contorchionist_core_export.h"
#include <cmath>
#include <limits>

namespace contorchionist {
namespace core {
namespace util_conversions {

// Template implementations

template <typename T>
T hzToMel(T frequency) {
    return static_cast<T>(2595.0) * std::log10(static_cast<T>(1.0) + frequency / static_cast<T>(700.0));
}

template <typename T>
T melToHz(T mel) {
    return static_cast<T>(700.0) * (std::pow(static_cast<T>(10.0), mel / static_cast<T>(2595.0)) - static_cast<T>(1.0));
}

template <typename T>
T complexToMagnitude(std::complex<T> c) {
    return std::abs(c);
}

template <typename T>
T complexToPhase(std::complex<T> c) {
    return std::arg(c);
}

template <typename T>
T radiansToDegrees(T radians) {
    return radians * static_cast<T>(180.0) / static_cast<T>(M_PI);
}

template <typename T>
std::pair<T, T> rectangularToPolar(T real, T imag) {
    T magnitude = std::sqrt(real * real + imag * imag);
    T phase = std::atan2(imag, real);
    return std::make_pair(magnitude, phase);
}

template <typename T>
std::pair<T, T> polarToRectangular(T magnitude, T phase_radians) {
    T real = magnitude * std::cos(phase_radians);
    T imag = magnitude * std::sin(phase_radians);
    return std::make_pair(real, imag);
}

template <typename T>
T magnitudeToPower(T magnitude) {
    return magnitude * magnitude;
}

template <typename T>
T powerToDb(T power, T reference_power, bool avoid_nan) {
    if (reference_power <= 0) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (power <= 0) {
        if (avoid_nan) {
            power = std::numeric_limits<T>::epsilon();
        } else {
            return std::numeric_limits<T>::quiet_NaN();
        }
    }
    return 10 * std::log10(power / reference_power);
}

template <typename T>
T dbToPower(T power_db, T reference_power) {
    if (reference_power <= 0) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    return reference_power * std::pow(10, power_db / 10);
}

template <typename T>
T powerToMagnitude(T power) {
    if (power < 0) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    return std::sqrt(power);
}

template <typename T>
T hzToMelHTK(T frequency) {
    return hzToMel(frequency);
}

template <typename T>
T melToHzHTK(T mel) {
    return melToHz(mel);
}

template <typename T>
T hzToMelSlaney(T frequency) {
    static const T f_min = static_cast<T>(0.0);
    static const T f_sp = static_cast<T>(200.0) / static_cast<T>(3.0);
    static const T min_log_hz = static_cast<T>(1000.0);
    static const T min_log_mel = (min_log_hz - f_min) / f_sp;
    static const T logstep = std::log(static_cast<T>(6.4)) / static_cast<T>(27.0);
    
    if (frequency >= min_log_hz) {
        return min_log_mel + std::log(frequency / min_log_hz) / logstep;
    } else {
        return (frequency - f_min) / f_sp;
    }
}

template <typename T>
T melToHzSlaney(T mel) {
    static const T f_min = static_cast<T>(0.0);
    static const T f_sp = static_cast<T>(200.0) / static_cast<T>(3.0);
    static const T min_log_hz = static_cast<T>(1000.0);
    static const T min_log_mel = (min_log_hz - f_min) / f_sp;
    static const T logstep = std::log(static_cast<T>(6.4)) / static_cast<T>(27.0);
    
    if (mel >= min_log_mel) {
        return min_log_hz * std::exp(logstep * (mel - min_log_mel));
    } else {
        return f_min + f_sp * mel;
    }
}

template <typename T>
T hzToMelCalc2(T frequency) {
    return static_cast<T>(1127.01048) * std::log(frequency / static_cast<T>(700.0) + static_cast<T>(1.0));
}

template <typename T>
T melToHzCalc2(T mel) {
    return static_cast<T>(700.0) * (std::exp(mel / static_cast<T>(1127.01048)) - static_cast<T>(1.0));
}

// Explicit template instantiations
template CONTORCHIONIST_CORE_EXPORT float hzToMel<float>(float frequency);
template CONTORCHIONIST_CORE_EXPORT double hzToMel<double>(double frequency);

template CONTORCHIONIST_CORE_EXPORT float melToHz<float>(float mel);
template CONTORCHIONIST_CORE_EXPORT double melToHz<double>(double mel);

template CONTORCHIONIST_CORE_EXPORT float complexToMagnitude<float>(std::complex<float> c);
template CONTORCHIONIST_CORE_EXPORT double complexToMagnitude<double>(std::complex<double> c);

template CONTORCHIONIST_CORE_EXPORT float complexToPhase<float>(std::complex<float> c);
template CONTORCHIONIST_CORE_EXPORT double complexToPhase<double>(std::complex<double> c);

template CONTORCHIONIST_CORE_EXPORT float radiansToDegrees<float>(float radians);
template CONTORCHIONIST_CORE_EXPORT double radiansToDegrees<double>(double radians);

template CONTORCHIONIST_CORE_EXPORT std::pair<float, float> rectangularToPolar<float>(float real, float imag);
template CONTORCHIONIST_CORE_EXPORT std::pair<double, double> rectangularToPolar<double>(double real, double imag);

template CONTORCHIONIST_CORE_EXPORT std::pair<float, float> polarToRectangular<float>(float magnitude, float phase_radians);
template CONTORCHIONIST_CORE_EXPORT std::pair<double, double> polarToRectangular<double>(double magnitude, double phase_radians);

template CONTORCHIONIST_CORE_EXPORT float magnitudeToPower<float>(float magnitude);
template CONTORCHIONIST_CORE_EXPORT double magnitudeToPower<double>(double magnitude);

template CONTORCHIONIST_CORE_EXPORT float powerToDb<float>(float power, float reference_power, bool avoid_nan);
template CONTORCHIONIST_CORE_EXPORT double powerToDb<double>(double power, double reference_power, bool avoid_nan);

template CONTORCHIONIST_CORE_EXPORT float dbToPower<float>(float power_db, float reference_power);
template CONTORCHIONIST_CORE_EXPORT double dbToPower<double>(double power_db, double reference_power);

template CONTORCHIONIST_CORE_EXPORT float powerToMagnitude<float>(float power);
template CONTORCHIONIST_CORE_EXPORT double powerToMagnitude<double>(double power);

template CONTORCHIONIST_CORE_EXPORT float hzToMelHTK<float>(float frequency);
template CONTORCHIONIST_CORE_EXPORT double hzToMelHTK<double>(double frequency);

template CONTORCHIONIST_CORE_EXPORT float melToHzHTK<float>(float mel);
template CONTORCHIONIST_CORE_EXPORT double melToHzHTK<double>(double mel);

template CONTORCHIONIST_CORE_EXPORT float hzToMelSlaney<float>(float frequency);
template CONTORCHIONIST_CORE_EXPORT double hzToMelSlaney<double>(double frequency);

template CONTORCHIONIST_CORE_EXPORT float melToHzSlaney<float>(float mel);
template CONTORCHIONIST_CORE_EXPORT double melToHzSlaney<double>(double mel);

template CONTORCHIONIST_CORE_EXPORT float hzToMelCalc2<float>(float frequency);
template CONTORCHIONIST_CORE_EXPORT double hzToMelCalc2<double>(double frequency);

template CONTORCHIONIST_CORE_EXPORT float melToHzCalc2<float>(float mel);
template CONTORCHIONIST_CORE_EXPORT double melToHzCalc2<double>(double mel);

// Non-template function implementations

SpectrumDataFormat string_to_spectrum_data_format(const std::string& format_str) {
    std::string lower_str = format_str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    if (lower_str == "complex") {
        return SpectrumDataFormat::COMPLEX;
    } else if (lower_str == "magphase" || lower_str == "mag") {
        return SpectrumDataFormat::MAGPHASE;
    } else if (lower_str == "powerphase" || lower_str == "power" || lower_str == "pow") {
        return SpectrumDataFormat::POWERPHASE;
    } else if (lower_str == "dbphase" || lower_str == "db") {
        return SpectrumDataFormat::DBPHASE;
    } else {
        throw std::invalid_argument("Unknown SpectrumDataFormat string: " + format_str);
    }
}

std::string spectrum_data_format_to_string(SpectrumDataFormat format) {
    switch (format) {
        case SpectrumDataFormat::COMPLEX:
            return "complex";
        case SpectrumDataFormat::MAGPHASE:
            return "magphase";
        case SpectrumDataFormat::POWERPHASE:
            return "powerphase";
        case SpectrumDataFormat::DBPHASE:
            return "dbphase";
        default:
            // Should not happen with a valid enum, but good practice
            return "unknown"; // Or throw std::invalid_argument
    }
}


std::string mel_formula_type_to_string(MelFormulaType type) {
    switch (type) {
        case MelFormulaType::SLANEY: return "slaney";
        case MelFormulaType::HTK:    return "htk";
        case MelFormulaType::CALC2:  return "calc2";
        default: return "unknown"; // Or throw
    }
}

MelFormulaType string_to_mel_formula_type(const std::string& type_str) {
    std::string lower_str = type_str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    if (lower_str == "slaney" || lower_str == "librosa") {
        return MelFormulaType::SLANEY;
    } else if (lower_str == "htk") {
        return MelFormulaType::HTK;
    } else if (lower_str == "calc2" || lower_str == "flucoma" || lower_str == "oshaughnessy" || lower_str == "osh") {
        return MelFormulaType::CALC2;
    }
    throw std::invalid_argument("Unknown MelFormulaType string: " + type_str + ". Use 'slaney', 'htk', or 'calc2'.");
}


// Torch tensor implementations
torch::Tensor fftFrequencies(float sr, int n_fft) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    return torch::arange(0, n_fft / 2 + 1, options) * sr / n_fft;
}

torch::Tensor melFrequencies(int n_mels, float fmin, float fmax, MelFormulaType formula) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    
    // Convert to mel scale
    float mel_min, mel_max;
    switch (formula) {
        case MelFormulaType::HTK:
            mel_min = hzToMelHTK(fmin);
            mel_max = hzToMelHTK(fmax);
            break;
        case MelFormulaType::CALC2:
            mel_min = hzToMelCalc2(fmin);
            mel_max = hzToMelCalc2(fmax);
            break;
        case MelFormulaType::SLANEY:
        default: // Default to Slaney if unknown or not specified
            mel_min = hzToMelSlaney(fmin);
            mel_max = hzToMelSlaney(fmax);
            break;
    }
    
    // Create linearly spaced mel frequencies
    torch::Tensor mel_f = torch::linspace(mel_min, mel_max, n_mels + 2, options);
    
    // Convert back to Hz
    switch (formula) {
        case MelFormulaType::HTK:
            for (int i = 0; i < mel_f.numel(); ++i) {
                mel_f[i] = melToHzHTK(mel_f[i].item<float>());
            }
            break;
        case MelFormulaType::CALC2:
            for (int i = 0; i < mel_f.numel(); ++i) {
                mel_f[i] = melToHzCalc2(mel_f[i].item<float>());
            }
            break;
        case MelFormulaType::SLANEY:
        default:
            for (int i = 0; i < mel_f.numel(); ++i) {
                mel_f[i] = melToHzSlaney(mel_f[i].item<float>());
            }
            break;
    }
    
    return mel_f;
}

torch::Tensor melFilterBank(int n_mels, int fft_size, float sample_rate, float fmin, float fmax, MelFormulaType formula) {
    // Create mel frequency points
    torch::Tensor mel_f = melFrequencies(n_mels, fmin, fmax, formula);
    
    // Create FFT frequency bins
    torch::Tensor fft_f = fftFrequencies(sample_rate, fft_size);
    
    int n_freqs = fft_size / 2 + 1;
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor filterbank = torch::zeros({n_mels, n_freqs}, options);
    
    // Build triangular filters
    for (int m = 0; m < n_mels; ++m) {
        float left_hz = mel_f[m].item<float>();
        float center_hz = mel_f[m + 1].item<float>();
        float right_hz = mel_f[m + 2].item<float>();
        
        for (int k = 0; k < n_freqs; ++k) {
            float freq_hz = fft_f[k].item<float>();
            float weight = 0.0f;
            
            if (freq_hz >= left_hz && freq_hz <= center_hz && center_hz != left_hz) {
                weight = (freq_hz - left_hz) / (center_hz - left_hz);
            } else if (freq_hz > center_hz && freq_hz <= right_hz && right_hz != center_hz) {
                weight = (right_hz - freq_hz) / (right_hz - center_hz);
            }
            
            if (weight > 0.0f) {
                filterbank[m][k] = weight;
            }
        }
    }
    
    return filterbank;
}

torch::Tensor applyMelFilterBank(torch::Tensor spectrum, torch::Tensor filter_bank) {
    // Apply mel filter bank to spectrum
    // spectrum: [batch?, freq_bins, time_frames] or [freq_bins, time_frames]
    // filter_bank: [n_mels, freq_bins]
    // Result: [batch?, n_mels, time_frames] or [n_mels, time_frames]
    
    if (spectrum.dim() == 2) {
        // spectrum: [freq_bins, time_frames]
        return torch::matmul(filter_bank, spectrum);
    } else if (spectrum.dim() == 3) {
        // spectrum: [batch, freq_bins, time_frames]
        return torch::matmul(filter_bank.unsqueeze(0), spectrum);
    } else {
        throw std::invalid_argument("Spectrum must be 2D or 3D tensor");
    }
}

torch::Tensor computeMFCC(torch::Tensor signal, int sample_rate, int n_mfcc, float fmin, float fmax, bool htk, bool calc2) {
    // Basic MFCC computation - this is a placeholder implementation
    // In a real implementation, you would:
    // 1. Compute STFT of the signal
    // 2. Apply mel filter bank
    // 3. Take log
    // 4. Apply DCT
    
    throw std::runtime_error("computeMFCC: Full implementation not yet available. Use audio_features.h functions instead.");
}

torch::Tensor computeIMFCC(torch::Tensor mfccs, int sample_rate, int n_mels, float fmin, float fmax, MelFormulaType formula) {
    // Basic IMFCC computation - this is a placeholder implementation
    throw std::runtime_error("computeIMFCC: Full implementation not yet available.");
}

} // namespace unit_conversions
} // namespace core
} // namespace contorchionist

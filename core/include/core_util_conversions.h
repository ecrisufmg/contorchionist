#ifndef CORE_UTIL_CONVERSIONS_H
#define CORE_UTIL_CONVERSIONS_H

#include "contorchionist_core/contorchionist_core_export.h" // Added for export macro
#include <cmath> // For std::log10, std::pow, std::sqrt, std::atan2, std::cos, std::sin
#include <complex> // For std::abs, std::arg
#include <utility> /**

 * @brief Converts a frequency from the Mel scale to Hertz using calc2-style conversion.
 * This follows the calc2 library conversion formula.
 * @tparam T The floating-point type (float or double).
 * @param mel The frequency in Mels.
 * @return The frequency in Hertz.
 */
template <typename T>
T melToHzCalc2(T mel);

#include <limits> // For std::numeric_limits
#include <string>
#include <stdexcept> // Required for std::invalid_argument
#include <torch/torch.h> // Add include for torch

namespace contorchionist {
namespace core {
namespace util_conversions {

/**
 * @brief Converts a frequency from Hertz to the Mel scale.
 * @tparam T The floating-point type (float or double).
 * @param frequency The frequency in Hertz.
 * @return The frequency in Mels.
 */
template <typename T>
T hzToMel(T frequency);

/**
 * @brief Converts a frequency from the Mel scale to Hertz.
 * @tparam T The floating-point type (float or double).
 * @param mel The frequency in Mels.
 * @return The frequency in Hertz.
 */
template <typename T>
T melToHz(T mel);

/**
 * @brief Calculates the magnitude of a complex number.
 * @tparam T The floating-point type of the complex number's components (float or double).
 * @param c The complex number.
 * @return The magnitude of the complex number.
 */
template <typename T>
T complexToMagnitude(std::complex<T> c);

/**
 * @brief Calculates the phase of a complex number in radians.
 * @tparam T The floating-point type of the complex number's components (float or double).
 * @param c The complex number.
 * @return The phase of the complex number in radians.
 */
template <typename T>
T complexToPhase(std::complex<T> c);

/**
 * @brief Converts an angle from radians to degrees.
 * @tparam T The floating-point type (float or double).
 * @param radians The angle in radians.
 * @return The angle in degrees.
 */
template <typename T>
T radiansToDegrees(T radians);

/**
 * @brief Converts rectangular coordinates (real, imaginary) to polar coordinates (magnitude, phase).
 * @tparam T The floating-point type (float or double).
 * @param real The real part of the complex number.
 * @param imag The imaginary part of the complex number.
 * @return A std::pair containing the magnitude (first) and phase in radians (second).
 */
template <typename T>
std::pair<T, T> rectangularToPolar(T real, T imag);

/**
 * @brief Converts polar coordinates (magnitude, phase) to rectangular coordinates (real, imaginary).
 * @tparam T The floating-point type (float or double).
 * @param magnitude The magnitude of the complex number.
 * @param phase_radians The phase of the complex number in radians.
 * @return A std::pair containing the real part (first) and imaginary part (second).
 */
template <typename T>
std::pair<T, T> polarToRectangular(T magnitude, T phase_radians);

/**
 * @brief Converts magnitude to power.
 *
 * Power is the square of the magnitude.
 *
 * @tparam T The floating-point type (float or double).
 * @param magnitude The magnitude value.
 * @return The corresponding power value.
 */
template <typename T>
T magnitudeToPower(T magnitude);

// Specialization for torch::Tensor
inline torch::Tensor magnitudeToPower(torch::Tensor magnitude) {
    return magnitude * magnitude;
}

/**
 * @brief Converts power to decibels (dB).
 *
 * The formula used is: dB = 10 * log10(power / reference_power).
 *
 * @tparam T The floating-point type (float or double).
 * @param power The power value.
 * @param reference_power The reference power. Defaults to 1.0.
 * @param avoid_nan If true (default), non-positive power inputs will result in a minimum dB
 *                  value instead of NaN. If false, returns NaN for non-positive power.
 *                  Reference power must always be positive, otherwise NaN is returned.
 * @return The power in dB.
 */
template <typename T>
T powerToDb(T power, T reference_power = static_cast<T>(1.0), bool avoid_nan = true);

// Specialization for torch::Tensor
inline torch::Tensor powerToDb(torch::Tensor power, torch::Tensor reference_power, bool avoid_nan = true) {
    auto options = power.options(); // To create tensors with same dtype and device

    // Reference power must be positive. If not, result is NaN.
    // Using float NaN as a common practice for mixed-type operations or as a default.
    auto safe_reference_power = torch::where(reference_power > 0, reference_power,
                                             torch::full_like(power, std::numeric_limits<float>::quiet_NaN()));

    torch::Tensor safe_power;
    if (avoid_nan) {
        torch::ScalarType dtype = power.scalar_type();
        torch::Tensor min_val_fill_tensor;
        if (dtype == torch::kDouble) {
            min_val_fill_tensor = torch::full_like(power, std::numeric_limits<double>::min());
        } else if (dtype == torch::kFloat) {
            min_val_fill_tensor = torch::full_like(power, std::numeric_limits<float>::min());
        } else if (dtype == torch::kHalf) {
            // c10::Half does not have std::numeric_limits::min() in the same way for positive normalized
            // It has lowest() and min() which are different.
            // Using a small constant for Half, or promoting to float for this operation might be safer.
            // For now, let's use its epsilon as a placeholder if min() isn't suitable,
            // or a pre-defined small float16 constant.
            // FLT16_MIN is approx 6.1e-5. Epsilon_f16 is 0.00097656
            // Let's use a small positive constant for half-float.
             min_val_fill_tensor = torch::full_like(power, static_cast<c10::Half>(6.0e-5f)); // Approx minimum positive half
        } else {
            // Fallback for other types (e.g., BFloat16, or integer tensors)
            // Using float min as a general small positive value.
            min_val_fill_tensor = torch::full_like(power, std::numeric_limits<float>::min());
        }
        safe_power = torch::where(power > 0, power, min_val_fill_tensor);
    } else {
        safe_power = torch::where(power > 0, power, torch::full_like(power, std::numeric_limits<float>::quiet_NaN()));
    }

    return 10 * torch::log10(safe_power / safe_reference_power);
}

// Overload for torch::Tensor with scalar reference_power
inline torch::Tensor powerToDb(torch::Tensor power, float reference_power_scalar = 1.0f, bool avoid_nan = true) {
    if (reference_power_scalar <= 0) {
        // Reference power must be positive.
        return torch::full_like(power, std::numeric_limits<float>::quiet_NaN());
    }
    // Create a tensor from the scalar reference, matching the input 'power' tensor's properties.
    torch::Tensor reference_power_tensor = torch::full_like(power, reference_power_scalar);

    torch::Tensor safe_power;
    if (avoid_nan) {
        torch::ScalarType dtype = power.scalar_type();
        torch::Tensor epsilon_fill_tensor;
        if (dtype == torch::kDouble) {
            epsilon_fill_tensor = torch::full_like(power, std::numeric_limits<double>::epsilon());
        } else if (dtype == torch::kFloat) {
            epsilon_fill_tensor = torch::full_like(power, std::numeric_limits<float>::epsilon());
        } else if (dtype == torch::kHalf) {
            epsilon_fill_tensor = torch::full_like(power, std::numeric_limits<c10::Half>::epsilon());
        } else {
            // Fallback for other types
            epsilon_fill_tensor = torch::full_like(power, std::numeric_limits<float>::epsilon());
        }
        safe_power = torch::where(power > 0, power, epsilon_fill_tensor);
    } else {
        safe_power = torch::where(power > 0, power, torch::full_like(power, std::numeric_limits<float>::quiet_NaN()));
    }

    return 10 * torch::log10(safe_power / reference_power_tensor);
}

/**
 * @brief Converts decibels (dB) to power.
 *
 * The formula used is: power = reference_power * 10^(dB / 10).
 *
 * @tparam T The floating-point type (float or double).
 * @param power_db The power in dB.
 * @param reference_power The reference power. Defaults to 1.0.
 * @return The power value. Returns NaN if reference_power is non-positive.
 */
template <typename T>
T dbToPower(T power_db, T reference_power = static_cast<T>(1.0));

// Specialization for torch::Tensor
inline torch::Tensor dbToPower(torch::Tensor power_db, torch::Tensor reference_power) {
    // Ensure reference_power is positive
    auto positive_reference_power = torch::where(reference_power > 0, reference_power, torch::tensor(std::numeric_limits<float>::quiet_NaN()).to(reference_power.device()));
    return positive_reference_power * torch::pow(10, power_db / 10);
}

// Overload for torch::Tensor with scalar reference_power
inline torch::Tensor dbToPower(torch::Tensor power_db, float reference_power_scalar = 1.0f) {
    if (reference_power_scalar <= 0) {
        return torch::full_like(power_db, std::numeric_limits<float>::quiet_NaN());
    }
    torch::Tensor reference_power_tensor = torch::full_like(power_db, reference_power_scalar);
    return reference_power_tensor * torch::pow(10, power_db / 10);
}

/**
 * @brief Converts power to magnitude.
 *
 * Magnitude is the square root of power.
 *
 * @tparam T The floating-point type (float or double).
 * @param power The power value.
 * @return The corresponding magnitude value. Returns NaN if power is negative.
 */
template <typename T>
T powerToMagnitude(T power);

// Specialization for torch::Tensor
inline torch::Tensor powerToMagnitude(torch::Tensor power) {
    // Ensure power is non-negative
    auto non_negative_power = torch::where(power >= 0, power, torch::tensor(std::numeric_limits<float>::quiet_NaN()).to(power.device()));
    return torch::sqrt(non_negative_power);
}

// Enum for spectrum data formats
enum class SpectrumDataFormat {
    COMPLEX,    // Real and Imaginary parts
    MAGPHASE,   // Magnitude and Phase (radians)
    POWERPHASE, // Power and Phase (radians)
    DBPHASE     // dB (Power) and Phase (radians)
};

CONTORCHIONIST_CORE_EXPORT SpectrumDataFormat string_to_spectrum_data_format(const std::string& format_str);
CONTORCHIONIST_CORE_EXPORT std::string spectrum_data_format_to_string(SpectrumDataFormat format);

// Enum for Mel Formula Types
enum class MelFormulaType {
    SLANEY, // Librosa default
    HTK,    // HTK formula
    CALC2   // Formula from calc~ and others (e.g. FluCoMa, Max/MSP an_melscale)
};

CONTORCHIONIST_CORE_EXPORT std::string mel_formula_type_to_string(MelFormulaType type);
CONTORCHIONIST_CORE_EXPORT MelFormulaType string_to_mel_formula_type(const std::string& type_str);

/**
 * @brief Converts a frequency from Hertz to the Mel scale using HTK-style conversion.
 * This is an alias for the default hzToMel function for clarity.
 * @tparam T The floating-point type (float or double).
 * @param frequency The frequency in Hertz.
 * @return The frequency in Mels.
 */
template <typename T>
T hzToMelHTK(T frequency);

/**
 * @brief Converts a frequency from the Mel scale to Hertz using HTK-style conversion.
 * This is an alias for the default melToHz function for clarity.
 * @tparam T The floating-point type (float or double).
 * @param mel The frequency in Mels.
 * @return The frequency in Hertz.
 */
template <typename T>
T melToHzHTK(T mel);

/**
 * @brief Converts a frequency from Hertz to the Mel scale using Slaney-style conversion.
 * This follows the librosa default conversion which uses a different scale than HTK.
 * @tparam T The floating-point type (float or double).
 * @param frequency The frequency in Hertz.
 * @return The frequency in Mels.
 */
template <typename T>
T hzToMelSlaney(T frequency);

/**
 * @brief Converts a frequency from the Mel scale to Hertz using Slaney-style conversion.
 * This follows the librosa default conversion which uses a different scale than HTK.
 * @tparam T The floating-point type (float or double).
 * @param mel The frequency in Mels.
 * @return The frequency in Hertz.
 */
template <typename T>
T melToHzSlaney(T mel);

/**
 * @brief Converts a frequency from Hertz to the Mel scale using calc2-style conversion.
 * This follows the calc2 library conversion formula.
 * @tparam T The floating-point type (float or double).
 * @param frequency The frequency in Hertz.
 * @return The frequency in Mels.
 */
template <typename T>
T hzToMelCalc2(T frequency);

/**
 * @brief Converts a frequency from the Mel scale to Hertz using calc2-style conversion.
 * This follows the alternative Mel scale conversion formula.
 * @tparam T The floating-point type (float or double).
 * @param mel The frequency in Mels.
 * @return The frequency in Hertz.
 */
template <typename T>
T melToHzCalc2(T mel);

/**
 * @brief Creates FFT frequency bins as a torch tensor.
 * @param sr Sample rate in Hz.
 * @param n_fft Size of FFT.
 * @return Torch tensor containing frequency bins from 0 to sr/2.
 */
CONTORCHIONIST_CORE_EXPORT torch::Tensor fftFrequencies(float sr, int n_fft);

/**
 * @brief Creates mel frequency points as a torch tensor.
 * @param n_mels Number of mel frequency points.
 * @param fmin Minimum frequency in Hz.
 * @param fmax Maximum frequency in Hz.
 * @param formula The mel formula to use (Slaney, HTK, Calc2).
 * @return Torch tensor containing mel frequency points in Hz.
 */
CONTORCHIONIST_CORE_EXPORT torch::Tensor melFrequencies(int n_mels, float fmin, float fmax, MelFormulaType formula = MelFormulaType::SLANEY);

/**
 * @brief Creates a mel filter bank as a torch tensor.
 * @param n_mels Number of mel filters.
 * @param fft_size Size of the FFT.
 * @param sample_rate Sample rate in Hz.
 * @param fmin Minimum frequency in Hz.
 * @param fmax Maximum frequency in Hz.
 * @param formula The mel formula to use (Slaney, HTK, Calc2).
 * @return Torch tensor of shape (n_mels, fft_size//2+1) containing the mel filter bank.
 */
CONTORCHIONIST_CORE_EXPORT torch::Tensor melFilterBank(int n_mels, int fft_size, float sample_rate, float fmin, float fmax, MelFormulaType formula = MelFormulaType::SLANEY);

/**
 * @brief Applies a mel filter bank to a spectrum (e.g., STFT magnitude) as a torch tensor.
 * @param spectrum Input spectrum (e.g., STFT magnitude) of shape (..., fft_size//2+1).
 * @param filter_bank Mel filter bank of shape (n_mels, fft_size//2+1).
 * @return Torch tensor containing the filtered spectrum.
 */
CONTORCHIONIST_CORE_EXPORT torch::Tensor applyMelFilterBank(torch::Tensor spectrum, torch::Tensor filter_bank);

/**
 * @brief Computes the mel-frequency cepstral coefficients (MFCCs) from a signal.
 * @param signal Input signal tensor.
 * @param sample_rate Sample rate of the signal.
 * @param n_mfcc Number of MFCCs to compute.
 * @param fmin Minimum frequency for mel filter bank.
 * @param fmax Maximum frequency for mel filter bank.
 * @param formula The mel formula to use.
 * @return Tensor containing the computed MFCCs.
 */
CONTORCHIONIST_CORE_EXPORT torch::Tensor computeMFCC(torch::Tensor signal, int sample_rate, int n_mfcc, float fmin, float fmax, MelFormulaType formula = MelFormulaType::SLANEY);

/**
 * @brief Computes the inverse mel-frequency cepstral coefficients (IMFCCs) from MFCCs.
 * @param mfccs Input MFCCs tensor.
 * @param sample_rate Sample rate of the original signal.
 * @param n_mels Number of mel filters used in MFCC computation.
 * @param fmin Minimum frequency for mel filter bank.
 * @param fmax Maximum frequency for mel filter bank.
 * @param formula The mel formula that was used.
 * @return Tensor containing the reconstructed signal.
 */
CONTORCHIONIST_CORE_EXPORT torch::Tensor computeIMFCC(torch::Tensor mfccs, int sample_rate, int n_mels, float fmin, float fmax, MelFormulaType formula = MelFormulaType::SLANEY);

// Explicit template instantiation declarations for exported functions
extern template CONTORCHIONIST_CORE_EXPORT float hzToMel<float>(float frequency);
extern template CONTORCHIONIST_CORE_EXPORT double hzToMel<double>(double frequency);
extern template CONTORCHIONIST_CORE_EXPORT float melToHz<float>(float mel);
extern template CONTORCHIONIST_CORE_EXPORT double melToHz<double>(double mel);
extern template CONTORCHIONIST_CORE_EXPORT float hzToMelHTK<float>(float frequency);
extern template CONTORCHIONIST_CORE_EXPORT double hzToMelHTK<double>(double frequency);
extern template CONTORCHIONIST_CORE_EXPORT float melToHzHTK<float>(float mel);
extern template CONTORCHIONIST_CORE_EXPORT double melToHzHTK<double>(double mel);
extern template CONTORCHIONIST_CORE_EXPORT float hzToMelSlaney<float>(float frequency);
extern template CONTORCHIONIST_CORE_EXPORT double hzToMelSlaney<double>(double frequency);
extern template CONTORCHIONIST_CORE_EXPORT float melToHzSlaney<float>(float mel);
extern template CONTORCHIONIST_CORE_EXPORT double melToHzSlaney<double>(double mel);
extern template CONTORCHIONIST_CORE_EXPORT float hzToMelCalc2<float>(float frequency);
extern template CONTORCHIONIST_CORE_EXPORT double hzToMelCalc2<double>(double frequency);
extern template CONTORCHIONIST_CORE_EXPORT float melToHzCalc2<float>(float mel);
extern template CONTORCHIONIST_CORE_EXPORT double melToHzCalc2<double>(double mel);

} // namespace util_conversions
} // namespace core
} // namespace contorchionist

#endif // CORE_UTIL_CONVERSIONS_H

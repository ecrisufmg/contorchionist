#include "core_util_normalizations.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace contorchionist {
namespace core {
namespace util_normalizations {

// ===== Helper Functions =====

std::string normalization_type_to_string(NormalizationType type) {
    switch (type) {
        case NormalizationType::NONE: return "none";
        case NormalizationType::BACKWARD: return "backward";
        case NormalizationType::FORWARD: return "forward";
        case NormalizationType::ORTHO: return "ortho";
        case NormalizationType::WINDOW: return "window";
        case NormalizationType::WINDOW_RECTANGULAR: return "rectangular";
        case NormalizationType::WINDOW_HANN: return "hann";
        case NormalizationType::WINDOW_HAMMING: return "hamming";
        case NormalizationType::WINDOW_BLACKMAN: return "blackman";
        case NormalizationType::WINDOW_BARTLETT: return "bartlett";
        case NormalizationType::WINDOW_TRIANGULAR: return "triangular";
        case NormalizationType::WINDOW_COSINE: return "cosine";
        case NormalizationType::WINDOW_BOXCAR: return "boxcar";
        case NormalizationType::WINDOW_TRIANG: return "triang";
        case NormalizationType::WINDOW_PARZEN: return "parzen";
        case NormalizationType::WINDOW_BOHMAN: return "bohman";
        case NormalizationType::WINDOW_NUTTALL: return "nuttall";
        case NormalizationType::WINDOW_BLACKMANHARRIS: return "blackmanharris";
        case NormalizationType::WINDOW_FLATTOP: return "flattop";
        case NormalizationType::WINDOW_BARTHANN: return "barthann";
        case NormalizationType::WINDOW_KAISER: return "kaiser";
        case NormalizationType::WINDOW_GAUSSIAN: return "gaussian";
        case NormalizationType::WINDOW_GENERAL_COSINE: return "general_cosine";
        case NormalizationType::WINDOW_GENERAL_HAMMING: return "general_hamming";
        case NormalizationType::WINDOW_CHEBWIN: return "chebwin";
        case NormalizationType::WINDOW_EXPONENTIAL: return "exponential";
        case NormalizationType::WINDOW_TUKEY: return "tukey";
        case NormalizationType::WINDOW_TAYLOR: return "taylor";
        case NormalizationType::WINDOW_DPSS: return "dpss";
        case NormalizationType::WINDOW_LANCZOS: return "lanczos";
        case NormalizationType::COHERENT_GAIN: return "coherent_gain";
        case NormalizationType::MAGNITUDE: return "magnitude";
        case NormalizationType::N_FFT: return "n_fft";
        case NormalizationType::POWERPEAK: return "power_peak";
        case NormalizationType::POWER: return "power";
        case NormalizationType::DENSITY: return "density";
        // case NormalizationType::POWER_SPECTROGRAM_WINDOW_SUM_SQUARES: return "power_spectrogram_window_sum_squares";
        // case NormalizationType::PSD_WINDOW_SUM_SQUARES: return "psd_window_sum_squares";
        default: return "unknown";
    }
}

NormalizationType string_to_normalization_type(const std::string& mode_str, bool is_inverse_op) {
    std::string lower_str = mode_str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                [](unsigned char c){ return std::tolower(c); });

    if (lower_str == "none") {
        return NormalizationType::NONE;
    } else if (lower_str == "backward" || lower_str == "n_fft") {
        return NormalizationType::BACKWARD;
    } else if (lower_str == "forward") {
        return NormalizationType::FORWARD;
    } else if (lower_str == "ortho") {
        return NormalizationType::ORTHO;
    } 
    // New simplified types (prioritized)
    else if (lower_str == "cg" || lower_str == "coherent_gain") {
        return NormalizationType::COHERENT_GAIN;
    } else if (lower_str == "magnitude" || lower_str == "mag") {
        return NormalizationType::MAGNITUDE;
    } else if (lower_str == "magnitude2" || lower_str == "mag2") {
        return NormalizationType::MAG2;
    } else if (lower_str == "power_peak" || lower_str == "powpeak") {
        return NormalizationType::POWERPEAK;
    } else if (lower_str == "power" || lower_str == "pow") {
        return NormalizationType::POWER;
    } else if (lower_str == "density" || lower_str == "psd" || lower_str == "power_spectral_density") {
        return NormalizationType::DENSITY;
    } else if (lower_str == "window" || lower_str == "win") {
        return NormalizationType::WINDOW;
    } 
    // Window-specific normalization types
    else if (lower_str == "rectangular" || lower_str == "rect") {
        return NormalizationType::WINDOW_RECTANGULAR;
    } else if (lower_str == "hann" || lower_str == "hanning") { // "hanning" common alternative
        return NormalizationType::WINDOW_HANN;
    } else if (lower_str == "hamming") {
        return NormalizationType::WINDOW_HAMMING;
    } else if (lower_str == "blackman") {
        return NormalizationType::WINDOW_BLACKMAN;
    } else if (lower_str == "bartlett") {
        return NormalizationType::WINDOW_BARTLETT;
    } else if (lower_str == "triangular") {
        return NormalizationType::WINDOW_TRIANGULAR;
    } else if (lower_str == "cosine") {
        return NormalizationType::WINDOW_COSINE;
    } else if (lower_str == "boxcar") {
        return NormalizationType::WINDOW_BOXCAR;
    } else if (lower_str == "triang") {
        return NormalizationType::WINDOW_TRIANG;
    } else if (lower_str == "parzen") {
        return NormalizationType::WINDOW_PARZEN;
    } else if (lower_str == "bohman") {
        return NormalizationType::WINDOW_BOHMAN;
    } else if (lower_str == "nuttall") {
        return NormalizationType::WINDOW_NUTTALL;
    } else if (lower_str == "blackmanharris") {
        return NormalizationType::WINDOW_BLACKMANHARRIS;
    } else if (lower_str == "flattop") {
        return NormalizationType::WINDOW_FLATTOP;
    } else if (lower_str == "barthann") {
        return NormalizationType::WINDOW_BARTHANN;
    } else if (lower_str == "kaiser") {
        return NormalizationType::WINDOW_KAISER;
    } else if (lower_str == "gaussian") {
        return NormalizationType::WINDOW_GAUSSIAN;
    } else if (lower_str == "general_cosine") {
        return NormalizationType::WINDOW_GENERAL_COSINE;
    } else if (lower_str == "general_hamming") {
        return NormalizationType::WINDOW_GENERAL_HAMMING;
    } else if (lower_str == "chebwin") {
        return NormalizationType::WINDOW_CHEBWIN;
    } else if (lower_str == "exponential") {
        return NormalizationType::WINDOW_EXPONENTIAL;
    } else if (lower_str == "tukey") {
        return NormalizationType::WINDOW_TUKEY;
    } else if (lower_str == "taylor") {
        return NormalizationType::WINDOW_TAYLOR;
    } else if (lower_str == "dpss") {
        return NormalizationType::WINDOW_DPSS;
    } else if (lower_str == "lanczos") {
        return NormalizationType::WINDOW_LANCZOS;
    }
    
    // Fallback to default based on operation type
    return is_inverse_op ? NormalizationType::FORWARD : NormalizationType::BACKWARD;
}

contorchionist::core::util_windowing::Type normalization_mode_to_fixed_window_type(NormalizationType mode) {
    switch (mode) {
        case NormalizationType::WINDOW_RECTANGULAR: return contorchionist::core::util_windowing::Type::RECTANGULAR;
        case NormalizationType::WINDOW_HANN: return contorchionist::core::util_windowing::Type::HANN;
        case NormalizationType::WINDOW_HAMMING: return contorchionist::core::util_windowing::Type::HAMMING;
        case NormalizationType::WINDOW_BLACKMAN: return contorchionist::core::util_windowing::Type::BLACKMAN;
        case NormalizationType::WINDOW_BARTLETT: return contorchionist::core::util_windowing::Type::BARTLETT;
        case NormalizationType::WINDOW_TRIANGULAR: return contorchionist::core::util_windowing::Type::BARTLETT; // Often Bartlett
        case NormalizationType::WINDOW_COSINE: return contorchionist::core::util_windowing::Type::COSINE;
        case NormalizationType::WINDOW_BOXCAR: return contorchionist::core::util_windowing::Type::BOXCAR;
        case NormalizationType::WINDOW_TRIANG: return contorchionist::core::util_windowing::Type::TRIANG;
        case NormalizationType::WINDOW_PARZEN: return contorchionist::core::util_windowing::Type::PARZEN;
        case NormalizationType::WINDOW_BOHMAN: return contorchionist::core::util_windowing::Type::BOHMAN;
        case NormalizationType::WINDOW_NUTTALL: return contorchionist::core::util_windowing::Type::NUTTALL;
        case NormalizationType::WINDOW_BLACKMANHARRIS: return contorchionist::core::util_windowing::Type::BLACKMANHARRIS;
        case NormalizationType::WINDOW_FLATTOP: return contorchionist::core::util_windowing::Type::FLATTOP;
        case NormalizationType::WINDOW_BARTHANN: return contorchionist::core::util_windowing::Type::BARTHANN;
        case NormalizationType::WINDOW_KAISER: return contorchionist::core::util_windowing::Type::KAISER;
        case NormalizationType::WINDOW_GAUSSIAN: return contorchionist::core::util_windowing::Type::GAUSSIAN;
        case NormalizationType::WINDOW_GENERAL_COSINE: return contorchionist::core::util_windowing::Type::GENERAL_COSINE;
        case NormalizationType::WINDOW_GENERAL_HAMMING: return contorchionist::core::util_windowing::Type::GENERAL_HAMMING;
        case NormalizationType::WINDOW_CHEBWIN: return contorchionist::core::util_windowing::Type::CHEBWIN;
        case NormalizationType::WINDOW_EXPONENTIAL: return contorchionist::core::util_windowing::Type::EXPONENTIAL;
        case NormalizationType::WINDOW_TUKEY: return contorchionist::core::util_windowing::Type::TUKEY;
        case NormalizationType::WINDOW_TAYLOR: return contorchionist::core::util_windowing::Type::TAYLOR;
        case NormalizationType::WINDOW_DPSS: return contorchionist::core::util_windowing::Type::DPSS;
        case NormalizationType::WINDOW_LANCZOS: return contorchionist::core::util_windowing::Type::LANCZOS;
        default:
            // This function is only for modes that imply a fixed window.
            // For other modes, especially NormalizationType::WINDOW, COHERENT_GAIN, MAGNITUDE, 
            // POWER, DENSITY, etc., the actual applied window's sum should be used.
            throw std::invalid_argument("Normalization mode does not imply a fixed window type for sum calculation.");
    }
}

NormalizationType window_type_to_normalization_type(contorchionist::core::util_windowing::Type window_type) {
    switch (window_type) {
        case contorchionist::core::util_windowing::Type::RECTANGULAR: return NormalizationType::WINDOW_RECTANGULAR;
        case contorchionist::core::util_windowing::Type::HANN: return NormalizationType::WINDOW_HANN;
        case contorchionist::core::util_windowing::Type::HAMMING: return NormalizationType::WINDOW_HAMMING;
        case contorchionist::core::util_windowing::Type::BLACKMAN: return NormalizationType::WINDOW_BLACKMAN;
        case contorchionist::core::util_windowing::Type::BARTLETT: return NormalizationType::WINDOW_BARTLETT;
        case contorchionist::core::util_windowing::Type::COSINE: return NormalizationType::WINDOW_COSINE;
        case contorchionist::core::util_windowing::Type::BOXCAR: return NormalizationType::WINDOW_BOXCAR;
        case contorchionist::core::util_windowing::Type::TRIANG: return NormalizationType::WINDOW_TRIANG;
        case contorchionist::core::util_windowing::Type::PARZEN: return NormalizationType::WINDOW_PARZEN;
        case contorchionist::core::util_windowing::Type::BOHMAN: return NormalizationType::WINDOW_BOHMAN;
        case contorchionist::core::util_windowing::Type::NUTTALL: return NormalizationType::WINDOW_NUTTALL;
        case contorchionist::core::util_windowing::Type::BLACKMANHARRIS: return NormalizationType::WINDOW_BLACKMANHARRIS;
        case contorchionist::core::util_windowing::Type::FLATTOP: return NormalizationType::WINDOW_FLATTOP;
        case contorchionist::core::util_windowing::Type::BARTHANN: return NormalizationType::WINDOW_BARTHANN;
        case contorchionist::core::util_windowing::Type::KAISER: return NormalizationType::WINDOW_KAISER;
        case contorchionist::core::util_windowing::Type::GAUSSIAN: return NormalizationType::WINDOW_GAUSSIAN;
        case contorchionist::core::util_windowing::Type::GENERAL_COSINE: return NormalizationType::WINDOW_GENERAL_COSINE;
        case contorchionist::core::util_windowing::Type::GENERAL_HAMMING: return NormalizationType::WINDOW_GENERAL_HAMMING;
        case contorchionist::core::util_windowing::Type::CHEBWIN: return NormalizationType::WINDOW_CHEBWIN;
        case contorchionist::core::util_windowing::Type::EXPONENTIAL: return NormalizationType::WINDOW_EXPONENTIAL;
        case contorchionist::core::util_windowing::Type::TUKEY: return NormalizationType::WINDOW_TUKEY;
        case contorchionist::core::util_windowing::Type::TAYLOR: return NormalizationType::WINDOW_TAYLOR;
        case contorchionist::core::util_windowing::Type::DPSS: return NormalizationType::WINDOW_DPSS;
        case contorchionist::core::util_windowing::Type::LANCZOS: return NormalizationType::WINDOW_LANCZOS;
        default:
            throw std::invalid_argument("Window type does not have a corresponding specific normalization mode.");
    }
}

bool is_fixed_window_specific_normalization(NormalizationType mode) {
    return mode == NormalizationType::WINDOW_RECTANGULAR ||
           mode == NormalizationType::WINDOW_HANN ||
           mode == NormalizationType::WINDOW_HAMMING ||
           mode == NormalizationType::WINDOW_BLACKMAN ||
           mode == NormalizationType::WINDOW_BARTLETT ||
           mode == NormalizationType::WINDOW_TRIANGULAR ||
           mode == NormalizationType::WINDOW_COSINE ||
           mode == NormalizationType::WINDOW_BOXCAR ||
           mode == NormalizationType::WINDOW_TRIANG ||
           mode == NormalizationType::WINDOW_PARZEN ||
           mode == NormalizationType::WINDOW_BOHMAN ||
           mode == NormalizationType::WINDOW_NUTTALL ||
           mode == NormalizationType::WINDOW_BLACKMANHARRIS ||
           mode == NormalizationType::WINDOW_FLATTOP ||
           mode == NormalizationType::WINDOW_BARTHANN ||
           mode == NormalizationType::WINDOW_KAISER ||
           mode == NormalizationType::WINDOW_GAUSSIAN ||
           mode == NormalizationType::WINDOW_GENERAL_COSINE ||
           mode == NormalizationType::WINDOW_GENERAL_HAMMING ||
           mode == NormalizationType::WINDOW_CHEBWIN ||
           mode == NormalizationType::WINDOW_EXPONENTIAL ||
           mode == NormalizationType::WINDOW_TUKEY ||
           mode == NormalizationType::WINDOW_TAYLOR ||
           mode == NormalizationType::WINDOW_DPSS ||
           mode == NormalizationType::WINDOW_LANCZOS;
}

// New simplified normalization functions
SpectrumNormFactors NormalizationFactors::get_norm_factors(
    int fft_mode,
    long n_fft,
    const torch::Tensor& window_tensor,
    NormalizationType norm_type,
    float fs,
    bool scale_by_window_sum,
    float window_sum_for_scaling
) {
    if (n_fft <= 0) {
        throw std::invalid_argument("n_fft must be positive");
    }
    
    SpectrumNormFactors factors;
    factors.overall_scale = 1.0f; // Default initialize
    
    // Calculate window sums
    float sum_window = 0.0f;
    float sum_win_vals_pow2 = 0.0f;
    
    if (window_tensor.numel() > 0) {
        sum_window = torch::sum(window_tensor).item<float>();
        sum_win_vals_pow2 = torch::sum(window_tensor * window_tensor).item<float>();
    } else {
        // Default to rectangular window if no window provided
        sum_window = static_cast<float>(n_fft);
        sum_win_vals_pow2 = static_cast<float>(n_fft);
    }
    
    // Avoid division by zero
    if (sum_window == 0.0f) sum_window = 1.0f;
    if (sum_win_vals_pow2 == 0.0f) sum_win_vals_pow2 = 1.0f;
    if (fs <= 0.0f) fs = 1.0f;
    factors.overall_scale = 1.0f; // Default overall scale for forward FFT 
        // Apply overall_scale logic based on scale_by_window_sum,
    // but only if norm_type is not NONE (handled above)
    // if (norm_type != NormalizationType::NONE) {
    //     if (scale_by_window_sum && sum_window > 0.0f) {
    //         factors.overall_scale = 1.0f / sum_window;
    //     } else {
    //         factors.overall_scale = 1.0f; // Default for forward if not scaling by window sum
    //     }
    // }
    // If norm_type IS NONE, factors.overall_scale is already 1.0f from initialization or explicit set.
    
    
    switch (norm_type) {
        // None: no normalization, just return default factors
        case NormalizationType::NONE:
            factors.mag_dc = 1.0f;
            factors.mag_nyquist = 1.0f;
            factors.mag_ac = 1.0f;
            factors.overall_scale = 1.0f; // Explicitly set for NONE type
            factors.pw_dc = 1.0f;
            factors.pw_nyquist = 1.0f;
            factors.pw_ac = 1.0f;
            // overall_scale remains 1.0f as per plan for NONE type
            break;
        
        // Magnitude normalization by window sum (coherent gain): sum(w[i])
        case NormalizationType::WINDOW:
        case NormalizationType::COHERENT_GAIN:
        case NormalizationType::MAGNITUDE:
            factors.mag_dc = 1.0f / sum_window;
            factors.mag_nyquist = 1.0f / sum_window;
            factors.mag_ac = static_cast<float>(fft_mode) / sum_window;
            factors.pw_dc = 1.0f;
            factors.pw_nyquist = 1.0f;
            factors.pw_ac = 1.0f;
            break;

        case NormalizationType::MAG2:
            factors.mag_dc = 2.0f / sum_window;
            factors.mag_nyquist = 2.0f / sum_window;
            factors.mag_ac = 2.0f * static_cast<float>(fft_mode) / sum_window;
            factors.pw_dc = 1.0f;
            factors.pw_nyquist = 1.0f;
            factors.pw_ac = 1.0f;
            break;    
        
        // Magnitude normalization by window size (n_fft): N    
        case NormalizationType::BACKWARD:
        case NormalizationType::N_FFT:
            factors.mag_dc = 1.0f / static_cast<float>(n_fft);
            factors.mag_nyquist = 1.0f / static_cast<float>(n_fft);
            factors.mag_ac = static_cast<float>(fft_mode) / static_cast<float>(n_fft);
            factors.pw_dc = 1.0f;
            factors.pw_nyquist = 1.0f;
            factors.pw_ac = 1.0f;
            break;
         
        // Power normalization by the sum of squared window values: sum(w[i]^2)
        // This is best suited for noise, non-tonal signals
        case NormalizationType::POWER: 
            factors.mag_dc = 1.0f; 
            factors.mag_nyquist = 1.0f;
            factors.mag_ac = 1.0f;
            factors.pw_dc = 1.0f / sum_win_vals_pow2;
            factors.pw_nyquist = 1.0f / sum_win_vals_pow2;
            factors.pw_ac = static_cast<float>(fft_mode) / sum_win_vals_pow2;
            break;

        // Normalization by the sum of squared window values divided by the sum of window values: sum(w[i]^2) / sum(w[i])
        case NormalizationType::POWERPEAK: // Power peak normalization: best suited for tonal signals (peak invariant)
            factors.mag_dc = 1.0f;
            factors.mag_nyquist = 1.0f;
            factors.mag_ac = 1.0f;
            factors.pw_dc = (1.0f / sum_win_vals_pow2) / sum_window; // Adjust for power normalization
            factors.pw_nyquist = (1.0f / sum_win_vals_pow2) / sum_window;
            factors.pw_ac = (static_cast<float>(fft_mode) / sum_win_vals_pow2) / sum_window;
            break;
            
        case NormalizationType::DENSITY:
            factors.mag_dc = 1.0f;
            factors.mag_nyquist = 1.0f;
            factors.mag_ac = 1.0f;
            factors.pw_dc = 1.0f / (fs * sum_win_vals_pow2);
            factors.pw_nyquist = 1.0f / (fs * sum_win_vals_pow2);
            factors.pw_ac = static_cast<float>(fft_mode) / (fs * sum_win_vals_pow2);
            break;
            
        default:
            // For unsupported types, return no normalization
            factors.mag_dc = 1.0f;
            factors.mag_nyquist = 1.0f;
            factors.mag_ac = 1.0f;
            factors.pw_dc = 1.0f;
            factors.pw_nyquist = 1.0f;
            factors.pw_ac = 1.0f;
            break;
    }


    return factors;
}

SpectrumNormFactors NormalizationFactors::get_denorm_factors(
    int fft_mode,
    long n_fft,
    const torch::Tensor& window_tensor,
    NormalizationType norm_type,
    float fs
) {
    if (n_fft <= 0) {
        throw std::invalid_argument("n_fft must be positive");
    }
    
    SpectrumNormFactors factors;
  // Default initialize, will be overridden
    
    // Calculate window sums
    float sum_window = 0.0f;
    float sum_win_vals_pow2 = 0.0f;
    
    if (window_tensor.numel() > 0) {
        sum_window = torch::sum(window_tensor).item<float>();
        sum_win_vals_pow2 = torch::sum(window_tensor * window_tensor).item<float>();
    } else {
        // Default to rectangular window if no window provided
        sum_window = static_cast<float>(n_fft);
        sum_win_vals_pow2 = static_cast<float>(n_fft);
    }
    
    // Avoid division by zero
    if (sum_window == 0.0f) sum_window = 1.0f;
    if (sum_win_vals_pow2 == 0.0f) sum_win_vals_pow2 = 1.0f;
    if (fs <= 0.0f) fs = 1.0f;

    factors.overall_scale = 1.0f; 
    
    switch (norm_type) {
        case NormalizationType::NONE:
            factors.mag_dc = 1.0f;
            factors.mag_nyquist = 1.0f;
            factors.mag_ac = 1.0f;
            factors.pw_dc = 1.0f;
            factors.pw_nyquist = 1.0f;
            factors.pw_ac = 1.0f;
            factors.overall_scale = 1.0f; // Explicitly set for NONE type
            break;
        
        // Magnitude denormalization by window sum (coherent gain)
        case NormalizationType::WINDOW:
        case NormalizationType::COHERENT_GAIN:
        case NormalizationType::MAGNITUDE:
            factors.mag_dc = sum_window;
            factors.mag_nyquist = sum_window;
            factors.mag_ac = sum_window / static_cast<float>(fft_mode);
            factors.pw_dc = 1.0f;
            factors.pw_nyquist = 1.0f;
            factors.pw_ac = 1.0f;
            break;
            
        // Magnitude denormalization by window sum (coherent gain)
        case NormalizationType::MAG2:
            factors.mag_dc = sum_window / 2.0f;
            factors.mag_nyquist = sum_window / 2.0f;
            factors.mag_ac = sum_window / (2.0f * static_cast<float>(fft_mode));
            factors.pw_dc = 1.0f;
            factors.pw_nyquist = 1.0f;
            factors.pw_ac = 1.0f;
            break;

        // Magnitude denormalization by window size (n_fft) 
        case NormalizationType::BACKWARD:
        case NormalizationType::N_FFT:
            factors.mag_dc = static_cast<float>(n_fft);
            factors.mag_nyquist = static_cast<float>(n_fft);
            factors.mag_ac = static_cast<float>(n_fft) / static_cast<float>(fft_mode);
            factors.pw_dc = 1.0f;
            factors.pw_nyquist = 1.0f;
            factors.pw_ac = 1.0f;
            break;
            

        // Power denormalization by the sum of squared window values 
        case NormalizationType::POWER:
            factors.mag_dc = 1.0f;
            factors.mag_nyquist = 1.0f;
            factors.mag_ac = 1.0f;
            factors.pw_dc = sum_win_vals_pow2;
            factors.pw_nyquist = sum_win_vals_pow2;
            factors.pw_ac = sum_win_vals_pow2 / static_cast<float>(fft_mode);
            break;
        
        // Denormalization by the sum of squared window values divided by the sum of window values: sum(w[i]^2) / sum(w[i])
        case NormalizationType::POWERPEAK: // Power peak normalization: best suited for tonal signals (peak invariant)
            factors.mag_dc = 1.0f;
            factors.mag_nyquist = 1.0f;
            factors.mag_ac = 1.0f;
            factors.pw_dc = sum_window / (1.0f / sum_win_vals_pow2);
            factors.pw_nyquist =  sum_window / (1.0f / sum_win_vals_pow2);
            factors.pw_ac = sum_window / ((static_cast<float>(fft_mode) / sum_win_vals_pow2));
            break;
            
        case NormalizationType::DENSITY:
            // Denormalization is the reciprocal of normalization
            // Norm: 1/(fs * sum_win_vals_pow2), Denorm: (fs * sum_win_vals_pow2)
            factors.mag_dc = 1.0f;
            factors.mag_nyquist = 1.0f;
            factors.mag_ac = 1.0f;
            factors.pw_dc = (fs * sum_win_vals_pow2);
            factors.pw_nyquist = (fs * sum_win_vals_pow2);
            factors.pw_ac = (fs * sum_win_vals_pow2) / static_cast<float>(fft_mode);
            break;
            
        default:
            // For unsupported types, return no denormalization
            factors.mag_dc = 1.0f;
            factors.mag_nyquist = 1.0f;
            factors.mag_ac = 1.0f;
            factors.pw_dc = 1.0f;
            factors.pw_nyquist = 1.0f;
            factors.pw_ac = 1.0f;
            // overall_scale for default denorm is 2.0/n_fft, handled below
            break;
    }

    // Set overall_scale for denormalization
    // If norm_type was NONE, it's already 1.0f and will be returned as such.
    // Otherwise, apply the standard IRFFT scaling.
    if (norm_type != NormalizationType::NONE) {
        if (n_fft > 0) { // Ensure n_fft is positive to avoid division by zero or infinity
            factors.overall_scale = 2.0f / static_cast<float>(n_fft);
        } else {
            factors.overall_scale = 1.0f; // Fallback if n_fft is not valid, though previous checks should catch this
        }
    }
    
    return factors;
}

} // namespace normalizations
} // namespace core
} // namespace contorchionist
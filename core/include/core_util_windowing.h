#ifndef CORE_UTIL_WINDOWING_H
#define CORE_UTIL_WINDOWING_H

#include <torch/torch.h>
#include <string>
#include <vector>
#include <cstdint>   // For int64_t
#include <stdexcept> // For std::invalid_argument
#include "contorchionist_core/contorchionist_core_export.h" // Added for symbol export


namespace contorchionist {
    namespace core {
        namespace util_windowing {

// Enum for LibTorch supported window types
enum class Type {
    RECTANGULAR, // torch::ones
    HANN,        // torch::hann_window
    HAMMING,     // torch::hamming_window
    BLACKMAN,    // torch::blackman_window
    BARTLETT,    // torch::bartlett_window (Triangular)
    COSINE,      // Implemented using sin based on scipy's cosine window which is sin(pi * k / (N-1))
                // SciPy's "cosine" is actually a sine window. PyTorch's native cosine is different.
                // We'll stick to SciPy's definition for now if that's the goal.

    // New window types from SciPy to be added
    BOXCAR,          // Equivalent to RECTANGULAR, torch::ones
    TRIANG,          // Triangular window, not necessarily zero at ends
    PARZEN,
    BOHMAN,
    NUTTALL,         // Uses general_cosine
    BLACKMANHARRIS,  // Uses general_cosine
    FLATTOP,         // Uses general_cosine
    BARTHANN,
    KAISER,          // Requires beta
    // KAISER_BESSEL_DERIVED, // Requires beta, M must be even - might skip for now due to complexity
    GAUSSIAN,        // Requires std
    GENERAL_COSINE,  // Requires a coefficient array 'a'
    GENERAL_HAMMING, // Requires alpha
    CHEBWIN,         // Requires attenuation 'at'
    EXPONENTIAL,     // Requires center and tau
    TUKEY,           // Requires alpha (shape parameter)
    TAYLOR,          // Requires nbar, sll, norm
    DPSS,            // Requires NW, Kmax
    LANCZOS          // Sinc window
};

/**
 * @brief Generates a window tensor using LibTorch functions.
 *
 * @param window_length The desired length of the window.
 * @param type The type of window to generate.
 * @param periodic Whether the window is periodic (typically true for FFT usage).
 * @param options Tensor options (device, dtype, etc.).
 * @return torch::Tensor The generated window.
 */
CONTORCHIONIST_CORE_EXPORT torch::Tensor generate_torch_window( // Added EXPORT
    int64_t window_length,
    Type type,
    bool periodic,
    const torch::TensorOptions& options
);

enum class Alignment { LEFT, CENTER, RIGHT };
CONTORCHIONIST_CORE_EXPORT Alignment string_to_torch_window_alignment(const std::string& strAlign);
CONTORCHIONIST_CORE_EXPORT std::string torch_window_alignment_to_string(Alignment alignment);

CONTORCHIONIST_CORE_EXPORT torch::Tensor generate_torch_window_aligned(
    int64_t window_length,
    Type type,
    bool periodic,
    int64_t zero_padding_samples,
    Alignment alignment,
    const torch::TensorOptions& options
);

/**
 * @brief Converts a string to a TorchWindowing::Type.
 * @param strType The string representation of the window type.
 * @return TorchWindowing::Type. Throws std::invalid_argument if the string is not recognized.
 */
CONTORCHIONIST_CORE_EXPORT Type string_to_torch_window_type(const std::string& strType);

/**
 * @brief Converts a TorchWindowing::Type to its string representation.
 * @param type The window type.
 * @return std::string.
 */
CONTORCHIONIST_CORE_EXPORT std::string torch_window_type_to_string(Type type);


// Specific window creation functions
// These allow passing window-specific parameters where needed.

// Existing types already covered by generate_torch_window directly or via LibTorch
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_hann_window(int64_t window_length, bool periodic, const torch::TensorOptions& options);
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_hamming_window(int64_t window_length, bool periodic, const torch::TensorOptions& options, double alpha = 0.54, double beta = 0.46); // Default to SciPy's Hamming
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_blackman_window(int64_t window_length, bool periodic, const torch::TensorOptions& options);
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_bartlett_window(int64_t window_length, bool periodic, const torch::TensorOptions& options);
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_cosine_window(int64_t window_length, bool periodic, const torch::TensorOptions& options); // SciPy's cosine (sine shape)
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_rectangular_window(int64_t window_length, const torch::TensorOptions& options); // Periodic has no effect

// New SciPy window types
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_boxcar_window(int64_t window_length, bool periodic, const torch::TensorOptions& options); // periodic is relevant due to _extend logic
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_triang_window(int64_t window_length, bool periodic, const torch::TensorOptions& options);
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_parzen_window(int64_t window_length, bool periodic, const torch::TensorOptions& options);
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_bohman_window(int64_t window_length, bool periodic, const torch::TensorOptions& options);
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_nuttall_window(int64_t window_length, bool periodic, const torch::TensorOptions& options);
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_blackmanharris_window(int64_t window_length, bool periodic, const torch::TensorOptions& options);
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_flattop_window(int64_t window_length, bool periodic, const torch::TensorOptions& options);
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_barthann_window(int64_t window_length, bool periodic, const torch::TensorOptions& options);
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_lanczos_window(int64_t window_length, bool periodic, const torch::TensorOptions& options);

// Windows requiring specific parameters
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_kaiser_window(int64_t window_length, double beta, bool periodic, const torch::TensorOptions& options);
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_gaussian_window(int64_t window_length, double std_dev, bool periodic, const torch::TensorOptions& options);
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_general_cosine_window(int64_t window_length, const torch::Tensor& coefficients, bool periodic, const torch::TensorOptions& options);
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_general_hamming_window(int64_t window_length, double alpha, bool periodic, const torch::TensorOptions& options);
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_chebwin_window(int64_t window_length, double attenuation, bool periodic, const torch::TensorOptions& options); // Chebwin - at
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_exponential_window(int64_t window_length, bool periodic, const torch::TensorOptions& options, c10::optional<double> center = c10::nullopt, double tau = 1.0);
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_tukey_window(int64_t window_length, double alpha, bool periodic, const torch::TensorOptions& options);
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_taylor_window(int64_t window_length, int nbar, double sll, bool norm, bool periodic, const torch::TensorOptions& options);
CONTORCHIONIST_CORE_EXPORT torch::Tensor create_dpss_window(int64_t window_length, double nw, int kmax, bool periodic, const torch::TensorOptions& options); // Simplified for now, Kmax usually means multiple windows


        } // namespace util_windowing
    } // namespace core
} // namespace contorchionist

#endif // CORE_UTIL_WINDOWING_H

    #ifndef CORE_UTIL_NORMALIZATIONS_H
    #define CORE_UTIL_NORMALIZATIONS_H

    #include <string>
    #include <torch/torch.h> // For torch::Tensor and torch::fft
    #include "contorchionist_core/contorchionist_core_export.h" // For symbol export
    #include "core_util_windowing.h" // For contorchionist::core::util_windowing::Type, assuming it might be needed for window type to string or similar indirect use.
                        // If not directly needed by functions here, it's okay as it's a header-only dependency if any.

    namespace contorchionist {
    namespace core {
    namespace util_normalizations {

    // Unified enum for Normalization Types
    enum class NormalizationType {
        NONE,      // Default, often maps to BACKWARD for forward transform
        BACKWARD,  // No normalization (or scaled by 1/N in some contexts, LibTorch default for rfft)
        FORWARD,   // Scales by 1/N (LibTorch default for irfft)
        ORTHO,     // Scales by 1/sqrt(N)
        WINDOW,    // Scales by K / sum(window coefficients) where K is 1.0 for FFT and 2.0 for RFFT (for peak preservation of sinusoids)

        // Window-specific normalization modes (independent of actual windowing application, uses sum of a specific window type)
        // Extended to support all window types from torchwins.h for future-proofing and completeness
        WINDOW_RECTANGULAR,
        WINDOW_HANN,
        WINDOW_HAMMING,
        WINDOW_BLACKMAN,
        WINDOW_BARTLETT,       // torch::bartlett_window (Triangular)
        WINDOW_TRIANGULAR,     // Explicitly (alias for BARTLETT)
        WINDOW_COSINE,         // SciPy's cosine (sine shape)
        
        // New SciPy window types
        WINDOW_BOXCAR,          // Equivalent to RECTANGULAR
        WINDOW_TRIANG,          // Triangular window, not necessarily zero at ends
        WINDOW_PARZEN,
        WINDOW_BOHMAN,
        WINDOW_NUTTALL,         // Uses general_cosine
        WINDOW_BLACKMANHARRIS,  // Uses general_cosine
        WINDOW_FLATTOP,         // Uses general_cosine
        WINDOW_BARTHANN,
        WINDOW_KAISER,          // Requires beta (default parameters for normalization)
        WINDOW_GAUSSIAN,        // Requires std (default parameters for normalization)
        WINDOW_GENERAL_COSINE,  // Uses general_cosine (default parameters for normalization)
        WINDOW_GENERAL_HAMMING, // Requires alpha (default parameters for normalization)
        WINDOW_CHEBWIN,         // Requires attenuation 'at' (default parameters for normalization)
        WINDOW_EXPONENTIAL,     // Requires center and tau (default parameters for normalization)
        WINDOW_TUKEY,           // Requires alpha shape parameter (default parameters for normalization)
        WINDOW_TAYLOR,          // Requires nbar, sll, norm (default parameters for normalization)
        WINDOW_DPSS,            // Requires NW, Kmax (default parameters for normalization)
        WINDOW_LANCZOS,         // Sinc window

        // New simplified normalization types for get_norm_factors/get_denorm_factors
        COHERENT_GAIN,          // Alias for WINDOW (coherent gain normalization)
        MAGNITUDE,              // Alias for WINDOW (magnitude normalization)
        MAG2,                   
        N_FFT,                  // Normalization by n_fft (was SRM)
        POWERPEAK,              // Power normalization using sum(win^2) / sum(win)
        POWER,                  // Power normalization using sum(win^2)
        DENSITY                 // Power spectral density normalization
    };

    // Helper to convert NormalizationType to string for LibTorch and logging
    CONTORCHIONIST_CORE_EXPORT std::string normalization_type_to_string(NormalizationType type);

    // Helper to convert string to NormalizationType
    CONTORCHIONIST_CORE_EXPORT NormalizationType string_to_normalization_type(const std::string& mode_str, bool is_inverse_op = false);

    // Helper to get a specific window type from normalization mode (for window-specific modes)
    // This is useful if a normalization mode implies using a fixed window type for its sum calculation,
    // regardless of the window actually applied to the signal.
    CONTORCHIONIST_CORE_EXPORT contorchionist::core::util_windowing::Type normalization_mode_to_fixed_window_type(NormalizationType mode);

    // Helper to convert a window type to its corresponding window-specific normalization mode
    // This is useful for automatically selecting window-based normalization when a window is specified
    CONTORCHIONIST_CORE_EXPORT NormalizationType window_type_to_normalization_type(contorchionist::core::util_windowing::Type window_type);

    // Helper to check if normalization mode is one of the fixed window-specific types
    CONTORCHIONIST_CORE_EXPORT bool is_fixed_window_specific_normalization(NormalizationType mode);

    // Struct to hold differentiated normalization factors for different spectral components
    struct SpectrumNormFactors {
        float mag_dc = 1.0f;      // Mag Factor for DC bin (bin 0)
        float mag_nyquist = 1.0f; // Mag Factor for Nyquist bin (bin N/2 for RFFT)
        float mag_ac = 1.0f;      // Mag Factor for AC bins (all other bins)
        float pw_dc = 1.0f;       // PW Factor for DC bin (bin 0)
        float pw_nyquist = 1.0f;  // PW Factor for Nyquist bin (bin N/2 for RFFT)
        float pw_ac = 1.0f;       // PW Factor for AC bins (all other bins)
        float overall_scale = 1.0f; // Overall scaling factor applied after specific component normalization
    };


    class NormalizationFactors {
    public:
        /**
         * @brief Gets normalization factors for analysis (forward transform) with simplified interface.
         *
         * @param fft_mode 1 for FFT, 2 for RFFT
         * @param n_fft The size of the FFT.
         * @param window_tensor The window tensor.
         * @param norm_type The normalization type to apply.
         * @param fs The sampling rate of the audio signal.
         * @return SpectrumNormFactors with differentiated factors for DC, AC, and Nyquist bins.
         */
        static CONTORCHIONIST_CORE_EXPORT SpectrumNormFactors get_norm_factors(
            int fft_mode,
            long n_fft,
            const torch::Tensor& window_tensor,
            NormalizationType norm_type,
            float fs,
            bool scale_by_window_sum,
            float window_sum_for_scaling
        );

        /**
         * @brief Gets denormalization factors for synthesis (inverse transform) with simplified interface.
         *
         * @param fft_mode 1 for FFT, 2 for RFFT
         * @param n_fft The size of the FFT.
         * @param window_tensor The window tensor.
         * @param norm_type The normalization type that was applied in analysis.
         * @param fs The sampling rate of the audio signal.
         * @return SpectrumNormFactors with differentiated factors for DC, AC, and Nyquist bins.
         */
        static CONTORCHIONIST_CORE_EXPORT SpectrumNormFactors get_denorm_factors(
            int fft_mode,
            long n_fft,
            const torch::Tensor& window_tensor,
            NormalizationType norm_type,
            float fs
        );
    };


    } // namespace normalizations
    } // namespace core
    } // namespace contorchionist

    #endif // CORE_UTIL_NORMALIZATIONS_H

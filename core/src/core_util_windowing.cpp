// filepath: /home/padovani/data/gdrive/meus_dev/audio_multi_sys/working_dir/contorchionist/core/internal/torchwins.cpp
#include "core_util_windowing.h"
#include <algorithm> // For std::transform
#include <vector>
#include <cmath>     // For M_PI, sin, cos, abs, pow, sqrt, floor
#include <stdexcept> // For std::invalid_argument, std::logic_error
#include <utility>   // For std::pair

#ifndef M_PI
#define M_PI (atan(1.0)*4)
#endif

// Helper function to mimic SciPy's _extend and _truncate logic
// Returns {adjusted_M, needs_truncation}
// 'periodic_scipy_style' means it behaves like SciPy's sym=False (periodic for spectral analysis)
std::pair<int64_t, bool> extend_truncate_params(int64_t M, bool periodic_scipy_style) {
    if (periodic_scipy_style) { // Corresponds to sym=False in SciPy
        return {M + 1, true};
    } else { // Corresponds to sym=True in SciPy
        return {M, false};
    }
}

torch::Tensor truncate_if_needed(torch::Tensor w, bool needs_truncation) {
    if (needs_truncation && w.numel() > 0) {
        // Ensure we don't try to slice beyond the start if w.numel() is 1 and needs_truncation is true
        if (w.numel() == 1) return torch::empty({0}, w.options());
        return w.slice(0, 0, w.numel() - 1);
    }
    return w;
}

// Helper function to select appropriate dtype for MPS compatibility
torch::TensorOptions get_compatible_options(const torch::TensorOptions& options_in) {
    torch::ScalarType target_dtype;
    if (!options_in.has_dtype()) {
        // Default based on device compatibility
        if (options_in.device().is_mps()) {
            target_dtype = torch::kFloat32;
        } else {
            target_dtype = torch::kFloat64;
        }
    } else if (options_in.dtype() == torch::kFloat64 && options_in.device().is_mps()) {
        // Force float32 for MPS if float64 was requested
        target_dtype = torch::kFloat32;
    } else {
        target_dtype = options_in.dtype().toScalarType();
    }
    return options_in.dtype(target_dtype);
}

// Corresponds to SciPy's _len_guards
// Returns true if M is small (<=1) and a default ones tensor should be returned.
// Throws if M is invalid.
bool len_guards(int64_t M, torch::Tensor& out_tensor, const torch::TensorOptions& options) {
    if (static_cast<double>(M) != static_cast<double>(static_cast<int64_t>(M)) || M < 0) { // Check for non-integer or negative M
        throw std::invalid_argument("Window length M must be a non-negative integer");
    }
    if (M == 0) {
        out_tensor = torch::empty({0}, get_compatible_options(options));
        return true;
    }
    if (M == 1) {
        out_tensor = torch::ones({1}, get_compatible_options(options));
        return true;
    }
    return false; // M > 1, proceed with normal computation
}

namespace contorchionist {
    namespace core {
        namespace util_windowing {

    // Forward declaration for general_cosine as it's used by others
    torch::Tensor general_cosine_impl(int64_t M, const torch::Tensor& a, bool periodic_scipy_style, const torch::TensorOptions& options);
    // Forward declaration for general_hamming
    torch::Tensor general_hamming_impl(int64_t M, double alpha, bool periodic_scipy_style, const torch::TensorOptions& options);


    torch::Tensor generate_torch_window(
        int64_t window_length, Type type, bool periodic_torch_style, const torch::TensorOptions& options_in
    ) {
        // Note on periodicity:
        // LibTorch 'periodic' = True for DFT-even (for spectral analysis, SciPy sym=False)
        // LibTorch 'periodic' = False for Symmetric (for filter design, SciPy sym=True)
        // So, `periodic_torch_style` == `!scipy_sym`
        // Let's define `scipy_sym_false` based on `periodic_torch_style` for clarity when using SciPy logic.
        bool scipy_sym_false = periodic_torch_style;

        // Use the provided dtype or default with device compatibility
        auto current_options = get_compatible_options(options_in);

        torch::Tensor guarded_tensor;
        if (len_guards(window_length, guarded_tensor, current_options)) {
            // For some windows, like cosine, even if M=0 or M=1, if sym=False (periodic_torch_style=true),
            // M_adj becomes M+1. If M=0, M_adj=1. Then it's truncated, resulting in empty.
            // SciPy's _len_guards returns ones(M), and _extend/_truncate handles the rest.
            // Our len_guards returns early for M=0 or M=1.
            // We need to reconcile this with the _extend/_truncate logic.
            // If M=0, len_guards returns empty. Correct.
            // If M=1, len_guards returns ones({1}).
            //    If scipy_sym_false=true (periodic): M_adj=2, needs_trunc=true.
            //       e.g. boxcar(1, sym=F) -> ones(2) -> trunc -> ones(1). Matches.
            //    If scipy_sym_false=false (symmetric): M_adj=1, needs_trunc=false.
            //       e.g. boxcar(1, sym=T) -> ones(1) -> no trunc -> ones(1). Matches.
            // So len_guards returning early for M=0, M=1 seems fine.
            return guarded_tensor;
        }

        auto extend_result = extend_truncate_params(window_length, scipy_sym_false);
        int64_t M_adj = extend_result.first;
        bool needs_trunc = extend_result.second;

        // Special handling if M_adj becomes 0 or less after extend, which can happen if window_length = 0 and scipy_sym_false = true.
        // However, len_guards already handles window_length = 0.
        // If window_length = 1 and scipy_sym_false = true, M_adj = 2.
        // If M_adj calculation results in 0 (e.g. theoretical M=-1, sym=F -> M_adj=0), return empty.
        if (M_adj <= 0 && type != Type::DPSS && type != Type::CHEBWIN) { // DPSS/CHEBWIN have specific M_adj logic
             return torch::empty({0}, current_options);
        }


        torch::Tensor w;

        switch (type) {
            // Windows directly supported by LibTorch (matching SciPy behavior for these when `periodic` is set correctly)
            case Type::HANN:
                return torch::hann_window(window_length, periodic_torch_style, current_options);
            case Type::HAMMING: // SciPy default alpha=0.54, beta=0.46
                return torch::hamming_window(window_length, periodic_torch_style, 0.54, 0.46, current_options);
            case Type::BLACKMAN:
                return torch::blackman_window(window_length, periodic_torch_style, current_options);
            case Type::BARTLETT: // SciPy's bartlett is zero at ends, matches torch::bartlett_window
                return torch::bartlett_window(window_length, periodic_torch_style, current_options);

            // Windows to implement based on SciPy
            case Type::RECTANGULAR: // Same as BOXCAR
            case Type::BOXCAR:
                // SciPy: w = xp.ones(M_adj). truncate if needed.
                // len_guards already handles M=0, M=1 for window_length.
                // If M_adj is 0 (e.g. from M=0, sym=F -> M_adj=1, then needs_trunc would make it empty if not for len_guards)
                if (M_adj == 0) return torch::empty({0}, current_options); // Should be caught by len_guards if window_length was 0.
                                                                    // This handles M_adj=0 from other hypothetical cases.
                w = torch::ones({M_adj}, current_options);
                return truncate_if_needed(w, needs_trunc);

            case Type::COSINE: // SciPy's cosine window: sin(pi * (arange(M_adj) + 0.5) / M_adj)
                               // This is distinct from torch.cosine_window
                if (M_adj == 0) return torch::empty({0}, current_options);
                // Original SciPy: w = xp.sin(xp.pi / M * (xp.arange(M) + .5))
                // M here is M_adj from our perspective.
                w = torch::sin(M_PI / M_adj * (torch::arange(0, M_adj, current_options) + 0.5));
                return truncate_if_needed(w, needs_trunc);

            case Type::TRIANG: // SciPy's triang window (not necessarily zero at ends)
                // SciPy:
                // n_tri = xp.arange(1, (M_adj + 1) // 2 + 1)
                // if M_adj % 2 == 0:  w_half = (2 * n_tri - 1.0) / M_adj; w = concat(w_half, flip(w_half))
                // else: w_half = 2 * n_tri / (M_adj + 1.0); w = concat(w_half, flip(w_half[:-1]))
                if (M_adj == 0) return torch::empty({0}, current_options);
                {
                    torch::Tensor n_tri = torch::arange(1, static_cast<int64_t>(std::floor((M_adj + 1) / 2.0)) + 1, current_options);
                    torch::Tensor w_half;
                    if (M_adj % 2 == 0) { // Even M_adj
                        w_half = (2 * n_tri - 1.0) / M_adj;
                        w = torch::cat({w_half, torch::flip(w_half, {0})});
                    } else { // Odd M_adj
                        w_half = 2 * n_tri / (M_adj + 1.0);
                        if (w_half.numel() > 1) {
                            w = torch::cat({w_half, torch::flip(w_half.slice(0, 0, w_half.numel() - 1), {0})});
                        } else { // M_adj=1 (e.g. window_length=1, sym=T or window_length=0, sym=F)
                                 // n_tri = arange(1,1+1) = [1]. w_half = 2*1/(1+1) = 1.
                            w = w_half;
                        }
                    }
                }
                return truncate_if_needed(w, needs_trunc);

            case Type::PARZEN:
                // n_parzen = xp.arange(-(M_adj - 1) / 2.0, (M_adj - 1) / 2.0 + 0.5, 1.0)
                // w1_cond = abs(n_parzen) <= (M_adj - 1) / 4.0
                // val1 = (1 - 6 * (abs(n_parzen) / (M_adj / 2.0)) ** 2.0 + 6 * (abs(n_parzen) / (M_adj / 2.0)) ** 3.0)
                // val2 = 2 * (1 - abs(n_parzen) / (M_adj / 2.0)) ** 3.0
                // w = where(w1_cond, val1, val2)
                if (M_adj == 0) return torch::empty({0}, current_options);
                {
                    torch::Tensor n_parzen = torch::arange(-(M_adj - 1) / 2.0, (M_adj - 1) / 2.0 + 0.4, 1.0, current_options); // +0.4 to ensure endpoint included like SciPy's +0.5 with float precision
                    if (n_parzen.numel() != M_adj && M_adj > 0) { // Adjust if arange endpoint behavior differs slightly
                        n_parzen = torch::linspace(-(M_adj-1)/2.0, (M_adj-1)/2.0, M_adj, current_options);
                    }

                    torch::Tensor abs_n = torch::abs(n_parzen);
                    torch::Tensor term_ratio_abs_n_div_M_half = abs_n / (M_adj / 2.0);

                    torch::Tensor val1 = 1.0 - 6.0 * torch::pow(term_ratio_abs_n_div_M_half, 2.0) + 6.0 * torch::pow(term_ratio_abs_n_div_M_half, 3.0);
                    torch::Tensor val2 = 2.0 * torch::pow(1.0 - term_ratio_abs_n_div_M_half, 3.0);

                    torch::Tensor w1_cond = abs_n <= (M_adj - 1) / 4.0;
                    w = torch::where(w1_cond, val1, val2);
                }
                return truncate_if_needed(w, needs_trunc);

            case Type::BOHMAN:
                // fac = abs(xp.linspace(-1, 1, M_adj)[1:-1])
                // w_core = (1 - fac) * xp.cos(xp.pi * fac) + 1.0 / xp.pi * xp.sin(xp.pi * fac)
                // w = xp.concat([zeros(1), w_core, zeros(1)])
                if (M_adj == 0) return torch::empty({0}, current_options);
                if (M_adj == 1) { // linspace(-1,1,1)[1:-1] is empty. SciPy results in [1.] due to _len_guards.
                    w = torch::ones({1}, current_options); // Match _len_guards behavior
                } else if (M_adj == 2) { // linspace(-1,1,2) is [-1,1]. [1:-1] is empty. SciPy uses formula directly for endpoints.
                                      // fac for M=2 is empty. Formula applies to M-2 points.
                                      // SciPy: fac = abs(linspace(-1,1,M)[1:-1]). For M=2, fac is empty. w is empty.
                                      // Then concat [0, empty, 0] -> [0,0]
                    w = torch::tensor({0.0, 0.0}, current_options);
                }
                else { // M_adj > 2
                    torch::Tensor fac_boh = torch::abs(torch::linspace(-1, 1, M_adj, current_options).slice(0, 1, M_adj - 1));
                    torch::Tensor w_core = (1.0 - fac_boh) * torch::cos(M_PI * fac_boh) + (1.0 / M_PI) * torch::sin(M_PI * fac_boh);
                    w = torch::cat({torch::zeros({1}, current_options), w_core, torch::zeros({1}, current_options)});
                }
                return truncate_if_needed(w, needs_trunc);

            case Type::NUTTALL: {
                torch::Tensor coeffs_nuttall = torch::tensor({0.3635819, 0.4891775, 0.1365995, 0.0106411}, current_options);
                return general_cosine_impl(window_length, coeffs_nuttall, scipy_sym_false, current_options);
            }
            case Type::BLACKMANHARRIS: {
                torch::Tensor coeffs_bkh = torch::tensor({0.35875, 0.48829, 0.14128, 0.01168}, current_options);
                return general_cosine_impl(window_length, coeffs_bkh, scipy_sym_false, current_options);
            }
            case Type::FLATTOP: {
                torch::Tensor coeffs_flattop = torch::tensor({0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368}, current_options);
                return general_cosine_impl(window_length, coeffs_flattop, scipy_sym_false, current_options);
            }
            case Type::BARTHANN:
                // n_bh = xp.arange(0, M_adj)
                // fac_bh = abs(n_bh / (M_adj - 1.0) - 0.5)
                // w = 0.62 - 0.48 * fac_bh + 0.38 * xp.cos(2 * xp.pi * fac_bh)
                if (M_adj == 0) return torch::empty({0}, current_options);
                if (M_adj == 1) { // Avoid division by zero if M_adj-1 is zero
                     // n=0. fac = abs(0/0 - 0.5) -> nan. SciPy _len_guards returns [1.]
                     w = torch::ones({1}, current_options);
                } else {
                    torch::Tensor n_bh = torch::arange(0, M_adj, current_options);
                    torch::Tensor fac_bh = torch::abs(n_bh / (M_adj - 1.0) - 0.5);
                    w = 0.62 - 0.48 * fac_bh + 0.38 * torch::cos(2.0 * M_PI * fac_bh);
                }
                return truncate_if_needed(w, needs_trunc);

            case Type::LANCZOS: // sinc window
                // SciPy: xpx.sinc(x) = sin(pi*x)/(pi*x)
                // torch.sinc(input) = sin(pi*input)/(pi*input) - Matches.
                // _calc_right_side_lanczos(n_start, m_val) -> sinc(2*arange(n_start,m_val)/(m_val-1) -1)
                // if M_adj % 2 == 0: (even)
                //   wh_calc = _calc_right_side_lanczos(M_adj/2, M_adj) -> sinc(2*arange(M_adj/2, M_adj)/(M_adj-1)-1)
                //   w = cat(flip(wh_calc), wh_calc)
                // else: (odd)
                //   wh_calc = _calc_right_side_lanczos((M_adj+1)/2, M_adj) -> sinc(2*arange((M_adj+1)/2, M_adj)/(M_adj-1)-1)
                //   w = cat(flip(wh_calc), ones(1), wh_calc)
                if (M_adj == 0) return torch::empty({0}, current_options);
                if (M_adj == 1) { // sinc(2*0/(1-1)-1) -> sinc(0/0-1) -> undefined. SciPy _len_guards returns [1.]
                    w = torch::ones({1}, current_options);
                } else {
                    auto calculate_sinc_arg = [&](torch::Tensor n_range) {
                        return 2.0 * n_range / (M_adj - 1.0) - 1.0;
                    };

                    if (M_adj % 2 == 0) { // Even M_adj
                        torch::Tensor n_half = torch::arange(M_adj / 2.0, M_adj, current_options);
                        torch::Tensor wh = torch::sinc(calculate_sinc_arg(n_half));
                        w = torch::cat({torch::flip(wh, {0}), wh});
                    } else { // Odd M_adj
                        // Middle point: n = (M_adj-1)/2. Arg = 2*((M_adj-1)/2)/(M_adj-1) - 1 = 1-1=0. sinc(0)=1.
                        torch::Tensor n_half_right = torch::arange(std::ceil(M_adj / 2.0), M_adj, current_options); // e.g. M_adj=5, range(2.5,5) -> [3,4]
                        torch::Tensor wh_right = torch::sinc(calculate_sinc_arg(n_half_right));

                        // For M_adj=3: n_half_right=arange(ceil(1.5),3)=[2]. arg=2*2/(2)-1=1. sinc(1)=0.
                        // wh_right=[0]. flip(wh_right)=[0]. cat([0], [1], [0]) = [0,1,0]. Correct.
                        // For M_adj=5: n_half_right=arange(ceil(2.5),5)=[3,4].
                        // arg for n=3: 2*3/(4)-1 = 1.5-1=0.5. sinc(0.5)=sin(pi/2)/(pi/2) = 1/(pi/2)=2/pi
                        // arg for n=4: 2*4/(4)-1 = 2-1=1. sinc(1)=0
                        // wh_right = [2/pi, 0]. flip = [0, 2/pi]. cat([0,2/pi], [1], [2/pi,0]). Correct.
                        w = torch::cat({torch::flip(wh_right, {0}), torch::ones({1}, current_options), wh_right});
                    }
                }
                return truncate_if_needed(w, needs_trunc);

            // Windows requiring parameters - these will throw until specific creation functions are made
            case Type::KAISER:          throw std::logic_error("KAISER window requires 'beta' parameter. Use specific creation function or overload.");
            case Type::GAUSSIAN:        throw std::logic_error("GAUSSIAN window requires 'std' parameter. Use specific creation function or overload.");
            case Type::GENERAL_COSINE:  throw std::logic_error("GENERAL_COSINE window requires 'a' (coefficients) parameter. Use specific creation function or overload.");
            case Type::GENERAL_HAMMING: throw std::logic_error("GENERAL_HAMMING window requires 'alpha' parameter. Use specific creation function or overload.");
            case Type::CHEBWIN:         throw std::logic_error("CHEBWIN window requires 'at' (attenuation) parameter. Use specific creation function or overload.");
            case Type::EXPONENTIAL:     throw std::logic_error("EXPONENTIAL window requires 'center' and 'tau' parameters. Use specific creation function or overload.");
            case Type::TUKEY:           throw std::logic_error("TUKEY window requires 'alpha' (shape) parameter. Use specific creation function or overload.");
            case Type::TAYLOR:          throw std::logic_error("TAYLOR window requires 'nbar', 'sll', 'norm' parameters. Use specific creation function or overload.");
            case Type::DPSS:            throw std::logic_error("DPSS window requires 'NW', 'Kmax' parameters. Use specific creation function or overload.");

            default:
                throw std::logic_error("Unknown or not-yet-fully-implemented window type in generate_torch_window: " + torch_window_type_to_string(type));
        }
    }

    // Implementation for general_cosine based windows (Nuttall, Blackman-Harris, Flattop, and SciPy's general_hamming)
    // SciPy's _general_cosine_impl(M, a, xp, device, sym=True)
    // sym=True (filter design) means periodic_scipy_style=false
    // sym=False (spectral analysis) means periodic_scipy_style=true
    torch::Tensor general_cosine_impl(int64_t window_length, const torch::Tensor&coeffs_a, bool periodic_scipy_style, const torch::TensorOptions& options_in) {
        const auto& current_options = get_compatible_options(options_in);
        torch::Tensor guarded_tensor;
        if (len_guards(window_length, guarded_tensor, current_options)) {
            return guarded_tensor;
        }

        auto extend_result_triang = extend_truncate_params(window_length, periodic_scipy_style);
        int64_t M_adj = extend_result_triang.first;
        bool needs_trunc = extend_result_triang.second;
        if (M_adj == 0) return torch::empty({0}, current_options);

        // fac = xp.linspace(-xp.pi, xp.pi, M_adj)
        torch::Tensor fac = torch::linspace(-M_PI, M_PI, M_adj, current_options);
        torch::Tensor w = torch::zeros({M_adj}, current_options);

        // Ensure coeffs_a is on the same device and dtype as fac and w for operations
        torch::Tensor a = coeffs_a.to(current_options.device()).to(current_options.dtype());

        for (int64_t k_idx = 0; k_idx < a.size(0); ++k_idx) {
            w += a[k_idx].item<double>() * torch::cos(static_cast<double>(k_idx) * fac);
        }
        return truncate_if_needed(w, needs_trunc);
    }

    // Implementation for SciPy's general_hamming(M, alpha, sym=True)
    // sym=True (filter design) means periodic_scipy_style=false
    // sym=False (spectral analysis) means periodic_scipy_style=true
    torch::Tensor general_hamming_impl(int64_t window_length, double alpha, bool periodic_scipy_style, const torch::TensorOptions& options_in) {
        const auto& current_options = get_compatible_options(options_in);
        // SciPy's general_hamming calls general_cosine with a = [alpha, 1.0-alpha]
        // This results in: alpha * cos(0*fac) + (1.0-alpha)*cos(1*fac) = alpha + (1.0-alpha)*cos(fac)
        // where fac = linspace(-pi, pi, M_adj)
        // This matches the formula: alpha + (1-alpha)cos(pi * x) where x is in [-1, 1]
        // The desired formula is: alpha - (1-alpha)cos(2*pi*n/(M-1))
        // Let's use the coefficients [alpha, 1.0-alpha] with general_cosine_impl to match SciPy's direct call.
        torch::Tensor coeffs_gh = torch::tensor({alpha, 1.0 - alpha}, current_options);
        return general_cosine_impl(window_length, coeffs_gh, periodic_scipy_style, current_options);
    }


    Alignment string_to_torch_window_alignment(const std::string& strAlign) {
        std::string lowerStrAlign = strAlign;
        std::transform(lowerStrAlign.begin(), lowerStrAlign.end(), lowerStrAlign.begin(), ::tolower);
        if (lowerStrAlign == "l" || lowerStrAlign == "left" || lowerStrAlign == "0") return Alignment::LEFT;
        if (lowerStrAlign == "c" || lowerStrAlign == "center" || lowerStrAlign == "1") return Alignment::CENTER;
        if (lowerStrAlign == "r" || lowerStrAlign == "right" || lowerStrAlign == "2") return Alignment::RIGHT;
        // Consider throwing an exception for unknown alignment like in string_to_torch_window_type
        // For now, defaulting to LEFT as in the original windowing.cpp for robustness in case of error.
        // TORCHWINS_POST or similar logging would be good here.
        return Alignment::LEFT;
    }

    std::string torch_window_alignment_to_string(Alignment alignment) {
        switch (alignment) {
            case Alignment::LEFT:   return "left";
            case Alignment::CENTER: return "center";
            case Alignment::RIGHT:  return "right";
            default:                return "unknown_alignment";
        }
    }

    torch::Tensor generate_torch_window_aligned(
        int64_t window_length,
        Type type,
        bool periodic,
        int64_t zero_padding_samples,
        Alignment alignment,
        const torch::TensorOptions& options
    ) {
        int64_t actual_signal_samples = window_length - zero_padding_samples;

        if (actual_signal_samples <= 0) {
            return torch::zeros({window_length}, options);
        }

        torch::Tensor base_window = generate_torch_window(actual_signal_samples, type, periodic, options);

        torch::Tensor output_window = torch::zeros({window_length}, options);

        int64_t first_sample_index = 0;
        switch (alignment) {
            case Alignment::LEFT:
                first_sample_index = 0;
                break;
            case Alignment::CENTER:
                first_sample_index = zero_padding_samples / 2;
                break;
            case Alignment::RIGHT:
                first_sample_index = zero_padding_samples;
                break;
        }

        // Ensure the slice does not go out of bounds
        if (first_sample_index + actual_signal_samples > window_length) {
            // This case should ideally not be reached if inputs are validated,
            // for example, if zero_padding_samples is negative or too large.
            // Handling defensively by adjusting or throwing an error.
            // For now, let's assume valid inputs leading to this point.
            // If actual_signal_samples is positive, first_sample_index must be < window_length.
            // And first_sample_index + actual_signal_samples must be <= window_length.
            // The current logic for actual_signal_samples and first_sample_index should prevent this,
            // but a robust implementation might add explicit checks or an assertion.
        }

        output_window.slice(0, first_sample_index, first_sample_index + actual_signal_samples) = base_window;

        return output_window;
    }

    // converts a string representation of a Torch window type to its corresponding Type enum
    Type string_to_torch_window_type(const std::string& strType) {
        std::string lowerStrType = strType;
        std::transform(lowerStrType.begin(), lowerStrType.end(), lowerStrType.begin(), ::tolower);
        
        if (lowerStrType == "hann" || lowerStrType == "hanning") return Type::HANN;
        if (lowerStrType == "hamming") return Type::HAMMING;
        if (lowerStrType == "blackman") return Type::BLACKMAN;
        if (lowerStrType == "bartlett") return Type::BARTLETT; // SciPy bartlett is zero-ended
        if (lowerStrType == "rectangular" || lowerStrType == "rect" || lowerStrType == "boxcar") return Type::BOXCAR; // BOXCAR is alias for RECTANGULAR
        if (lowerStrType == "cosine") return Type::COSINE; // SciPy's specific cosine (sine shape)
        if (lowerStrType == "triang" || lowerStrType == "triangular") return Type::TRIANG; // SciPy triang (non-zero ends generally)
        if (lowerStrType == "parzen") return Type::PARZEN;
        if (lowerStrType == "bohman") return Type::BOHMAN;
        if (lowerStrType == "nuttall") return Type::NUTTALL;
        if (lowerStrType == "blackmanharris" || lowerStrType == "blackharr") return Type::BLACKMANHARRIS;
        if (lowerStrType == "flattop") return Type::FLATTOP;
        if (lowerStrType == "barthann") return Type::BARTHANN;
        if (lowerStrType == "kaiser") return Type::KAISER;
        if (lowerStrType == "gaussian" || lowerStrType == "gauss") return Type::GAUSSIAN;
        if (lowerStrType == "general_cosine" || lowerStrType == "generalcosine") return Type::GENERAL_COSINE;
        if (lowerStrType == "general_hamming" || lowerStrType == "generalhamming") return Type::GENERAL_HAMMING;
        if (lowerStrType == "chebwin" || lowerStrType == "chebyshev") return Type::CHEBWIN;
        if (lowerStrType == "exponential" || lowerStrType == "poisson") return Type::EXPONENTIAL;
        if (lowerStrType == "tukey") return Type::TUKEY;
        if (lowerStrType == "taylor") return Type::TAYLOR;
        if (lowerStrType == "dpss" || lowerStrType == "slepian") return Type::DPSS;
        if (lowerStrType == "lanczos" || lowerStrType == "sinc") return Type::LANCZOS;
        // Note: RECTANGULAR enum val could be removed if BOXCAR is preferred, and handled above.
        // For now, allow "rectangular" to map to BOXCAR as they are functionally identical here.
        
        throw std::invalid_argument("Unknown torch window type string: " + strType);
    }

    // gets the string representation of a Torch window type
    std::string torch_window_type_to_string(Type type) {
        switch (type) {
            case Type::HANN: return "Hann";
            case Type::HAMMING: return "Hamming";
            case Type::BLACKMAN: return "Blackman";
            case Type::BARTLETT: return "Bartlett"; // Was "Bartlett (Triangular)", but Triang is now separate
            case Type::RECTANGULAR: return "Rectangular"; // Kept for existing usage, maps to Boxcar logic
            case Type::COSINE: return "Cosine (SciPy style)";
            case Type::BOXCAR: return "Boxcar";
            case Type::TRIANG: return "Triang";
            case Type::PARZEN: return "Parzen";
            case Type::BOHMAN: return "Bohman";
            case Type::NUTTALL: return "Nuttall";
            case Type::BLACKMANHARRIS: return "Blackman-Harris";
            case Type::FLATTOP: return "Flattop";
            case Type::BARTHANN: return "Barthann";
            case Type::KAISER: return "Kaiser";
            case Type::GAUSSIAN: return "Gaussian";
            case Type::GENERAL_COSINE: return "General Cosine";
            case Type::GENERAL_HAMMING: return "General Hamming";
            case Type::CHEBWIN: return "Chebyshev";
            case Type::EXPONENTIAL: return "Exponential";
            case Type::TUKEY: return "Tukey";
            case Type::TAYLOR: return "Taylor";
            case Type::DPSS: return "DPSS (Slepian)";
            case Type::LANCZOS: return "Lanczos (Sinc)";
            default: return "Unknown"; // Should not happen if Type enum is used correctly
        }
    }

// Bessel function i0 approximation for Kaiser window
// Using Abramowitz and Stegun approximation (9.8.1), good for beta > ~2
// For smaller beta, taylor series expansion is better, but kaiser usually uses larger beta.
// For beta = 0, i0(0) = 1.
// This approximation is what NumPy uses for float32, and is often sufficient.
// For higher precision, one might need a more complex series or different algorithm.
torch::Tensor i0_approx(const torch::Tensor& x_in) {
    // Ensure input is float64 for precision, and on the same device
    torch::Tensor x = x_in.to(torch::kFloat64);
    torch::Tensor abs_x = torch::abs(x);

    // Coefficients for the approximation
    // From a common implementation, e.g., NumPy's older versions or public domain sources
    // P = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.0360768, 0.0045813]
    // Q = [0.39894228, 0.01328592, 0.00225319, -0.00157565, 0.00916281, -0.002057706, 0.002635537] // These are for I0(x)/sqrt(x) like forms or different approx.

    // Simpler polynomial approximation for I0, often used:
    // For |x| <= 3.75
    // y = (x/3.75)^2
    // I0 = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    // For |x| > 3.75
    // ax = abs(x)
    // I0 = (exp(ax) / sqrt(ax)) * (0.39894228 + 0.01328592/ax + 0.00225319/(ax^2) - ...)
    // Let's use the one from SciPy's special.i0 which is likely more robust or calls underlying C/Fortran.
    // Since we don't have `special.i0` directly, we need an implementation.
    // A common approach for `i0e` (exp(-abs(x)) * i0(x)) is used in some libraries to avoid overflow.
    // Then i0(x) = exp(abs(x)) * i0e(x).
    //
    // SciPy's `special.i0` is a wrapper around `amos::zbesi` or similar.
    // Let's use a polynomial approximation that is common for i0.
    // This specific one is from a widely cited source (e.g., Numerical Recipes, though coefficients vary slightly).
    // This version is for i0, not i0e.
    auto poly_approx = [](const torch::Tensor& val_y) {
        return 1.0 + val_y * (3.5156229 + val_y * (3.0899424 + val_y * (1.2067492 + val_y * (0.2659732 + val_y * (0.0360768 + val_y * 0.0045813)))));
    };

    auto ratio_approx = [](const torch::Tensor& ax) {
        return (torch::exp(ax) / torch::sqrt(ax)) * (0.39894228 + (0.01328592 / ax) + (0.00225319 / torch::pow(ax, 2))
                     - (0.00157565 / torch::pow(ax, 3)) + (0.00916281 / torch::pow(ax, 4))
                     - (0.02057706 / torch::pow(ax, 5)) + (0.02635537 / torch::pow(ax, 6))
                     - (0.01647633 / torch::pow(ax, 7)) + (0.00392377 / torch::pow(ax, 8)));
    };

    torch::Tensor y = torch::pow(x / 3.75, 2);
    torch::Tensor result_small = poly_approx(y);
    torch::Tensor result_large = ratio_approx(abs_x);

    // Create a mask for small and large values
    torch::Tensor small_mask = abs_x <= 3.75;
    torch::Tensor large_mask = abs_x > 3.75;

    // Initialize result tensor
    torch::Tensor result = torch::zeros_like(x);
    result.masked_scatter_(small_mask, result_small.masked_select(small_mask));
    result.masked_scatter_(large_mask, result_large.masked_select(large_mask));

    // Handle x == 0 separately as ratio_approx would divide by zero. i0(0) = 1.
    result.masked_fill_(x == 0, 1.0);

    return result;
}


// Specific window creation function implementations

torch::Tensor create_hann_window(int64_t window_length, bool periodic, const torch::TensorOptions& options) {
    return generate_torch_window(window_length, Type::HANN, periodic, options);
}

torch::Tensor create_hamming_window(int64_t window_length, bool periodic, const torch::TensorOptions& options, double alpha, double beta) {
    // LibTorch's hamming_window takes full alpha and beta.
    // Standard Hamming is alpha=0.54, beta=0.46.
    // If one wants to use the generate_torch_window with default Hamming, it's already set.
    // This function allows specifying alpha and beta if needed, otherwise uses default libtorch ones.
    // SciPy's `hamming` calls `general_hamming(M, 0.54, sym)`
    // SciPy's `general_hamming(M, alpha, sym)` calls `_general_cosine_impl(M, [alpha, 1-alpha], sym)`
    // This means the `beta` parameter in `torch::hamming_window` is `1-alpha`.
    // So, torch::hamming_window(len, per, alpha, 1-alpha, opts) should match SciPy's general_hamming(len, alpha, !per)
    // The provided alpha and beta are for torch::hamming_window's direct definition: alpha - beta * cos(...)
    // If the user calls this with default alpha=0.54, beta=0.46, it matches `generate_torch_window` for Type::HAMMING.
    return torch::hamming_window(window_length, periodic, alpha, beta, get_compatible_options(options));
}

torch::Tensor create_blackman_window(int64_t window_length, bool periodic, const torch::TensorOptions& options) {
    return generate_torch_window(window_length, Type::BLACKMAN, periodic, options);
}

torch::Tensor create_bartlett_window(int64_t window_length, bool periodic, const torch::TensorOptions& options) {
    return generate_torch_window(window_length, Type::BARTLETT, periodic, options);
}

torch::Tensor create_cosine_window(int64_t window_length, bool periodic, const torch::TensorOptions& options) {
    return generate_torch_window(window_length, Type::COSINE, periodic, options);
}

torch::Tensor create_rectangular_window(int64_t window_length, const torch::TensorOptions& options) {
    // For rectangular, periodic flag has no effect on torch::ones, but send false for consistency with SciPy sym=True default
    return generate_torch_window(window_length, Type::RECTANGULAR, false, options);
}

// New SciPy window types
torch::Tensor create_boxcar_window(int64_t window_length, bool periodic, const torch::TensorOptions& options) {
    return generate_torch_window(window_length, Type::BOXCAR, periodic, options);
}

torch::Tensor create_triang_window(int64_t window_length, bool periodic, const torch::TensorOptions& options) {
    return generate_torch_window(window_length, Type::TRIANG, periodic, options);
}

torch::Tensor create_parzen_window(int64_t window_length, bool periodic, const torch::TensorOptions& options) {
    return generate_torch_window(window_length, Type::PARZEN, periodic, options);
}

torch::Tensor create_bohman_window(int64_t window_length, bool periodic, const torch::TensorOptions& options) {
    return generate_torch_window(window_length, Type::BOHMAN, periodic, options);
}

torch::Tensor create_nuttall_window(int64_t window_length, bool periodic, const torch::TensorOptions& options) {
    return generate_torch_window(window_length, Type::NUTTALL, periodic, options);
}

torch::Tensor create_blackmanharris_window(int64_t window_length, bool periodic, const torch::TensorOptions& options) {
    return generate_torch_window(window_length, Type::BLACKMANHARRIS, periodic, options);
}

torch::Tensor create_flattop_window(int64_t window_length, bool periodic, const torch::TensorOptions& options) {
    return generate_torch_window(window_length, Type::FLATTOP, periodic, options);
}

torch::Tensor create_barthann_window(int64_t window_length, bool periodic, const torch::TensorOptions& options) {
    return generate_torch_window(window_length, Type::BARTHANN, periodic, options);
}

torch::Tensor create_lanczos_window(int64_t window_length, bool periodic, const torch::TensorOptions& options) {
    return generate_torch_window(window_length, Type::LANCZOS, periodic, options);
}

// Implementations for windows requiring specific parameters

torch::Tensor create_kaiser_window(int64_t window_length, double beta, bool periodic, const torch::TensorOptions& options_in) {
    // SciPy: kaiser(M, beta, sym=True)
    // w = i0(beta * sqrt(1 - ((n - alpha_k) / alpha_k)^2)) / i0(beta)
    // n = arange(0, M_adj)
    // alpha_k = (M_adj - 1) / 2.0
    // periodic_torch_style == !scipy_sym
    bool scipy_sym_false = periodic;
    const auto& current_options = get_compatible_options(options_in);

    torch::Tensor guarded_tensor;
    if (len_guards(window_length, guarded_tensor, current_options)) {
        return guarded_tensor; // Returns empty for M=0, ones({1}) for M=1
    }

    auto extend_result_parzen = extend_truncate_params(window_length, scipy_sym_false);
    int64_t M_adj = extend_result_parzen.first;
    bool needs_trunc = extend_result_parzen.second;
    if (M_adj == 0) return torch::empty({0}, current_options);

    torch::Tensor n = torch::arange(0, M_adj, current_options);
    double alpha_k = (M_adj - 1.0) / 2.0;

    torch::Tensor term = (n - alpha_k) / alpha_k;
    torch::Tensor sqrt_term = torch::sqrt(1.0 - torch::pow(term, 2.0));
    // Handle potential NaNs from sqrt if term^2 > 1 due to floating point issues at edges, though mathematically should be <=1.
    // SciPy's i0 handles NaNs in input by returning NaN. We should aim for similar.
    // If 1 - term^2 is negative, sqrt will be NaN.
    sqrt_term.masked_fill_((1.0 - torch::pow(term, 2.0)) < 0, std::numeric_limits<double>::quiet_NaN());


    torch::Tensor beta_tensor = torch::tensor(beta, current_options);
    torch::Tensor num = i0_approx(beta_tensor * sqrt_term);
    torch::Tensor den = i0_approx(beta_tensor);

    torch::Tensor w = num / den;

    // If beta is 0, i0(0)=1, so w should be all 1s.
    // Our i0_approx(0) returns 1. So num/den = 1/1 = 1. This is correct.
    // If any part of sqrt_term is NaN, num will be NaN, w will be NaN.

    return truncate_if_needed(w, needs_trunc);
}

torch::Tensor create_gaussian_window(int64_t window_length, double std_dev, bool periodic, const torch::TensorOptions& options_in) {
    // SciPy: gaussian(M, std, sym=True)
    // n = arange(0, M_adj) - (M_adj - 1.0) / 2.0
    // sig2 = 2 * std * std
    // w = exp(-n^2 / sig2)
    bool scipy_sym_false = periodic;
    const auto& current_options = get_compatible_options(options_in);

    torch::Tensor guarded_tensor;
    if (len_guards(window_length, guarded_tensor, current_options)) {
        return guarded_tensor;
    }

    auto extend_result = extend_truncate_params(window_length, scipy_sym_false);
    int64_t M_adj = extend_result.first;
    bool needs_trunc = extend_result.second;
    if (M_adj == 0) return torch::empty({0}, current_options);

    torch::Tensor n = torch::arange(0, M_adj, current_options) - (M_adj - 1.0) / 2.0;
    double sig2 = 2.0 * std_dev * std_dev;
    if (sig2 == 0) { // Avoid division by zero; if std_dev is 0, it's a dirac delta (or 1 at center, 0 elsewhere for discrete)
        torch::Tensor w = torch::zeros({M_adj}, current_options);
        if (M_adj % 2 == 1) { // Odd length, single center point
            w[M_adj / 2] = 1.0;
        } else { // Even length, two center points get 0.5 if we want sum to 1, or handle as per SciPy (which yields all zeros if std=0 due to exp(-inf))
                 // SciPy with std=0 gives exp(-n^2/0). For n=0, exp(nan)=nan. For n!=0, exp(-inf)=0.
                 // Let's return mostly zeros, with 1 at center if M_adj is odd.
                 // If std_dev is very small but non-zero, it will be a narrow spike.
                 // A true std_dev=0 results in NaN from division by zero in exp if not handled.
                 // For simplicity, if std_dev is 0, make it a 1 at the center for odd M_adj, else all zeros.
                 // This matches some interpretations of Gaussian with sigma=0.
        }
        return truncate_if_needed(w, needs_trunc);
    }
    torch::Tensor w = torch::exp(-torch::pow(n, 2.0) / sig2);
    return truncate_if_needed(w, needs_trunc);
}

torch::Tensor create_general_cosine_window(int64_t window_length, const torch::Tensor& coefficients, bool periodic, const torch::TensorOptions& options) {
    bool scipy_sym_false = periodic;
    return general_cosine_impl(window_length, coefficients, scipy_sym_false, options);
}

torch::Tensor create_general_hamming_window(int64_t window_length, double alpha, bool periodic, const torch::TensorOptions& options) {
    bool scipy_sym_false = periodic;
    // This directly calls the internal general_hamming_impl which uses general_cosine_impl
    // with coefficients [alpha, 1.0-alpha] to match SciPy's general_hamming behavior.
    return general_hamming_impl(window_length, alpha, scipy_sym_false, options);
}

torch::Tensor create_chebwin_window(int64_t window_length, double attenuation, bool periodic, const torch::TensorOptions& options_in) {
    // SciPy: chebwin(M, at, sym=True)
    // This is complex, involves Chebyshev polynomials and IFFT.
    // For now, placeholder. A proper implementation is non-trivial.
    // Requires: acosh, cosh, fft.
    // Note: LibTorch has torch.fft.fft and torch.fft.ifft
    bool scipy_sym_false = periodic;
    const auto& current_options = get_compatible_options(options_in); // Output should be compatible with device

    torch::Tensor guarded_tensor;
    if (len_guards(window_length, guarded_tensor, current_options)) {
        return guarded_tensor;
    }
     auto extend_result = extend_truncate_params(window_length, scipy_sym_false);
     int64_t M_adj = extend_result.first;
     bool needs_trunc = extend_result.second;
    if (M_adj == 0) return torch::empty({0}, current_options);

    // Parameter beta computation
    double order = static_cast<double>(M_adj) - 1.0;
    torch::Tensor val_10_at_20 = torch::tensor(std::pow(10.0, std::abs(attenuation) / 20.0), current_options);
    torch::Tensor beta = torch::acosh(val_10_at_20) / order; // This is 1/order * acosh(...)
    beta = torch::cosh(beta); // beta = cosh(1/order * acosh(10^(at/20)))

    torch::Tensor k = torch::arange(0, M_adj, current_options);
    torch::Tensor x = beta * torch::cos(M_PI * k / M_adj);

    // Chebyshev polynomial calculation T_order(x)
    // T_n(x) = cos(n * acos(x)) for |x| <= 1
    // T_n(x) = cosh(n * acosh(x)) for x > 1
    // T_n(x) = (-1)^n * cosh(n * acosh(-x)) for x < -1
    torch::Tensor p = torch::zeros({M_adj}, current_options);

    torch::Tensor x_gt_1 = x > 1.0;
    torch::Tensor x_lt_n1 = x < -1.0;
    torch::Tensor x_abs_le_1 = torch::abs(x) <= 1.0;

    if (x_gt_1.any().item<bool>()) {
        p.masked_scatter_(x_gt_1, torch::cosh(order * torch::acosh(x.masked_select(x_gt_1))));
    }
    if (x_lt_n1.any().item<bool>()) {
        double sign = (static_cast<int64_t>(order) % 2 == 0) ? 1.0 : -1.0; // (-1)^order approx. M_adj is int.
        if ( (M_adj % 2) != 0 ) sign = sign * -1; // SciPy (2*(M%2)-1) factor for x < -1. M is M_adj here.
                                                  // (2*(M_adj % 2) -1) -> if M_adj even, -1. if M_adj odd, 1.
                                                  // My 'sign' is for T_order. The scipy one is an additional mult.
        double scipy_sign_factor = (M_adj % 2 == 0) ? -1.0 : 1.0;


        p.masked_scatter_(x_lt_n1, scipy_sign_factor * torch::cosh(order * torch::acosh(-x.masked_select(x_lt_n1))));
    }
    if (x_abs_le_1.any().item<bool>()) {
        p.masked_scatter_(x_abs_le_1, torch::cos(order * torch::acos(x.masked_select(x_abs_le_1))));
    }

    // IFFT
    // SciPy uses fft.fft (for DFT coeffs) then takes real part of fft (IDFT)
    // The 'p' array is W(k) in freq domain. We need IFFT to get time domain window.
    // SciPy: if M % 2: w = real(fft(p)); ... else: p_phased = p * exp(j*pi/M * arange(M)); w = real(fft(p_phased))
    // LibTorch's ifft is for complex inputs. p is real.
    // We need a Type II DCT-like transform or careful use of IFFT on a symmetrically extended p.
    // Or, if p represents DFT coefficients directly (real and even for real symmetric time signal),
    // then irfft can be used.
    // SciPy's example uses `fft` on `p` which are the Chebyshev polynomial values, not directly DFT magnitudes.
    // This part is tricky and needs careful mapping of SciPy's FFT behavior for this specific algorithm.
    // The "fft" in scipy.signal.chebwin's source is `sp_fft.fft`
    // Let's assume p are coefficients that need an IDFT.
    // For real symmetric window, DFT coefficients are real.
    // `w = real(sp_fft.fft(p))` -> This is unusual. Typically IFFT.
    // It might be that p is already structured for FFT to act as IFFT.

    // Given the complexity and potential for subtle errors in FFT usage here without deeper analysis
    // of SciPy's `chebwin`'s FFT stage, this will be a rough port.
    torch::Tensor w_time;
    if (M_adj % 2 != 0) { // Odd length
        // p needs to be treated as complex for c2c fft
        w_time = torch::real(torch::fft::fft(p.to(torch::kComplexDouble))); // Real part of FFT
        int64_t n_half = (M_adj + 1) / 2;
        torch::Tensor w_first_half = w_time.slice(0, 0, n_half);
        w_time = torch::cat({torch::flip(w_first_half.slice(0,1,n_half), {0}), w_first_half});
    } else { // Even length
        torch::Tensor phase = M_PI / M_adj * torch::arange(0, M_adj, current_options.dtype(torch::kComplexDouble));
        torch::Tensor i_phase = torch::complex(torch::zeros_like(phase), phase); // Create complex tensor 0 + i*phase
        torch::Tensor p_phased = p.to(torch::kComplexDouble) * torch::exp(i_phase);
        w_time = torch::real(torch::fft::fft(p_phased));
        int64_t n_half = M_adj / 2 + 1;
        // SciPy: w = concat(flip(w[1:n]), w[1:n])
        // This means taking elements from index 1 up to n_half-1
        torch::Tensor w_slice = w_time.slice(0, 1, n_half);
        w_time = torch::cat({torch::flip(w_slice, {0}), w_slice});
    }

    w_time = w_time / torch::max(w_time); // Normalize
    return truncate_if_needed(w_time, needs_trunc);
}


torch::Tensor create_exponential_window(int64_t window_length, bool periodic, const torch::TensorOptions& options_in, c10::optional<double> center_opt, double tau) {
    // SciPy: exponential(M, center=None, tau=1., sym=True)
    // if sym and center is not None: error
    // if center is None: center = (M_adj - 1) / 2
    // n = arange(0, M_adj)
    // w = exp(-abs(n - center) / tau)
    bool scipy_sym_false = periodic;
    const auto& current_options = get_compatible_options(options_in);

    if (!scipy_sym_false && center_opt.has_value()) { // sym=True case, center must be None
        throw std::invalid_argument("For symmetric exponential window, center must not be specified (it's auto-calculated).");
    }

    torch::Tensor guarded_tensor;
    if (len_guards(window_length, guarded_tensor, current_options)) {
        return guarded_tensor;
    }
    auto extend_result = extend_truncate_params(window_length, scipy_sym_false);
    int64_t M_adj = extend_result.first;
    bool needs_trunc = extend_result.second;
    if (M_adj == 0) return torch::empty({0}, current_options);

    double center_val;
    if (center_opt.has_value()) {
        center_val = center_opt.value();
    } else {
        center_val = (M_adj - 1.0) / 2.0;
    }

    torch::Tensor n = torch::arange(0, M_adj, current_options);
    if (tau == 0) { // Avoid division by zero, treat as delta function or all zeros
        torch::Tensor w = torch::zeros({M_adj}, current_options);
        // A single point at 'center_val' if it's an integer index, otherwise complex.
        // SciPy with tau=0 gives exp(-inf) = 0 for n!=center, exp(nan) for n=center.
        // For simplicity, return zeros if tau is zero.
        return truncate_if_needed(w, needs_trunc);
    }
    torch::Tensor w = torch::exp(-torch::abs(n - center_val) / tau);
    return truncate_if_needed(w, needs_trunc);
}

torch::Tensor create_tukey_window(int64_t window_length, double alpha_shape, bool periodic, const torch::TensorOptions& options_in) {
    // SciPy: tukey(M, alpha=0.5, sym=True)
    // if alpha <=0: return ones
    // if alpha >=1: return hann
    // n = arange(0, M_adj)
    // width = floor(alpha_shape * (M_adj - 1) / 2.0)
    // n1 = n[0 : width+1]
    // n2 = n[width+1 : M_adj-width-1]
    // n3 = n[M_adj-width-1 :]
    // w1 = 0.5 * (1 + cos(pi * (-1 + 2*n1 / (alpha_shape*(M_adj-1)))))
    // w2 = ones(n2.shape)
    // w3 = 0.5 * (1 + cos(pi * (-2/alpha_shape + 1 + 2*n3 / (alpha_shape*(M_adj-1)))))
    // w = concat(w1,w2,w3)
    bool scipy_sym_false = periodic;
    const auto& current_options = get_compatible_options(options_in);

    torch::Tensor guarded_tensor;
    if (len_guards(window_length, guarded_tensor, current_options)) {
        // If M=0 or M=1, SciPy returns ones. This matches len_guards.
        // Tukey specific alpha checks happen after this.
        return guarded_tensor;
    }

    if (alpha_shape <= 0) { // Return rectangular window
        return create_boxcar_window(window_length, periodic, options_in);
    }
    if (alpha_shape >= 1.0) { // Return Hann window
        return create_hann_window(window_length, periodic, options_in);
    }

    auto extend_result = extend_truncate_params(window_length, scipy_sym_false);
    int64_t M_adj = extend_result.first;
    bool needs_trunc = extend_result.second;
    if (M_adj == 0) return torch::empty({0}, current_options);
    if (M_adj == 1) return torch::ones({1}, current_options); // Tukey with M_adj=1 should be 1.

    torch::Tensor n = torch::arange(0, M_adj, current_options);
    int64_t width = static_cast<int64_t>(std::floor(alpha_shape * (M_adj - 1) / 2.0));

    // Ensure width is not negative if M_adj=1 (alpha_shape*(0)/2 = 0)
    if (width < 0) width = 0;


    torch::Tensor n1 = n.slice(0, 0, width + 1);
    torch::Tensor n2 = n.slice(0, width + 1, M_adj - width - 1);
    torch::Tensor n3 = n.slice(0, M_adj - width - 1, M_adj);

    torch::Tensor w1, w2, w3;

    if (n1.numel() > 0) {
         if (alpha_shape * (M_adj - 1) == 0 && M_adj > 1) { // Avoid div by zero if alpha_shape is small leading to alpha_shape*(M_adj-1) = 0
            w1 = torch::ones_like(n1); // Or some other appropriate value, cosine arg becomes problematic
        } else if (M_adj == 1) { // alpha_shape*(M_adj-1) is 0
             w1 = torch::ones_like(n1); // Should be single point 1.
        }
         else {
            w1 = 0.5 * (1.0 + torch::cos(M_PI * (-1.0 + 2.0 * n1 / (alpha_shape * (M_adj - 1.0)))));
        }
    } else {
        w1 = torch::empty({0}, current_options);
    }

    if (n2.numel() > 0) {
        w2 = torch::ones_like(n2);
    } else {
        w2 = torch::empty({0}, current_options);
    }

    if (n3.numel() > 0) {
         if (alpha_shape * (M_adj-1) == 0 && M_adj > 1) {
            w3 = torch::ones_like(n3);
        } else if (M_adj == 1) {
            w3 = torch::ones_like(n3);
        }
        else {
            w3 = 0.5 * (1.0 + torch::cos(M_PI * (-2.0 / alpha_shape + 1.0 + 2.0 * n3 / (alpha_shape * (M_adj - 1.0)))));
        }
    } else {
        w3 = torch::empty({0}, current_options);
    }

    torch::Tensor w = torch::cat({w1, w2, w3});
    return truncate_if_needed(w, needs_trunc);
}


torch::Tensor create_taylor_window(int64_t window_length, int nbar, double sll, bool norm, bool periodic, const torch::TensorOptions& options_in) {
    // SciPy: taylor(M, nbar=4, sll=30, norm=True, sym=True)
    // B = 10**(sll/20)
    // A = acosh(B)/pi
    // s2 = nbar^2 / (A^2 + (nbar-0.5)^2)
    // ma = arange(1, nbar)
    // Fm = zeros(nbar-1) ... loop to calculate Fm
    // W_func = lambda n_idx: 1 + 2*sum(Fm * cos(2*pi*ma*(n_idx-M_adj/2+0.5)/M_adj))
    // w = W_func(arange(M_adj))
    // if norm: scale = 1.0 / W_func((M_adj-1)/2); w *= scale
    bool scipy_sym_false = periodic;
    const auto& current_options = get_compatible_options(options_in);

    torch::Tensor guarded_tensor;
    if (len_guards(window_length, guarded_tensor, current_options)) {
        return guarded_tensor;
    }
    auto extend_result = extend_truncate_params(window_length, scipy_sym_false);
    int64_t M_adj = extend_result.first;
    bool needs_trunc = extend_result.second;
    if (M_adj == 0) return torch::empty({0}, current_options);

    double B_taylor = std::pow(10.0, sll / 20.0);
    double A_taylor = std::acosh(B_taylor) / M_PI;
    double s2 = static_cast<double>(nbar * nbar) / (A_taylor * A_taylor + std::pow(nbar - 0.5, 2.0));

    torch::Tensor ma = torch::arange(1, nbar, current_options);
    torch::Tensor Fm = torch::zeros({nbar - 1}, current_options);

    for (int64_t mi = 0; mi < ma.size(0); ++mi) {
        double m_val = ma[mi].item<double>();
        double m2_mi = m_val * m_val;

        double numer_prod = 1.0;
        for (int64_t ma_idx = 0; ma_idx < ma.size(0); ++ma_idx) {
            numer_prod *= (1.0 - m2_mi / s2 / (A_taylor * A_taylor + torch::pow(ma[ma_idx] - 0.5, 2.0).item<double>()));
        }
        // The SciPy line `xp.prod(1 - m2[mi]/s2/(A**2 + (ma - 0.5)**2))` is confusing.
        // It seems to imply a product over all `ma` for a single `m2[mi]`.
        // Let's assume it's product over (1 - m_i^2 / (s^2 * (A^2 + (m_k - 0.5)^2))) for k != i
        // No, the formula is F_m = [ (-1)^(m-1) \prod_{k=1, k!=m}^{nbar-1} (1 - m^2/z_k^2) ] / [ 2 \prod_{k=1, k!=m}^{nbar-1} (1 - m^2/k^2) ]
        // where z_k^2 = s2 * (A^2 + (k-0.5)^2). This seems more standard for Taylor.
        // The SciPy code is `numer = signs[mi] * xp.prod(1 - m2[mi]/s2/(A**2 + (ma - 0.5)**2))`
        // This `ma` in `(A**2 + (ma - 0.5)**2)` means it's a product over an array resulting from an array operation.
        // This is likely `(1 - m_i^2 / (s2 * (A^2 + (m_k - 0.5)^2)))` for each k in ma.
        // And then a product of these terms.

        // Recalculating numerator based on typical Taylor window formula structure:
        // numer_prod part: product over k from 1 to nbar-1, where k != m
        // (1 - m^2 / (s2 * (A^2 + (k-0.5)^2)))
        // SciPy's code seems to be: Product over k of (1 - m_i^2 / (s2 * (A^2 + (ma_k - 0.5)^2)) )
        // This means for a given m_i (m_val), we iterate through all ma_k.
        double term_prod_val = 1.0;
        for(int k_inner = 0; k_inner < ma.numel(); ++k_inner) {
            double ma_k_val = ma[k_inner].item<double>();
            term_prod_val *= (1.0 - (m2_mi / (s2 * (A_taylor*A_taylor + std::pow(ma_k_val - 0.5, 2.0)) ) ) );
        }
        double numer = ((static_cast<int64_t>(m_val) -1 ) % 2 == 0 ? 1.0 : -1.0) * term_prod_val;
        // The above signs[mi] is (-1)^(m-1). If m starts at 1, then signs[0] for m=1 is (-1)^0=1. Correct.

        double denom_prod_val = 1.0;
         for(int k_inner = 0; k_inner < ma.numel(); ++k_inner) {
            if (k_inner == mi) continue;
            denom_prod_val *= (1.0 - m2_mi / ma[k_inner].item<double>()/ma[k_inner].item<double>() );
         }
        double denom = 2.0 * denom_prod_val;
        Fm[mi] = numer / denom;
    }

    auto W_func = [&](const torch::Tensor& n_indices) {
        torch::Tensor sum_terms = torch::zeros_like(n_indices, current_options);
        for (int64_t i = 0; i < Fm.size(0); ++i) {
            sum_terms += Fm[i] * torch::cos(2.0 * M_PI * ma[i] * (n_indices - M_adj / 2.0 + 0.5) / M_adj);
        }
        return 1.0 + 2.0 * sum_terms;
    };

    torch::Tensor w = W_func(torch::arange(0, M_adj, current_options));

    if (norm) {
        double scale_val = W_func(torch::tensor({(M_adj - 1.0) / 2.0}, current_options)).item<double>();
        if (scale_val != 0) {
            w = w / scale_val;
        }
    }
    return truncate_if_needed(w, needs_trunc);
}

torch::Tensor create_dpss_window(int64_t window_length, double nw_param, int kmax, bool periodic, const torch::TensorOptions& options) {
    // SciPy: dpss(M, NW, Kmax=None, sym=True, norm=None, return_ratios=False)
    // This is highly complex, involving eigenvalue decomposition of a tridiagonal matrix.
    // A full port is beyond a quick step. Placeholder.
    // For a single window (Kmax=1, which is typical for get_window), it's the first Slepian sequence.
    throw std::logic_error("DPSS window is highly complex and not yet implemented. Requires eigenvalue solver.");
    // return torch::ones({window_length}, options.dtype(torch::kFloat64)); // Placeholder if error is not desired
}

        } // namespace util_windowing
    } // namespace core
} // namespace contorchionist

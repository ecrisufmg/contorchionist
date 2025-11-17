#ifndef CORE_AP_RFFT_H
#define CORE_AP_RFFT_H

#include <torch/torch.h>
#include "core_util_windowing.h"
#include <string>
#include <vector>
#include <stdexcept>
#include "core_util_conversions.h" // For SpectrumDataFormat
#include "core_util_normalizations.h" // Unified normalization logic

namespace contorchionist {
    namespace core {
        namespace ap_rfft {

// enum class for FFT modes: 1 for standard FFT, 2 for RFFT
enum class FFTMode {
    FFT = 1,
    RFFT = 2
};

template<typename T>
class RFFTProcessor {
public:
    RFFTProcessor(torch::Device device = torch::kCPU, 
                  contorchionist::core::util_windowing::Type window_type = contorchionist::core::util_windowing::Type::RECTANGULAR, 
                  bool windowing_enabled = false,
                  bool verbose = false,
                  core::util_normalizations::NormalizationType norm_type = core::util_normalizations::NormalizationType::NONE,
                  contorchionist::core::util_conversions::SpectrumDataFormat output_format = contorchionist::core::util_conversions::SpectrumDataFormat::COMPLEX,
                  T fs = 48000.0f);
    ~RFFTProcessor();

    // Device Management
    void set_device(torch::Device device);
    torch::Device get_device() const;

    // Verbosity Management
    void set_verbose(bool verbose);
    bool is_verbose() const;

    // Window Management
    void set_window_type(contorchionist::core::util_windowing::Type window_type);
    contorchionist::core::util_windowing::Type get_window_type() const;
    void enable_windowing(bool enabled);
    bool is_windowing_enabled() const;
    bool is_window_prepared() const;
    void initialize_window(long n);
    void release_window();

    // Sampling Rate Management
    void set_sampling_rate(T fs);
    T get_sampling_rate() const;

    // Normalization Management
    void set_normalization_type(core::util_normalizations::NormalizationType normtype);
    core::util_normalizations::NormalizationType get_normalization_type() const;
    
    void set_normalization(int fftdirection, int fft_mode, long n_fft, 
                       core::util_normalizations::NormalizationType norm_type, T fs, T overlap_factor);

    // Output Format Management
    void set_output_format(contorchionist::core::util_conversions::SpectrumDataFormat format);
    contorchionist::core::util_conversions::SpectrumDataFormat get_output_format() const;

    // Spectrum Normalization Helper
    void apply_norm_factors(torch::Tensor& spectrum_tensor,
                                     FFTMode fft_mode,
                                     T dc,
                                     T ac,
                                     T nyquist);

 
    // RFFT Processing (forward FFT)
    std::vector<torch::Tensor> process_rfft(const torch::Tensor& input_tensor_cpu);

    // IRFFT Processing (inverse FFT)
    torch::Tensor process_irfft(const std::vector<torch::Tensor>& input_spectrum_vec, long output_n,
                               contorchionist::core::util_conversions::SpectrumDataFormat input_format, 
                               bool use_input_phase = true);

    // Scale by window sum management
    void set_scale_by_window_sum(bool enable);

private:
    // Core device and configuration
    torch::Device device_;
    contorchionist::core::util_windowing::Type window_type_;
    bool windowing_enabled_;
    bool window_prepared_;
    bool verbose_;
    core::util_normalizations::NormalizationType normalization_type_;
    contorchionist::core::util_conversions::SpectrumDataFormat output_format_;
    T fs_;

    // Window tensors and properties
    torch::Tensor window_;
    torch::Tensor window_cpu_;
    long current_window_n_;
    T current_sum_window_;
    T current_sum_sq_window_;

    // Normalization factors
    T mag_dc_norm_factor_;
    T mag_ac_norm_factor_;
    T mag_nyquist_norm_factor_;
    T pw_dc_norm_factor_;
    T pw_ac_norm_factor_;
    T pw_nyquist_norm_factor_;
    T overlap_factor_;
    int fft_mode_;
    T overall_scale_ = 1.0f;
    bool scale_by_window_sum_ = false;

    // Helper methods
    void log(const std::string& message) const;
    void update_window_sums(long n_for_sum_calc);
};

// Implementation

template<typename T>
RFFTProcessor<T>::RFFTProcessor(torch::Device device,
                             contorchionist::core::util_windowing::Type window_type,
                             bool windowing_enabled,
                             bool verbose,
                             core::util_normalizations::NormalizationType norm_type,
                             contorchionist::core::util_conversions::SpectrumDataFormat output_format,
                             T fs)
    : device_(device),
      window_type_(window_type),
      windowing_enabled_(windowing_enabled),
      window_prepared_(false),
      verbose_(verbose),
      normalization_type_(norm_type),
      output_format_(output_format),
      fs_(fs),
      current_window_n_(-1),
      current_sum_window_(0.0f),
      current_sum_sq_window_(0.0f),
      mag_dc_norm_factor_(1.0f),
      mag_ac_norm_factor_(1.0f),
      mag_nyquist_norm_factor_(1.0f),
      pw_dc_norm_factor_(1.0f),
      pw_ac_norm_factor_(1.0f),
      pw_nyquist_norm_factor_(1.0f),
      overlap_factor_(1.0f),
      fft_mode_(2),
      overall_scale_(1.0f),
      scale_by_window_sum_(false)
{
    log("RFFTProcessor initialized.");
}

template<typename T>
RFFTProcessor<T>::~RFFTProcessor() {
    release_window();
    log("RFFTProcessor destroyed.");
}

template<typename T>
void RFFTProcessor<T>::set_verbose(bool verbose) {
    verbose_ = verbose;
}

template<typename T>
bool RFFTProcessor<T>::is_verbose() const {
    return verbose_;
}

template<typename T>
void RFFTProcessor<T>::set_device(torch::Device device) {
    device_ = device;
    if (window_.defined()) {
        window_ = window_.to(device_);
    }
    log("Device set to: " + device_.str());
}

template<typename T>
torch::Device RFFTProcessor<T>::get_device() const {
    return device_;
}

template<typename T>
void RFFTProcessor<T>::set_window_type(contorchionist::core::util_windowing::Type window_type) {
    window_type_ = window_type;
    if (current_window_n_ > 0) {
        initialize_window(current_window_n_);
    }
    log("Window type set.");
}

template<typename T>
void RFFTProcessor<T>::enable_windowing(bool enabled) {
    windowing_enabled_ = enabled;
    log(std::string("Windowing ") + (enabled ? "enabled." : "disabled."));
}

template<typename T>
bool RFFTProcessor<T>::is_windowing_enabled() const {
    return windowing_enabled_;
}

template<typename T>
bool RFFTProcessor<T>::is_window_prepared() const {
    return window_prepared_;
}

template<typename T>
void RFFTProcessor<T>::initialize_window(long n) {
    if (n <= 0) {
        release_window();
        log("Window initialization skipped: n must be positive.");
        return;
    }
    current_window_n_ = n;
    auto window_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    window_cpu_ = contorchionist::core::util_windowing::generate_torch_window(current_window_n_, window_type_, true, window_opts);

    if (device_.type() != torch::kCPU) {
        window_ = window_cpu_.to(device_);
    } else {
        window_ = window_cpu_;
    }

    update_window_sums(current_window_n_);
    window_prepared_ = true;
    log("Window initialized with size " + std::to_string(n) + " for type " + contorchionist::core::util_windowing::torch_window_type_to_string(window_type_));
}

template<typename T>
void RFFTProcessor<T>::release_window() {
    window_ = torch::Tensor();
    window_cpu_ = torch::Tensor();
    current_window_n_ = -1;
    current_sum_window_ = 0.0f;
    current_sum_sq_window_ = 0.0f;
    log("Window released.");
}

template<typename T>
void RFFTProcessor<T>::update_window_sums(long n_for_sum_calc) {
    if (!window_cpu_.defined() || window_cpu_.numel() == 0) {
        current_sum_window_ = 0.0f;
        current_sum_sq_window_ = 0.0f;
        log("Window sums not updated: CPU window not defined/empty.");
        return;
    }
    torch::Tensor window_float = window_cpu_.to(torch::kFloat32);
    current_sum_window_ = window_float.sum().item<T>();
    current_sum_sq_window_ = window_float.pow(2).sum().item<T>();
    log("Window sums updated: sum=" + std::to_string(current_sum_window_) + ", sum_sq=" + std::to_string(current_sum_sq_window_));
}

template<typename T>
void RFFTProcessor<T>::set_sampling_rate(T fs) {
    fs_ = fs;
    if (verbose_) log("Sampling rate set to: " + std::to_string(fs) + " Hz");
}

template<typename T>
T RFFTProcessor<T>::get_sampling_rate() const {
    return fs_;
}

template<typename T>
void RFFTProcessor<T>::set_normalization_type(core::util_normalizations::NormalizationType normtype) {
    normalization_type_ = normtype;
    log("Normalization mode set.");
}

template<typename T>
core::util_normalizations::NormalizationType RFFTProcessor<T>::get_normalization_type() const {
    return normalization_type_;
}

template<typename T>
void RFFTProcessor<T>::set_normalization(int fftdirection, int fft_mode, long n_fft,
                                      core::util_normalizations::NormalizationType norm_type, T fs, T overlap_factor) {

    fft_mode_ = fft_mode;
    overlap_factor_ = overlap_factor;
    normalization_type_ = norm_type;
    set_sampling_rate(fs);

    initialize_window(n_fft);

    torch::Tensor window_for_norm = window_cpu_.defined() ? window_cpu_ : window_;
    core::util_normalizations::SpectrumNormFactors norm_factors;

    if (fftdirection == 1) {
        norm_factors = core::util_normalizations::NormalizationFactors::get_norm_factors(
                fft_mode, n_fft, window_for_norm, norm_type, fs_, scale_by_window_sum_, current_sum_window_);
    } else if (fftdirection == -1) {
        norm_factors = core::util_normalizations::NormalizationFactors::get_denorm_factors(
                fft_mode, n_fft, window_for_norm, norm_type, fs_);
    }

    mag_dc_norm_factor_ = norm_factors.mag_dc;
    mag_ac_norm_factor_ = norm_factors.mag_ac;
    mag_nyquist_norm_factor_ = norm_factors.mag_nyquist;
    pw_dc_norm_factor_ = norm_factors.pw_dc;
    pw_ac_norm_factor_ = norm_factors.pw_ac;
    pw_nyquist_norm_factor_ = norm_factors.pw_nyquist;
    overall_scale_ = norm_factors.overall_scale;
}

template<typename T>
void RFFTProcessor<T>::apply_norm_factors(torch::Tensor& spectrum_tensor,
                                                    FFTMode fft_mode,
                                                    T dc,
                                                    T ac,
                                                    T nyquist) {
    if (spectrum_tensor.numel() == 0) {
        return;
    }
    long spec_len = spectrum_tensor.size(-1);

    switch (fft_mode) {
        case FFTMode::RFFT: {
            if (spec_len > 0) spectrum_tensor.select(-1, 0) *= dc;
            if (spec_len > 1) {
                spectrum_tensor.select(-1, spec_len - 1) *= nyquist;
                if (spec_len > 2) {
                    spectrum_tensor.slice(-1, 1, spec_len - 1) *= ac;
                }
            }
            break;
        }
        case FFTMode::FFT: {
            if (spec_len > 0) spectrum_tensor.select(-1, 0) *= dc;
            if (spec_len > 1) {
                if (spec_len % 2 == 0) {
                    spectrum_tensor.select(-1, spec_len / 2) *= nyquist;
                    spectrum_tensor.slice(-1, 1, spec_len / 2) *= ac;
                    spectrum_tensor.slice(-1, spec_len / 2 + 1, spec_len) *= ac;
                } else {
                    spectrum_tensor.slice(-1, 1, spec_len) *= ac;
                }
            }
            break;
        }
        default:
            throw std::invalid_argument("Invalid FFTMode provided to apply_spectrum_normalization.");
    }
}

template<typename T>
void RFFTProcessor<T>::set_scale_by_window_sum(bool enable) {
    scale_by_window_sum_ = enable;
    log("Scale by window sum " + std::string(enable ? "enabled." : "disabled."));
}

template<typename T>
void RFFTProcessor<T>::set_output_format(contorchionist::core::util_conversions::SpectrumDataFormat format) {
    output_format_ = format;
    log("Output format set.");
}

template<typename T>
contorchionist::core::util_conversions::SpectrumDataFormat RFFTProcessor<T>::get_output_format() const {
    return output_format_;
}

template<typename T>
std::vector<torch::Tensor> RFFTProcessor<T>::process_rfft(const torch::Tensor& input_tensor_cpu) {
    long n_fft = input_tensor_cpu.size(-1);
    torch::Tensor processed_input = input_tensor_cpu.to(device_);

    if (windowing_enabled_ && window_prepared_) {
        if (overlap_factor_ <= 1.0f) {
            processed_input = processed_input * window_;
        } else {
            processed_input = processed_input * window_ / overlap_factor_;
        }
    }

    torch::Tensor complex_spectrum;
    if (fft_mode_ == 2) {
        complex_spectrum = torch::fft::rfft(processed_input, n_fft, -1, "backward");
    } else {
        complex_spectrum = torch::fft::fft(processed_input, n_fft, -1, "backward");
    }

    FFTMode current_fft_mode = static_cast<FFTMode>(fft_mode_);
    complex_spectrum *= overall_scale_;

    apply_norm_factors(complex_spectrum, current_fft_mode, mag_dc_norm_factor_, mag_ac_norm_factor_, mag_nyquist_norm_factor_);

    if (output_format_ == core::util_conversions::SpectrumDataFormat::COMPLEX) {
        if (pw_dc_norm_factor_ == 1.0f && pw_ac_norm_factor_ == 1.0f && pw_nyquist_norm_factor_ == 1.0f) {
            return {complex_spectrum};
        }
    }

    auto phase = torch::angle(complex_spectrum);
    auto power = torch::square(torch::real(complex_spectrum)) + torch::square(torch::imag(complex_spectrum));

    apply_norm_factors(power, current_fft_mode, pw_dc_norm_factor_, pw_ac_norm_factor_, pw_nyquist_norm_factor_);

    if (output_format_ == core::util_conversions::SpectrumDataFormat::DBPHASE) {
        auto db = 10.0 * torch::log10(torch::clamp(power, 1e-10f));
        return {db, phase};
    }

    if (output_format_ == core::util_conversions::SpectrumDataFormat::POWERPHASE) {
        return {power, phase};
    }

    if (output_format_ == core::util_conversions::SpectrumDataFormat::MAGPHASE) {
        auto magnitude = torch::sqrt(torch::clamp(power, 0.0f));
        return {magnitude, phase};
    }

    complex_spectrum = torch::polar(torch::sqrt(torch::clamp(power, 0.0f)), phase);
    return {complex_spectrum};
}

template<typename T>
torch::Tensor RFFTProcessor<T>::process_irfft(const std::vector<torch::Tensor>& input_spectrum_vec, long output_n,
                                           contorchionist::core::util_conversions::SpectrumDataFormat input_format, bool use_input_phase) {
    if (output_n <= 0) {
        throw std::invalid_argument("Output length n for IRFFT must be positive.");
    }

    torch::Tensor complex_spectrum;

    FFTMode current_fft_mode = static_cast<FFTMode>(fft_mode_);
    torch::Tensor power;
    torch::Tensor phase;

    switch (input_format) {
        case core::util_conversions::SpectrumDataFormat::COMPLEX: {
            auto real_part = input_spectrum_vec[0];
            auto imag_part = input_spectrum_vec[1];
            power = torch::square(real_part) + torch::square(imag_part);

            if (use_input_phase) {
                phase = torch::angle(torch::complex(real_part, imag_part));
            }
            break;
        }
        case core::util_conversions::SpectrumDataFormat::MAGPHASE: {
            auto magnitude = input_spectrum_vec[0];
            power = torch::square(magnitude);
            if (use_input_phase) {
                phase = input_spectrum_vec[1];
            }
            break;
        }
        case core::util_conversions::SpectrumDataFormat::POWERPHASE: {
            power = input_spectrum_vec[0];
            if (use_input_phase) {
                phase = input_spectrum_vec[1];
            }
            break;
        }
        case core::util_conversions::SpectrumDataFormat::DBPHASE: {
            auto db = input_spectrum_vec[0];
            power = torch::pow(10.0, db / 10.0);
            if (use_input_phase) {
                phase = input_spectrum_vec[1];
            }
            break;
        }
        default: {
            throw std::invalid_argument("IRFFT: Unsupported input format");
        }
    }

    if (!use_input_phase) {
        phase = torch::zeros_like(power);
    }

    apply_norm_factors(power, current_fft_mode, pw_dc_norm_factor_, pw_ac_norm_factor_, pw_nyquist_norm_factor_);
    auto magnitude = torch::sqrt(torch::clamp(power, 0.0f));
    torch::Tensor final_complex_spectrum = torch::polar(magnitude, phase);
    apply_norm_factors(final_complex_spectrum, current_fft_mode, mag_dc_norm_factor_, mag_ac_norm_factor_, mag_nyquist_norm_factor_);

    final_complex_spectrum *= overall_scale_;
    final_complex_spectrum = final_complex_spectrum.to(device_);

    torch::Tensor time_signal;
    if (fft_mode_ == 2) {
        time_signal = torch::fft::irfft(final_complex_spectrum, output_n, -1, "forward");
    } else {
        torch::Tensor complex_time_signal = torch::fft::ifft(final_complex_spectrum, output_n, -1, "forward");
        time_signal = torch::real(complex_time_signal);
    }

    if (overlap_factor_ >= 1.0f) {
        time_signal = time_signal / overlap_factor_;
    }
    return time_signal;
}

template<typename T>
void RFFTProcessor<T>::log(const std::string& message) const {
    if (verbose_) {
        std::cout << "[RFFTProcessor LOG]: " << message << std::endl;
    }
}

        } // namespace ap_rfft
    } // namespace core
} // namespace contorchionist


#endif // CORE_AP_RFFT_H

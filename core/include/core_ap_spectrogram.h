#ifndef CORE_AP_SPECTROGRAM_H
#define CORE_AP_SPECTROGRAM_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cctype>

#include "core_util_circbuffer.h"
#include "core_ap_rfft.h"
#include "core_util_windowing.h"
#include "core_util_normalizations.h"
#include "core_util_conversions.h"

namespace contorchionist {
namespace core {
namespace ap_spectrogram {

template<typename T>
class SpectrogramProcessor {
public:
    SpectrogramProcessor(int n_fft,
                         int hop_size,
                         contorchionist::core::util_windowing::Type window_type,
                         contorchionist::core::util_conversions::SpectrumDataFormat output_format,
                         contorchionist::core::util_normalizations::NormalizationType norm_type,
                         torch::Device device,
                         float fs = 48000.0f,
                         bool verbose = false)
        : n_fft_(n_fft),
          hop_size_(hop_size),
          window_type_(window_type),
          output_format_(output_format),
          normalization_type_(norm_type),
          current_fs_(fs),
          device_(device),
          verbose_(verbose),
          circular_buffer_(n_fft),
          samples_processed_current_hop_(0) {

        if (verbose_) {
            std::cout << "SpectrogramProcessor: Initializing with n_fft=" << n_fft_
                      << ", hop_size=" << hop_size_
                      << ", window=" << contorchionist::core::util_windowing::torch_window_type_to_string(window_type_)
                      << ", format=" << core::util_conversions::spectrum_data_format_to_string(output_format_)
                      << ", norm=" << core::util_normalizations::normalization_type_to_string(normalization_type_)
                      << ", device=" << device_.str()
                      << ", fs=" << current_fs_ << std::endl;
        }

        if (n_fft_ <= 0 || hop_size_ <= 0) {
            throw std::invalid_argument("SpectrogramProcessor: n_fft and hop_size must be positive.");
        }

        internal_frame_buffer_.resize(n_fft_);
        setup_rfft_processor();
    }

    // Process a block of audio data and produce spectrogram frames
    // args: <input_buffer, buffer_size, output_frame1, output_frame2>
    bool process(const T* input_buffer, int buffer_size,
                 std::vector<T>& output_frame1, std::vector<T>& output_frame2) {
        if (verbose_) {
            // std::cout << "[SpectrogramProcessor DEBUG] process() called with buffer_size=" << buffer_size << std::endl;
        }

        circular_buffer_.write_overwrite(input_buffer, buffer_size);
        samples_processed_current_hop_ += buffer_size;

        if (samples_processed_current_hop_ >= hop_size_) {
            if (circular_buffer_.getSamplesAvailable() < n_fft_) {
                if (verbose_) {
                    // std::cout << "[SpectrogramProcessor DEBUG] Not enough samples for full window yet. Available: "
                    //           << circular_buffer_.getSamplesAvailable() << ", Needed: " << n_fft_ << std::endl;
                }
                return false;
            }

            size_t samples_read = circular_buffer_.peek_with_delay_and_fill(internal_frame_buffer_.data(), n_fft_, 0);

            if (samples_read < n_fft_) {
                if (verbose_) {
                    std::cerr << "SpectrogramProcessor: Error reading full window from circular buffer. Read: "
                              << samples_read << ", Expected: " << n_fft_ << std::endl;
                }
                return false;
            }

            torch::Tensor frame_tensor_cpu = torch::from_blob(internal_frame_buffer_.data(), {n_fft_},
                                                              torch::TensorOptions().dtype(torch::kFloat32));

            std::vector<torch::Tensor> rfft_result;
            try {
                rfft_result = rfft_processor_->process_rfft(frame_tensor_cpu);
            } catch (const std::exception& e) {
                std::cerr << "SpectrogramProcessor: Error during RFFT processing: " << e.what() << std::endl;
                return false;
            }

            if (rfft_result.empty() || !rfft_result[0].defined()) {
                std::cerr << "SpectrogramProcessor: RFFT processing failed or returned empty/undefined tensor." << std::endl;
                return false;
            }

            torch::Tensor result_tensor1 = rfft_result[0].to(torch::kCPU);
            int64_t output_spectrum_size = result_tensor1.numel();

            output_frame1.resize(output_spectrum_size);
            output_frame2.clear();

            if (!result_tensor1.is_contiguous()) result_tensor1 = result_tensor1.contiguous();

            if (result_tensor1.is_complex()) {
                result_tensor1 = torch::real(result_tensor1);
            }

            std::memcpy(output_frame1.data(), result_tensor1.data_ptr<T>(), output_spectrum_size * sizeof(T));

            if (rfft_result.size() > 1 && rfft_result[1].defined()) {
                torch::Tensor result_tensor2 = rfft_result[1].to(torch::kCPU);
                int64_t output_spectrum_size2 = result_tensor2.numel();
                output_frame2.resize(output_spectrum_size2);
                if (!result_tensor2.is_contiguous()) result_tensor2 = result_tensor2.contiguous();
                std::memcpy(output_frame2.data(), result_tensor2.data_ptr<T>(), output_spectrum_size2 * sizeof(T));
            }

            samples_processed_current_hop_ -= hop_size_;
            return true;
        }
        return false;
    }

    void set_output_format(contorchionist::core::util_conversions::SpectrumDataFormat format) {
        if (output_format_ != format) {
            output_format_ = format;
            rfft_processor_->set_output_format(format);
            if (verbose_) std::cout << "SpectrogramProcessor: Output format set to "
                                    << core::util_conversions::spectrum_data_format_to_string(format) << std::endl;
        }
    }

    contorchionist::core::util_conversions::SpectrumDataFormat get_output_format() const {
        return output_format_;
    }

    void set_normalization_type(contorchionist::core::util_normalizations::NormalizationType norm_type) {
        if (normalization_type_ != norm_type) {
            normalization_type_ = norm_type;
            rfft_processor_->set_normalization(1, 2, n_fft_, normalization_type_, current_fs_, 1.0f);
            if (verbose_) std::cout << "SpectrogramProcessor: Normalization type set to "
                                    << core::util_normalizations::normalization_type_to_string(norm_type) << std::endl;
        }
    }

    contorchionist::core::util_normalizations::NormalizationType get_normalization_type() const {
        return normalization_type_;
    }

    void set_window_type(contorchionist::core::util_windowing::Type window_type) {
        if (window_type_ != window_type) {
            window_type_ = window_type;
            rfft_processor_->set_window_type(window_type_);
            rfft_processor_->set_normalization(1, 2, n_fft_, normalization_type_, current_fs_, 1.0f);
            if (verbose_) std::cout << "SpectrogramProcessor: Window type set to "
                                     << contorchionist::core::util_windowing::torch_window_type_to_string(window_type_) << std::endl;
        }
    }

    contorchionist::core::util_windowing::Type get_window_type() const {
        return window_type_;
    }

    void set_sampling_rate(float fs) {
        if (current_fs_ != fs) {
            current_fs_ = fs;
            rfft_processor_->set_sampling_rate(fs);
            rfft_processor_->set_normalization(1, 2, n_fft_, normalization_type_, current_fs_, 1.0f);
            if (verbose_) std::cout << "SpectrogramProcessor: Sampling rate set to " << fs << std::endl;
        }
    }

    float get_sampling_rate() const {
        return current_fs_;
    }

    int get_n_fft() const {
        return n_fft_;
    }

    int get_hop_size() const {
        return hop_size_;
    }

    void set_verbose(bool verbose) {
        verbose_ = verbose;
        if (rfft_processor_) {
            rfft_processor_->set_verbose(verbose_);
        }
    }

private:
    void setup_rfft_processor() {
        if (verbose_) {
            std::cout << "SpectrogramProcessor: Setting up RFFTProcessor..." << std::endl;
            std::cout << "  n_fft: " << n_fft_ << std::endl;
            std::cout << "  window_type: " << contorchionist::core::util_windowing::torch_window_type_to_string(window_type_) << std::endl;
            std::cout << "  output_format: " << core::util_conversions::spectrum_data_format_to_string(output_format_) << std::endl;
            std::cout << "  normalization_type: " << core::util_normalizations::normalization_type_to_string(normalization_type_) << std::endl;
            std::cout << "  sampling_rate: " << current_fs_ << std::endl;
            std::cout << "  device: " << device_.str() << std::endl;
        }

        rfft_processor_ = std::make_unique<contorchionist::core::ap_rfft::RFFTProcessor<T>>(
            device_,
            window_type_,
            true,
            verbose_,
            normalization_type_,
            output_format_,
            current_fs_
        );

        rfft_processor_->set_normalization(1, 2, n_fft_, normalization_type_, current_fs_, 1.0f);

        if (verbose_) {
            std::cout << "SpectrogramProcessor: RFFTProcessor setup complete." << std::endl;
        }
    }

    int n_fft_;
    int hop_size_;
    contorchionist::core::util_windowing::Type window_type_;
    contorchionist::core::util_conversions::SpectrumDataFormat output_format_;
    contorchionist::core::util_normalizations::NormalizationType normalization_type_;
    float current_fs_;

    torch::Device device_;
    bool verbose_;

    std::unique_ptr<contorchionist::core::ap_rfft::RFFTProcessor<T>> rfft_processor_;
    contorchionist::core::util_circbuffer::CircularBuffer<T> circular_buffer_;
    std::vector<T> internal_frame_buffer_;
    int samples_processed_current_hop_;
};

} // namespace ap_spectrogram
} // namespace core
} // namespace contorchionist

#endif // CORE_AP_SPECTROGRAM_H

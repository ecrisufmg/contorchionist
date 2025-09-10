#ifndef CORE_AP_MELSPECTROGRAM_H
#define CORE_AP_MELSPECTROGRAM_H

#include <torch/torch.h>
#include "core_util_windowing.h" 
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cctype>

#include "core_util_circbuffer.h"
#include "core_util_normalizations.h"
#include "core_util_conversions.h"
#include "core_ap_rfft.h"

namespace contorchionist {
    namespace core {
        namespace ap_melspectrogram {

// Enum for Mel Normalization Modes
enum class MelNormMode {
    NONE,         // No additional energy/magnitude sum normalization
    ENERGY_POWER, // Conserve sum of power: sum(Mel_Power_Out) == sum(RFFT_Power_In)
    MAGNITUDE_SUM // Conserve sum of magnitudes: sum(Mel_Magnitude_Out) == sum(RFFT_Magnitude_In)
};

// Helper functions for MelNormMode
inline std::string mel_norm_mode_to_string(MelNormMode mode) {
    switch (mode) {
        case MelNormMode::NONE: return "none";
        case MelNormMode::ENERGY_POWER: return "energy";
        case MelNormMode::MAGNITUDE_SUM: return "magnitude_sum";
        default: return "unknown";
    }
}

inline MelNormMode string_to_mel_norm_mode(const std::string& mode_str) {
    std::string lower_str = mode_str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    if (lower_str == "none" || lower_str == "off") {
        return MelNormMode::NONE;
    } else if (lower_str == "energy" || lower_str == "power" || lower_str == "pw") {
        return MelNormMode::ENERGY_POWER;
    } else if (lower_str == "magnitude_sum" || lower_str == "mag_sum" || lower_str == "ms" || lower_str == "magnitude") {
        return MelNormMode::MAGNITUDE_SUM;
    }
    throw std::invalid_argument("Unknown MelNormMode string: " + mode_str);
}


// Mel filterbank creation function
inline torch::Tensor create_mel_filterbank(
    float sr, int n_fft, int n_mels, float fmin, float fmax_actual,
    contorchionist::core::util_conversions::MelFormulaType mel_formula, const std::string& norm_str, torch::Device device) {

    using namespace contorchionist::core::util_conversions;

    if (sr <= 0) throw std::invalid_argument("Sample rate must be positive. Got: " + std::to_string(sr));
    if (n_fft <= 0) throw std::invalid_argument("n_fft must be positive. Got: " + std::to_string(n_fft));
    if (n_mels <= 0) throw std::invalid_argument("n_mels must be positive. Got: " + std::to_string(n_mels));
    if (fmin < 0) throw std::invalid_argument("fmin must be non-negative. Got: " + std::to_string(fmin));
    if (fmax_actual <= fmin) throw std::invalid_argument("fmax_actual (" + std::to_string(fmax_actual) + ") must be greater than fmin (" + std::to_string(fmin) + ").");

    auto options_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor weights = torch::zeros({n_mels, n_fft / 2 + 1}, options_cpu);
    torch::Tensor fft_freqs = fftFrequencies(sr, n_fft).to(device);
    torch::Tensor mel_freqs_hz = melFrequencies(n_mels + 2, fmin, fmax_actual, mel_formula).to(device);

    torch::Tensor fdiff = torch::diff(mel_freqs_hz);
    torch::Tensor ramps = mel_freqs_hz.unsqueeze(1) - fft_freqs.unsqueeze(0);

    for (int i = 0; i < n_mels; ++i) {
        torch::Tensor lower = -ramps.index({i}) / fdiff.index({i});
        torch::Tensor upper = ramps.index({i + 2}) / fdiff.index({i + 1});
        weights.index_put_({i}, torch::maximum(torch::zeros_like(lower), torch::minimum(lower, upper)).to(torch::kCPU));
    }

    if (norm_str == "slaney") {
        torch::Tensor mel_f_slice = mel_freqs_hz.slice(0, 2, n_mels + 2);
        torch::Tensor mel_f_start = mel_freqs_hz.slice(0, 0, n_mels);

        if (mel_f_slice.size(0) != n_mels || mel_f_start.size(0) != n_mels) {
            throw std::runtime_error(
                "Slaney norm: Mismatch in slice sizes. Expected " + std::to_string(n_mels) +
                ", got mel_f_slice: " + std::to_string(mel_f_slice.size(0)) +
                ", mel_f_start: " + std::to_string(mel_f_start.size(0)));
        }

        torch::Tensor enorm = 2.0 / (mel_f_slice - mel_f_start);
        weights *= enorm.unsqueeze(1).to(torch::kCPU);
    } else if (norm_str == "htk") {
        torch::Tensor mel_f_in_mel_scale = torch::empty_like(mel_freqs_hz);
        for (int j = 0; j < mel_freqs_hz.numel(); ++j) {
            float val_hz = mel_freqs_hz[j].item<float>();
            switch (mel_formula) {
                case contorchionist::core::util_conversions::MelFormulaType::HTK:
                    mel_f_in_mel_scale[j] = hzToMelHTK(val_hz);
                    break;
                case contorchionist::core::util_conversions::MelFormulaType::CALC2:
                    mel_f_in_mel_scale[j] = hzToMelCalc2(val_hz);
                    break;
                case contorchionist::core::util_conversions::MelFormulaType::SLANEY:
                default:
                    mel_f_in_mel_scale[j] = hzToMelSlaney(val_hz);
                    break;
            }
        }
        torch::Tensor mel_f_left_mel = mel_f_in_mel_scale.index({torch::indexing::Slice(torch::indexing::None, n_mels)});
        torch::Tensor mel_f_right_mel = mel_f_in_mel_scale.index({torch::indexing::Slice(2, torch::indexing::None)});
        torch::Tensor mel_bandwidths_melscale = mel_f_right_mel - mel_f_left_mel;
        mel_bandwidths_melscale.masked_fill_(mel_bandwidths_melscale == 0.0f, 1e-8f);
        weights /= mel_bandwidths_melscale.unsqueeze(1).to(torch::kCPU);
    } else if (!norm_str.empty() && norm_str != "none") {
        throw std::invalid_argument("Unsupported mel_norm: " + norm_str + ". Supported: 'slaney', 'htk', 'none'.");
    }

    auto max_weight_per_filter = std::get<0>(torch::max(weights, /*dim=*/1));
    if (torch::any(max_weight_per_filter == 0.0f).item<bool>()) {
        std::cerr << "[MelSpectrogramProcessor create_mel_filterbank] Warning: Empty filters detected." << std::endl;
    }
    return weights.to(device);
}

template<typename T>
class MelSpectrogramProcessor {
public:
    MelSpectrogramProcessor(
        int n_fft = 2048,
        int hop_length = 512,
        int win_length_param = -1, // If -1 or 0, use n_fft
        contorchionist::core::util_windowing::Type window_type = contorchionist::core::util_windowing::Type::HANN,
        contorchionist::core::util_normalizations::NormalizationType rfft_norm_type = contorchionist::core::util_normalizations::NormalizationType::POWER,
        contorchionist::core::util_conversions::SpectrumDataFormat output_format = contorchionist::core::util_conversions::SpectrumDataFormat::POWERPHASE,
        float sample_rate = 48000.0f,
        int n_mels = 128,
        float fmin_mel = 0.0f,
        float fmax_mel_param_val = -1.0f, // If -1 or 0, use sample_rate / 2.0f
        contorchionist::core::util_conversions::MelFormulaType mel_formula = contorchionist::core::util_conversions::MelFormulaType::CALC2,
        const std::string& filterbank_norm_str = "slaney",
        MelNormMode mel_norm_mode_val = MelNormMode::ENERGY_POWER,
        torch::Device device = torch::kCPU,
        bool verbose = false
    ) : device_(device), window_type_(window_type), verbose_(verbose),
        sample_rate_(sample_rate), n_mels_(n_mels), n_fft_(n_fft), hop_length_(hop_length),
        win_length_((win_length_param <= 0) ? n_fft : win_length_param),
        fmin_mel_(fmin_mel), fmax_mel_param_(fmax_mel_param_val),
        mel_formula_(mel_formula),
        filterbank_norm_(filterbank_norm_str),
        mel_norm_mode_(mel_norm_mode_val),
        rfft_normalization_type_(rfft_norm_type),
        output_format_(output_format),
        circular_buffer_(nullptr),
        samples_processed_current_hop_(0),
        rfft_processor_(nullptr),
        mel_filterbank_initialized_(false)
    {
        validate_parameters();

        fmax_mel_actual_ = (fmax_mel_param_ <= 0 && sample_rate_ > 0) ? (sample_rate_ / 2.0f) : fmax_mel_param_;

        circular_buffer_ = std::make_unique<contorchionist::core::util_circbuffer::CircularBuffer<T>>(win_length_);
        internal_frame_buffer_.resize(win_length_);

        setup_rfft_processor();
        initialize_mel_filterbank();

        if (verbose_) {
            log("MelSpectrogramProcessor created. sr=" + std::to_string(sample_rate_) +
                ", n_fft=" + std::to_string(n_fft_) + ", hop=" + std::to_string(hop_length_) +
                ", win_len=" + std::to_string(win_length_) + ", n_mels=" + std::to_string(n_mels_) +
                ", mel_formula=" + contorchionist::core::util_conversions::mel_formula_type_to_string(mel_formula_) +
                ", rfft_norm=" + contorchionist::core::util_normalizations::normalization_type_to_string(rfft_normalization_type_) +
                ", filterbank_norm=" + filterbank_norm_ +
                ", mel_norm_mode=" + mel_norm_mode_to_string(mel_norm_mode_));
        }
    }
    
    ~MelSpectrogramProcessor() {
        if (verbose_) {
            log("MelSpectrogramProcessor destroyed");
        }
    }

    bool process(const T* input_buffer_ptr, int buffer_size,
                 std::vector<T>& output_frame1, std::vector<T>& output_frame2) {
        if (!rfft_processor_ || !mel_filterbank_initialized_) {
            if (verbose_) log("Processor or filterbank not ready. Skipping process.");
            output_frame1.clear();
            output_frame2.clear();
            return false;
        }
        output_frame2.clear();

        circular_buffer_->write_overwrite(input_buffer_ptr, buffer_size);
        samples_processed_current_hop_ += buffer_size;

        if (samples_processed_current_hop_ >= hop_length_) {
            if (circular_buffer_->getSamplesAvailable() < static_cast<size_t>(win_length_)) {
                if (verbose_) {
                    log("Not enough samples for a full window. Available: " +
                        std::to_string(circular_buffer_->getSamplesAvailable()) +
                        ", Needed: " + std::to_string(win_length_));
                }
                return false;
            }

            size_t samples_read = circular_buffer_->peek_with_delay_and_fill(
                internal_frame_buffer_.data(), win_length_, 0
            );

            if (samples_read < static_cast<size_t>(win_length_)) {
                if (verbose_) {
                    log("Error reading full window from circular buffer. Read: " +
                        std::to_string(samples_read) + ", Expected: " + std::to_string(win_length_));
                }
                return false;
            }

            torch::Tensor frame_tensor_cpu = torch::from_blob(
                internal_frame_buffer_.data(), {win_length_},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
            );

            std::vector<torch::Tensor> rfft_result;
            try {
                torch::Tensor input_for_rfft = frame_tensor_cpu;
                if (win_length_ < n_fft_) {
                    int64_t padding_needed = n_fft_ - win_length_;
                    input_for_rfft = torch::nn::functional::pad(frame_tensor_cpu, torch::nn::functional::PadFuncOptions({0, padding_needed}));
                } else if (win_length_ > n_fft_) {
                    if (verbose_) log("Error: win_length_ > n_fft_. Truncating input for RFFT. This is not ideal.");
                    input_for_rfft = frame_tensor_cpu.slice(0, 0, n_fft_);
                }
                rfft_result = rfft_processor_->process_rfft(input_for_rfft);
            } catch (const std::exception& e) {
                if (verbose_) log("Error during RFFT processing: " + std::string(e.what()));
                return false;
            }

            if (rfft_result.empty() || !rfft_result[0].defined()) {
                if (verbose_) log("RFFT processing returned empty or undefined tensor.");
                return false;
            }

            torch::Tensor power_spectrogram = rfft_result[0].to(device_);

            if (power_spectrogram.dim() == 2 && power_spectrogram.size(1) == 1) {
                power_spectrogram = power_spectrogram.squeeze(1);
            } else if (power_spectrogram.dim() != 1) {
                if (verbose_) log("Power spectrogram from RFFT has unexpected dimensions: " + torch::str(power_spectrogram.sizes()) + ". Expected 1D or [N,1].");
                return false;
            }

            int input_freq_bins = power_spectrogram.size(0);
            int expected_freq_bins = mel_filterbank_.size(1);
            if (input_freq_bins != expected_freq_bins) {
                if (verbose_) {
                    log("Mismatch in frequency bins for Mel transform. Power spec has " + std::to_string(input_freq_bins) +
                        ", filterbank expects " + std::to_string(expected_freq_bins) + ". n_fft was " + std::to_string(n_fft_));
                }
                return false;
            }

            torch::Tensor mel_power_tensor = torch::matmul(mel_filterbank_, power_spectrogram);

            torch::Tensor final_mel_tensor_basis = mel_power_tensor.clone();

            if (mel_norm_mode_ == MelNormMode::ENERGY_POWER) {
                if (verbose_) log("Applying MelNormMode::ENERGY_POWER");

                torch::Tensor input_power_for_sum = rfft_result[0].to(device_);
                if (input_power_for_sum.dim() == 2 && input_power_for_sum.size(1) == 1) {
                    input_power_for_sum = input_power_for_sum.squeeze(1);
                }

                float e_insum = torch::sum(input_power_for_sum).item<float>();
                float e_mel_bruto = torch::sum(mel_power_tensor).item<float>();

                float m_norm_factor = 1.0f;
                if (e_mel_bruto != 0.0f) {
                    m_norm_factor = e_insum / e_mel_bruto;
                } else if (e_insum != 0.0f && verbose_) {
                    log("Warning: E_mel_bruto is 0 for ENERGY_POWER but E_insum is not. Mel output will be zero.");
                }
                final_mel_tensor_basis = mel_power_tensor * m_norm_factor;

            } else if (mel_norm_mode_ == MelNormMode::MAGNITUDE_SUM) {
                if (verbose_) log("Applying MelNormMode::MAGNITUDE_SUM");

                torch::Tensor rfft_power_in = rfft_result[0].to(device_);
                if (rfft_power_in.dim() == 2 && rfft_power_in.size(1) == 1) {
                    rfft_power_in = rfft_power_in.squeeze(1);
                }
                torch::Tensor rfft_magnitude_in = contorchionist::core::util_conversions::powerToMagnitude(rfft_power_in);
                torch::Tensor mel_magnitude_bruto = contorchionist::core::util_conversions::powerToMagnitude(mel_power_tensor);

                float sum_rfft_mag_in = torch::sum(rfft_magnitude_in).item<float>();
                float sum_mel_mag_bruto = torch::sum(mel_magnitude_bruto).item<float>();

                float m_norm_factor_mag = 1.0f;
                if (sum_mel_mag_bruto != 0.0f) {
                    m_norm_factor_mag = sum_rfft_mag_in / sum_mel_mag_bruto;
                } else if (sum_rfft_mag_in != 0.0f && verbose_) {
                    log("Warning: Sum_mel_mag_bruto is 0 for MAGNITUDE_SUM but Sum_rfft_mag_in is not. Mel output will be zero magnitude.");
                }

                final_mel_tensor_basis = mel_magnitude_bruto * m_norm_factor_mag;
            }

            torch::Tensor final_output_tensor;
            bool basis_is_magnitude = (mel_norm_mode_ == MelNormMode::MAGNITUDE_SUM);

            if (basis_is_magnitude) {
                switch (output_format_) {
                    case contorchionist::core::util_conversions::SpectrumDataFormat::MAGPHASE:
                        final_output_tensor = final_mel_tensor_basis;
                        break;
                    case contorchionist::core::util_conversions::SpectrumDataFormat::POWERPHASE:
                        final_output_tensor = contorchionist::core::util_conversions::magnitudeToPower(final_mel_tensor_basis);
                        break;
                    case contorchionist::core::util_conversions::SpectrumDataFormat::DBPHASE:
                        final_output_tensor = contorchionist::core::util_conversions::powerToDb(contorchionist::core::util_conversions::magnitudeToPower(final_mel_tensor_basis));
                        break;
                    default:
                        final_output_tensor = final_mel_tensor_basis;
                        break;
                }
            } else {
                switch (output_format_) {
                    case contorchionist::core::util_conversions::SpectrumDataFormat::MAGPHASE:
                        final_output_tensor = contorchionist::core::util_conversions::powerToMagnitude(final_mel_tensor_basis);
                        break;
                    case contorchionist::core::util_conversions::SpectrumDataFormat::POWERPHASE:
                        final_output_tensor = final_mel_tensor_basis;
                        break;
                    case contorchionist::core::util_conversions::SpectrumDataFormat::DBPHASE:
                        final_output_tensor = contorchionist::core::util_conversions::powerToDb(final_mel_tensor_basis);
                        break;
                    default:
                        final_output_tensor = final_mel_tensor_basis;
                        break;
                }
            }

            final_output_tensor = final_output_tensor.to(torch::kCPU);
            if (!final_output_tensor.is_contiguous()) final_output_tensor = final_output_tensor.contiguous();

            output_frame1.resize(n_mels_);
            if (final_output_tensor.numel() == n_mels_) {
                std::memcpy(output_frame1.data(), final_output_tensor.data_ptr<T>(), n_mels_ * sizeof(T));
            } else {
                if (verbose_) log("ERROR: Size of final_output_tensor (" + std::to_string(final_output_tensor.numel()) + ") does not match n_mels (" + std::to_string(n_mels_) + "). Output will be empty.");
                output_frame1.clear();
            }

            circular_buffer_->discard(hop_length_);
            samples_processed_current_hop_ -= hop_length_;

            return true;
        }
        return false;
    }


    void set_device(torch::Device device) {
        if (device_ != device) {
            device_ = device;
            if (rfft_processor_) rfft_processor_->set_device(device_);
            if (mel_filterbank_initialized_) mel_filterbank_ = mel_filterbank_.to(device_);
            if (verbose_) log("Device set to: " + device_.str());
        }
    }
    torch::Device get_device() const { return device_; }

    void set_window_type(contorchionist::core::util_windowing::Type window_type) {
        if (window_type_ != window_type) {
            window_type_ = window_type;
            if (rfft_processor_) {
                rfft_processor_->set_window_type(window_type_);
                setup_rfft_processor();
            }
            if (verbose_) log("Window type set to: " + contorchionist::core::util_windowing::torch_window_type_to_string(window_type_));
        }
    }
    contorchionist::core::util_windowing::Type get_window_type() const { return window_type_; }

    void set_verbose(bool verbose) {
        if (verbose_ != verbose) {
            verbose_ = verbose;
            if (rfft_processor_) rfft_processor_->set_verbose(verbose_);
            if (verbose_) log("Verbose mode turned on.");
        }
    }
    bool is_verbose() const { return verbose_; }

    void set_sample_rate(float sample_rate) {
        if (sample_rate <= 0) throw std::invalid_argument("Sample rate must be positive.");
        if (sample_rate_ != sample_rate) {
            sample_rate_ = sample_rate;
            fmax_mel_actual_ = (fmax_mel_param_ <= 0) ? (sample_rate_ / 2.0f) : fmax_mel_param_;
            validate_parameters();
            if (rfft_processor_) rfft_processor_->set_sampling_rate(sample_rate_);
            setup_rfft_processor();
            initialize_mel_filterbank();
            if (verbose_) log("Sample rate set to: " + std::to_string(sample_rate_));
        }
    }
    float get_sample_rate() const { return sample_rate_; }

    void set_n_mels(int n_mels) {
        if (n_mels <= 0) throw std::invalid_argument("n_mels must be positive.");
        if (n_mels_ != n_mels) {
            n_mels_ = n_mels;
            validate_parameters();
            initialize_mel_filterbank();
            if (verbose_) log("n_mels set to: " + std::to_string(n_mels_));
        }
    }
    int get_n_mels() const { return n_mels_; }

    void set_n_fft(int n_fft) {
        if (n_fft <= 0) throw std::invalid_argument("n_fft must be positive.");
        if (n_fft_ != n_fft) {
            int old_n_fft = n_fft_;
            n_fft_ = n_fft;

            if (win_length_ == old_n_fft || win_length_ > n_fft_) {
                 int old_win_length = win_length_;
                 win_length_ = n_fft_;
                 if (verbose_) log("win_length auto-adjusted from " + std::to_string(old_win_length) + " to " + std::to_string(win_length_) + " due to n_fft change.");
            }
            validate_parameters();

            setup_rfft_processor();
            initialize_mel_filterbank();

            if (circular_buffer_ && circular_buffer_->getCapacity() != static_cast<size_t>(win_length_)) {
                 circular_buffer_ = std::make_unique<contorchionist::core::util_circbuffer::CircularBuffer<T>>(win_length_);
            }
            internal_frame_buffer_.resize(win_length_);

            if (verbose_) log("n_fft set to: " + std::to_string(n_fft_) + ". win_length is " + std::to_string(win_length_));
        }
    }
    int get_n_fft() const { return n_fft_; }

    void set_hop_length(int hop_length) {
        if (hop_length <= 0) throw std::invalid_argument("hop_length must be positive.");
        if (hop_length_ != hop_length) {
            hop_length_ = hop_length;
            validate_parameters();
            if (verbose_) log("hop_length set to: " + std::to_string(hop_length_));
        }
    }
    int get_hop_length() const { return hop_length_; }

    void set_win_length(int win_length_param) {
        int new_win_length = (win_length_param <= 0) ? n_fft_ : win_length_param;
        if (new_win_length <=0 || new_win_length > n_fft_) {
            throw std::invalid_argument("Proposed win_length (" + std::to_string(new_win_length) + ") must be positive and <= n_fft (" + std::to_string(n_fft_) + "). Original input: " + std::to_string(win_length_param));
        }

        if (win_length_ != new_win_length) {
            win_length_ = new_win_length;
            validate_parameters();
            setup_rfft_processor();

            if (circular_buffer_ && circular_buffer_->getCapacity() != static_cast<size_t>(win_length_)) {
                circular_buffer_ = std::make_unique<contorchionist::core::util_circbuffer::CircularBuffer<T>>(win_length_);
            }
            internal_frame_buffer_.resize(win_length_);

            if (verbose_) log("win_length set to: " + std::to_string(win_length_));
        }
    }
    int get_win_length() const { return win_length_; }

    void set_normalization_type(contorchionist::core::util_normalizations::NormalizationType norm_type) {
        if (rfft_normalization_type_ != norm_type) {
            rfft_normalization_type_ = norm_type;
            if (rfft_processor_) {
                setup_rfft_processor();
            }
            if (verbose_) log("RFFT normalization type set to: " + contorchionist::core::util_normalizations::normalization_type_to_string(norm_type));
        }
    }
    contorchionist::core::util_normalizations::NormalizationType get_normalization_type() const { return rfft_normalization_type_; }

    void set_output_format(contorchionist::core::util_conversions::SpectrumDataFormat format) {
        if (output_format_ != format) {
            output_format_ = format;
            if (verbose_) log("Output format set to: " + contorchionist::core::util_conversions::spectrum_data_format_to_string(format));
        }
    }
    contorchionist::core::util_conversions::SpectrumDataFormat get_output_format() const {
        return output_format_;
    }

    void set_fmin_mel(float fmin_mel) {
        if (fmin_mel < 0) throw std::invalid_argument("fmin_mel must be non-negative.");
        if (fmin_mel_ != fmin_mel) {
            fmin_mel_ = fmin_mel;
            validate_parameters();
            initialize_mel_filterbank();
            if (verbose_) log("fmin_mel set to: " + std::to_string(fmin_mel_));
        }
    }
    float get_fmin_mel() const { return fmin_mel_; }

    void set_fmax_mel(float fmax_mel_param_val) {
        if (fmax_mel_param_ != fmax_mel_param_val) {
            fmax_mel_param_ = fmax_mel_param_val;
            fmax_mel_actual_ = (fmax_mel_param_ <= 0 && sample_rate_ > 0) ? (sample_rate_ / 2.0f) : fmax_mel_param_;
            validate_parameters();
            initialize_mel_filterbank();
            if (verbose_) log("fmax_mel_param set to: " + std::to_string(fmax_mel_param_) + ", actual fmax used: " + std::to_string(fmax_mel_actual_));
        }
    }
    float get_fmax_mel() const {
        return fmax_mel_actual_;
    }

    void set_mel_formula(contorchionist::core::util_conversions::MelFormulaType formula) {
        if (mel_formula_ != formula) {
            mel_formula_ = formula;
            initialize_mel_filterbank();
            if (verbose_) log("Mel formula set to: " + contorchionist::core::util_conversions::mel_formula_type_to_string(mel_formula_));
        }
    }
    contorchionist::core::util_conversions::MelFormulaType get_mel_formula() const {
        return mel_formula_;
    }

    void set_filterbank_norm(const std::string& norm) {
        if (filterbank_norm_ != norm) {
            filterbank_norm_ = norm;
            initialize_mel_filterbank();
            if (verbose_) log("Filterbank norm set to: " + filterbank_norm_);
        }
    }
    std::string get_filterbank_norm() const { return filterbank_norm_; }

    void set_mel_norm_mode(MelNormMode mode) {
        if (mel_norm_mode_ != mode) {
            mel_norm_mode_ = mode;
            if (verbose_) log("Mel normalization mode set to: " + mel_norm_mode_to_string(mode));
        }
    }
    MelNormMode get_mel_norm_mode() const { return mel_norm_mode_; }

    torch::Tensor get_mel_filterbank() const {
        if (mel_filterbank_initialized_) {
            return mel_filterbank_.clone();
        }
        if (verbose_) log("Warning: get_mel_filterbank() called but filterbank not initialized. Returning empty tensor.");
        return torch::empty({0}, torch::TensorOptions().device(device_).dtype(torch::kFloat));
    }

    void clear_buffer() {
        if (circular_buffer_) {
            circular_buffer_->clear();
            samples_processed_current_hop_ = 0;
            if (verbose_) log("Circular buffer cleared.");
        }
    }

private:
    void log(const std::string& message) const {
        if (verbose_) {
            std::cout << "[MelSpectrogramProcessor LOG] " << message << std::endl;
        }
    }
    void validate_parameters() {
        if (sample_rate_ <= 0) throw std::invalid_argument("Sample rate must be positive. Got: " + std::to_string(sample_rate_));
        if (n_mels_ <= 0) throw std::invalid_argument("n_mels must be positive. Got: " + std::to_string(n_mels_));
        if (n_fft_ <= 0) throw std::invalid_argument("n_fft must be positive. Got: " + std::to_string(n_fft_));
        if (hop_length_ <= 0) throw std::invalid_argument("hop_length must be positive. Got: " + std::to_string(hop_length_));
        if (win_length_ <= 0 || win_length_ > n_fft_) {
            throw std::invalid_argument("Resolved win_length ("+ std::to_string(win_length_) +") must be positive and <= n_fft ("+ std::to_string(n_fft_) +").");
        }
        if (fmin_mel_ < 0) throw std::invalid_argument("fmin_mel must be non-negative. Got: " + std::to_string(fmin_mel_));

        float temp_fmax_actual = (fmax_mel_param_ <= 0 && sample_rate_ > 0) ? (sample_rate_ / 2.0f) : fmax_mel_param_;
        if (temp_fmax_actual <= fmin_mel_ && sample_rate_ > 0) {
             throw std::invalid_argument("fmax_mel (" + std::to_string(temp_fmax_actual) + ") must be greater than fmin_mel (" + std::to_string(fmin_mel_) + ").");
        }
    }

    void setup_rfft_processor() {
        if (!rfft_processor_) {
            rfft_processor_ = std::make_unique<contorchionist::core::ap_rfft::RFFTProcessor<T>>(
                device_, window_type_, true, verbose_,
                rfft_normalization_type_,
                contorchionist::core::util_conversions::SpectrumDataFormat::POWERPHASE,
                sample_rate_
            );
        } else {
            rfft_processor_->set_device(device_);
            rfft_processor_->set_window_type(window_type_);
            rfft_processor_->enable_windowing(true);
            rfft_processor_->set_verbose(verbose_);
            rfft_processor_->set_normalization_type(rfft_normalization_type_);
            rfft_processor_->set_output_format(contorchionist::core::util_conversions::SpectrumDataFormat::POWERPHASE);
            rfft_processor_->set_sampling_rate(sample_rate_);
        }
        rfft_processor_->set_normalization(1, 2, n_fft_, rfft_normalization_type_, sample_rate_, 1.0f);
        rfft_processor_->initialize_window(win_length_);

        if (verbose_) log("RFFTProcessor (re)configured for MelSpectrogramProcessor.");
    }


    void initialize_mel_filterbank() {
        if (sample_rate_ <= 0) {
             throw std::runtime_error("Sample rate (sr) must be positive to initialize mel filterbank.");
        }
        fmax_mel_actual_ = (fmax_mel_param_ <= 0) ? (sample_rate_ / 2.0f) : fmax_mel_param_;

        validate_parameters();

        try {
            mel_filterbank_ = create_mel_filterbank(
                sample_rate_, n_fft_, n_mels_, fmin_mel_, fmax_mel_actual_,
                mel_formula_, filterbank_norm_, device_
            );
            mel_filterbank_initialized_ = true;
            if (verbose_) {
                log("Mel filterbank (re)initialized: " + std::to_string(n_mels_) +
                    " filters, for n_fft=" + std::to_string(n_fft_) +
                    ", sr=" + std::to_string(sample_rate_) +
                    ", fmin=" + std::to_string(fmin_mel_) +
                    ", fmax_actual=" + std::to_string(fmax_mel_actual_) +
                    ", mel_formula=" + contorchionist::core::util_conversions::mel_formula_type_to_string(mel_formula_) +
                    ", filterbank_norm=" + filterbank_norm_);
            }
        } catch (const std::exception& e) {
            mel_filterbank_initialized_ = false;
            std::string error_msg = "Failed to initialize mel filterbank: " + std::string(e.what());
            if (verbose_) log("ERROR: " + error_msg);
            throw std::runtime_error(error_msg);
        }
    }

    torch::Device device_;
    contorchionist::core::util_windowing::Type window_type_;
    bool verbose_;
    
    float sample_rate_;
    int n_mels_;
    int n_fft_;
    int hop_length_;
    int win_length_;
    
    float fmin_mel_;
    float fmax_mel_param_;
    float fmax_mel_actual_;
    contorchionist::core::util_conversions::MelFormulaType mel_formula_;
    std::string filterbank_norm_;
    MelNormMode mel_norm_mode_;
    contorchionist::core::util_normalizations::NormalizationType rfft_normalization_type_;
    contorchionist::core::util_conversions::SpectrumDataFormat output_format_;

    std::unique_ptr<contorchionist::core::util_circbuffer::CircularBuffer<T>> circular_buffer_;
    std::vector<T> internal_frame_buffer_;
    int samples_processed_current_hop_;

    std::unique_ptr<contorchionist::core::ap_rfft::RFFTProcessor<T>> rfft_processor_;
    torch::Tensor mel_filterbank_;
    bool mel_filterbank_initialized_;
};

        } // namespace ap_melspectrogram
    } // namespace core
} // namespace contorchionist

#endif // CORE_AP_MELSPECTROGRAM_H

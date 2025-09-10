#ifndef CORE_AP_MFCC_H
#define CORE_AP_MFCC_H

#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>

#include "contorchionist_core/contorchionist_core_export.h"
#include "core_ap_dct.h"
#include "core_ap_melspectrogram.h"
#include "core_util_conversions.h"
#include "core_util_windowing.h"
#include "core_util_normalizations.h"

namespace contorchionist {
    namespace core {
        namespace ap_mfcc {

template<typename SampleType = float>
class CONTORCHIONIST_CORE_EXPORT MFCCProcessor {
public:
    // Constructor with all mel spectrogram and MFCC parameters
    MFCCProcessor(
        // MFCC specific parameters
        int n_mfcc = 13,
        int first_mfcc = 0,
        int dct_type = 2,
        const std::string& dct_norm = "ortho",
        
        // Mel Spectrogram parameters
        int n_fft = 2048,
        int hop_length = 512,
        int win_length = -1,
        contorchionist::core::util_windowing::Type window_type = contorchionist::core::util_windowing::Type::HANN,
        contorchionist::core::util_normalizations::NormalizationType rfft_norm = contorchionist::core::util_normalizations::NormalizationType::NONE,
        contorchionist::core::util_conversions::SpectrumDataFormat output_unit = contorchionist::core::util_conversions::SpectrumDataFormat::POWERPHASE,
        SampleType sample_rate = static_cast<SampleType>(48000.0),
        int n_mels = 40,
        SampleType fmin_mel = static_cast<SampleType>(20.0),
        SampleType fmax_mel = static_cast<SampleType>(20000.0),
        contorchionist::core::util_conversions::MelFormulaType mel_formula = contorchionist::core::util_conversions::MelFormulaType::SLANEY,
        const std::string& filterbank_norm = "slaney",
        contorchionist::core::ap_melspectrogram::MelNormMode mel_norm_mode = contorchionist::core::ap_melspectrogram::MelNormMode::NONE,
        
        // Common parameters
        torch::Device device = torch::kCPU,
        bool verbose = false
    );

    ~MFCCProcessor();

    // Process audio samples directly to MFCC coefficients
    bool process(const SampleType* input_samples, int n_samples, std::vector<SampleType>& mfcc_output);
    
    // Process mel spectrogram tensor to MFCC (for backward compatibility)
    torch::Tensor process_mel_tensor(const torch::Tensor& mel_spectrogram_db);

    // Device management
    void set_device(torch::Device device);
    torch::Device get_device() const;

    // Verbosity control
    void set_verbose(bool verbose);
    bool is_verbose() const;

    // MFCC parameters
    void set_n_mfcc(int n_mfcc);
    int get_n_mfcc() const;

    void set_first_mfcc(int first_mfcc);
    int get_first_mfcc() const;

    void set_dct_type(int dct_type);
    int get_dct_type() const;

    void set_dct_norm(const std::string& dct_norm);
    std::string get_dct_norm() const;

    // Mel Spectrogram parameters
    void set_n_mels(int n_mels);
    int get_n_mels() const;

    void set_n_fft(int n_fft);
    int get_n_fft() const;

    void set_hop_length(int hop_length);
    int get_hop_length() const;

    void set_win_length(int win_length);
    int get_win_length() const;

    void set_window_type(contorchionist::core::util_windowing::Type window_type);
    contorchionist::core::util_windowing::Type get_window_type() const;

    void set_rfft_norm(contorchionist::core::util_normalizations::NormalizationType rfft_norm);
    contorchionist::core::util_normalizations::NormalizationType get_rfft_norm() const;

    void set_output_unit(contorchionist::core::util_conversions::SpectrumDataFormat output_unit);
    contorchionist::core::util_conversions::SpectrumDataFormat get_output_unit() const;

    void set_sample_rate(SampleType sample_rate);
    SampleType get_sample_rate() const;

    void set_fmin_mel(SampleType fmin_mel);
    SampleType get_fmin_mel() const;

    void set_fmax_mel(SampleType fmax_mel);
    SampleType get_fmax_mel() const;

    void set_mel_formula(contorchionist::core::util_conversions::MelFormulaType mel_formula);
    contorchionist::core::util_conversions::MelFormulaType get_mel_formula() const;

    void set_filterbank_norm(const std::string& filterbank_norm);
    std::string get_filterbank_norm() const;

    void set_mel_norm_mode(contorchionist::core::ap_melspectrogram::MelNormMode mel_norm_mode);
    contorchionist::core::ap_melspectrogram::MelNormMode get_mel_norm_mode() const;

private:
    void log(const std::string& message) const;
    void validate_parameters();
    void configure_mel_processor();

    torch::Device device_;
    bool verbose_;

    // MFCC parameters
    int n_mfcc_;
    int first_mfcc_;
    int dct_type_;
    std::string dct_norm_;

    // Mel Spectrogram processor and parameters
    std::unique_ptr<contorchionist::core::ap_melspectrogram::MelSpectrogramProcessor<SampleType>> mel_processor_;
    std::vector<float> mel_frame_data_buffer_;
    std::vector<float> phase_data_buffer_;
    std::vector<float> float_samples_buffer_;
    
    int n_fft_;
    int hop_length_;
    int win_length_;
    contorchionist::core::util_windowing::Type window_type_;
    contorchionist::core::util_normalizations::NormalizationType rfft_norm_;
    contorchionist::core::util_conversions::SpectrumDataFormat output_unit_;
    SampleType sample_rate_;
    int n_mels_;
    SampleType fmin_mel_;
    SampleType fmax_mel_;
    contorchionist::core::util_conversions::MelFormulaType mel_formula_;
    std::string filterbank_norm_;
    contorchionist::core::ap_melspectrogram::MelNormMode mel_norm_mode_;
};

// Type aliases for common use cases
using MFCCProcessorFloat = MFCCProcessor<float>;
using MFCCProcessorDouble = MFCCProcessor<double>;

// Template implementations (inline)
template<typename SampleType>
MFCCProcessor<SampleType>::MFCCProcessor(
    int n_mfcc, int first_mfcc, int dct_type, const std::string& dct_norm,
    int n_fft, int hop_length, int win_length,
    contorchionist::core::util_windowing::Type window_type,
    contorchionist::core::util_normalizations::NormalizationType rfft_norm,
    contorchionist::core::util_conversions::SpectrumDataFormat output_unit,
    SampleType sample_rate, int n_mels, SampleType fmin_mel, SampleType fmax_mel,
    contorchionist::core::util_conversions::MelFormulaType mel_formula,
    const std::string& filterbank_norm,
    contorchionist::core::ap_melspectrogram::MelNormMode mel_norm_mode,
    torch::Device device, bool verbose
) : device_(device), verbose_(verbose),
    n_mfcc_(n_mfcc), first_mfcc_(first_mfcc), dct_type_(dct_type), dct_norm_(dct_norm),
    n_fft_(n_fft), hop_length_(hop_length), win_length_(win_length),
    window_type_(window_type), rfft_norm_(rfft_norm), output_unit_(output_unit),
    sample_rate_(sample_rate), n_mels_(n_mels), fmin_mel_(fmin_mel), fmax_mel_(fmax_mel),
    mel_formula_(mel_formula), filterbank_norm_(filterbank_norm), mel_norm_mode_(mel_norm_mode)
{
    validate_parameters();
    configure_mel_processor();
    
    if (verbose_) {
        log("MFCCProcessor created. n_mfcc=" + std::to_string(n_mfcc_) +
            ", first_mfcc=" + std::to_string(first_mfcc_) +
            ", dct_type=" + std::to_string(dct_type_) +
            ", dct_norm=" + dct_norm_ +
            ", n_mels=" + std::to_string(n_mels_) +
            ", n_fft=" + std::to_string(n_fft_) +
            ", hop_length=" + std::to_string(hop_length_) +
            ", sample_rate=" + std::to_string(sample_rate_));
    }
}

template<typename SampleType>
MFCCProcessor<SampleType>::~MFCCProcessor() {
    if (verbose_) {
        log("MFCCProcessor destroyed");
    }
}

template<typename SampleType>
void MFCCProcessor<SampleType>::configure_mel_processor() {
    try {
        if (sample_rate_ <= 0 || n_fft_ <= 0 || hop_length_ <= 0) {
            if (verbose_) {
                log("Warning: Invalid parameters for mel processor configuration. sample_rate=" + 
                    std::to_string(sample_rate_) + ", n_fft=" + std::to_string(n_fft_) + 
                    ", hop_length=" + std::to_string(hop_length_));
            }
            mel_processor_.reset();
            return;
        }

        mel_processor_ = std::make_unique<contorchionist::core::ap_melspectrogram::MelSpectrogramProcessor<SampleType>>(
            n_fft_, hop_length_, win_length_, window_type_, rfft_norm_, output_unit_,
            static_cast<float>(sample_rate_), n_mels_, static_cast<float>(fmin_mel_), static_cast<float>(fmax_mel_),
            mel_formula_, filterbank_norm_, mel_norm_mode_, device_, verbose_
        );

        mel_frame_data_buffer_.resize(n_mels_);
        phase_data_buffer_.resize(n_mels_);

        if (verbose_) {
            log("Mel spectrogram processor configured successfully");
        }
    } catch (const std::exception& e) {
        if (verbose_) {
            log("Error configuring mel processor: " + std::string(e.what()));
        }
        mel_processor_.reset();
        throw;
    }
}

template<typename SampleType>
bool MFCCProcessor<SampleType>::process(const SampleType* input_samples, int n_samples, std::vector<SampleType>& mfcc_output) {
    if (!mel_processor_) {
        if (verbose_) log("Mel processor not configured");
        return false;
    }

    if (verbose_) {
        log("Processing MFCC with " + std::to_string(n_samples) + " input samples");
    }

    try {
        float_samples_buffer_.assign(input_samples, input_samples + n_samples);

        if (mel_processor_->process(float_samples_buffer_.data(), n_samples, mel_frame_data_buffer_, phase_data_buffer_)) {
            if (!mel_frame_data_buffer_.empty()) {
                torch::Tensor mel_tensor_db = torch::from_blob(
                    mel_frame_data_buffer_.data(),
                    {static_cast<long>(mel_frame_data_buffer_.size())},
                    torch::kFloat32
                ).to(device_);

                torch::Tensor mfcc_tensor = process_mel_tensor(mel_tensor_db);

                mfcc_tensor = mfcc_tensor.to(torch::kCPU);
                auto* data_ptr = mfcc_tensor.data_ptr<float>();
                mfcc_output.assign(data_ptr, data_ptr + mfcc_tensor.numel());

                if (verbose_) {
                    log("MFCC processing completed, output size: " + std::to_string(mfcc_output.size()));
                }
                return true;
            }
        }
        
        return false;
    } catch (const std::exception& e) {
        if (verbose_) {
            log("Error in MFCC processing: " + std::string(e.what()));
        }
        throw;
    }
}

template<typename SampleType>
torch::Tensor MFCCProcessor<SampleType>::process_mel_tensor(const torch::Tensor& mel_spectrogram_db) {
    if (verbose_) {
        log("Processing MFCC from mel tensor with size: " + std::to_string(mel_spectrogram_db.numel()));
    }

    // Verify tensor validity
    if (!mel_spectrogram_db.defined()) {
        throw std::runtime_error("Input tensor is not defined");
    }

    if (mel_spectrogram_db.numel() == 0) {
        if (verbose_) log("Warning: Input tensor is empty");
        return torch::empty({n_mfcc_}, mel_spectrogram_db.options());
    }

    if (mel_spectrogram_db.size(-1) != n_mels_) {
        auto msg = "Input tensor last dimension (" + std::to_string(mel_spectrogram_db.size(-1)) + ") does not match n_mels (" + std::to_string(n_mels_) + ")";
        throw std::runtime_error(msg);
    }

    torch::Tensor mfcc;

    try {
        // Librosa applies DCT on the last axis. Here we assume the input is [..., n_mels]
        // The dct functions in this project also apply on the last axis.
        if (dct_type_ == 2) {
            int n_coeffs_to_compute = n_mfcc_ + first_mfcc_;
            torch::Tensor dct_output = contorchionist::core::ap_dct::DCT<SampleType>::dctType2(mel_spectrogram_db, n_coeffs_to_compute, dct_norm_, device_);
            mfcc = dct_output.slice(-1, first_mfcc_, n_coeffs_to_compute);
        } else {
            // In the future, other DCT types can be added here.
            // For now, we stick to what librosa uses.
            throw std::runtime_error("MFCCProcessor currently only supports DCT type 2.");
        }

        if (verbose_) {
            log("MFCC processing completed, output size: " + std::to_string(mfcc.numel()));
        }
    } catch (const std::exception& e) {
        if (verbose_) {
            log("Error in MFCC processing: " + std::string(e.what()));
        }
        throw;
    }

    return mfcc;
}

template<typename SampleType>
void MFCCProcessor<SampleType>::log(const std::string& message) const {
    if (verbose_) {
        std::cout << "[MFCCProcessor LOG] " << message << std::endl;
    }
}

template<typename SampleType>
void MFCCProcessor<SampleType>::validate_parameters() {
    if (n_mels_ <= 0) throw std::invalid_argument("n_mels must be positive.");
    if (n_mfcc_ <= 0) throw std::invalid_argument("n_mfcc must be positive.");
    if (first_mfcc_ < 0) throw std::invalid_argument("first_mfcc must be non-negative.");
    if (n_mfcc_ + first_mfcc_ > n_mels_) {
        log("Warning: n_mfcc + first_mfcc (" + std::to_string(n_mfcc_ + first_mfcc_) + ") is greater than n_mels (" + std::to_string(n_mels_) + ").");
    }
    if (dct_type_ != 2) {
        // For now, we are focusing on librosa compatibility which uses DCT type 2
        log("Warning: dct_type " + std::to_string(dct_type_) + " is not the librosa default (2).");
    }
}

// Define all the getter/setter template implementations here (shortened for brevity)
template<typename SampleType>
void MFCCProcessor<SampleType>::set_device(torch::Device device) {
    if (device_ != device) {
        device_ = device;
        if (mel_processor_) mel_processor_->set_device(device_);
        if (verbose_) log("Device set to: " + device_.str());
    }
}

template<typename SampleType>
torch::Device MFCCProcessor<SampleType>::get_device() const { return device_; }

template<typename SampleType>
void MFCCProcessor<SampleType>::set_verbose(bool verbose) {
    if (verbose_ != verbose) {
        verbose_ = verbose;
        if (mel_processor_) mel_processor_->set_verbose(verbose_);
        if (verbose_) log("Verbose mode turned on.");
    }
}

template<typename SampleType>
bool MFCCProcessor<SampleType>::is_verbose() const { return verbose_; }

template<typename SampleType>
void MFCCProcessor<SampleType>::set_sample_rate(SampleType sample_rate) {
    if (sample_rate <= 0) throw std::invalid_argument("sample_rate must be positive.");
    if (sample_rate_ != sample_rate) {
        sample_rate_ = sample_rate;
        configure_mel_processor();
        if (verbose_) log("sample_rate set to: " + std::to_string(sample_rate_));
    }
}

template<typename SampleType>
SampleType MFCCProcessor<SampleType>::get_sample_rate() const { return sample_rate_; }

template<typename SampleType>
void MFCCProcessor<SampleType>::set_n_mfcc(int n_mfcc) {
    if (n_mfcc <= 0) throw std::invalid_argument("n_mfcc must be positive.");
    if (n_mfcc_ != n_mfcc) {
        n_mfcc_ = n_mfcc;
        if (verbose_) log("n_mfcc set to: " + std::to_string(n_mfcc_));
    }
}

template<typename SampleType>
int MFCCProcessor<SampleType>::get_n_mfcc() const { return n_mfcc_; }

template<typename SampleType>
void MFCCProcessor<SampleType>::set_first_mfcc(int first_mfcc) {
    if (first_mfcc < 0) throw std::invalid_argument("first_mfcc must be non-negative.");
    if (first_mfcc_ != first_mfcc) {
        first_mfcc_ = first_mfcc;
        validate_parameters();
        if (verbose_) log("first_mfcc set to: " + std::to_string(first_mfcc_));
    }
}

template<typename SampleType>
int MFCCProcessor<SampleType>::get_first_mfcc() const { return first_mfcc_; }

template<typename SampleType>
void MFCCProcessor<SampleType>::set_dct_type(int dct_type) {
    if (dct_type <= 0) throw std::invalid_argument("dct_type must be positive.");
    if (dct_type_ != dct_type) {
        dct_type_ = dct_type;
        if (verbose_) log("dct_type set to: " + std::to_string(dct_type_));
    }
}

template<typename SampleType>
int MFCCProcessor<SampleType>::get_dct_type() const { return dct_type_; }

template<typename SampleType>
void MFCCProcessor<SampleType>::set_dct_norm(const std::string& dct_norm) {
    if (dct_norm_ != dct_norm) {
        dct_norm_ = dct_norm;
        if (verbose_) log("dct_norm set to: " + dct_norm_);
    }
}

template<typename SampleType>
std::string MFCCProcessor<SampleType>::get_dct_norm() const { return dct_norm_; }

template<typename SampleType>
void MFCCProcessor<SampleType>::set_output_unit(contorchionist::core::util_conversions::SpectrumDataFormat output_unit) {
    if (output_unit_ != output_unit) {
        output_unit_ = output_unit;
        if (mel_processor_) mel_processor_->set_output_format(output_unit_);
        if (verbose_) log("output_unit set to: " + contorchionist::core::util_conversions::spectrum_data_format_to_string(output_unit_));
    }
}

template<typename SampleType>
contorchionist::core::util_conversions::SpectrumDataFormat MFCCProcessor<SampleType>::get_output_unit() const { return output_unit_; }

template<typename SampleType>
void MFCCProcessor<SampleType>::set_mel_formula(contorchionist::core::util_conversions::MelFormulaType mel_formula) {
    if (mel_formula_ != mel_formula) {
        mel_formula_ = mel_formula;
        if (mel_processor_) mel_processor_->set_mel_formula(mel_formula_);
        if (verbose_) log("mel_formula set to: " + contorchionist::core::util_conversions::mel_formula_type_to_string(mel_formula_));
    }
}

template<typename SampleType>
contorchionist::core::util_conversions::MelFormulaType MFCCProcessor<SampleType>::get_mel_formula() const { return mel_formula_; }

template<typename SampleType>
void MFCCProcessor<SampleType>::set_filterbank_norm(const std::string& filterbank_norm) {
    if (filterbank_norm_ != filterbank_norm) {
        filterbank_norm_ = filterbank_norm;
        if (mel_processor_) mel_processor_->set_filterbank_norm(filterbank_norm_);
        if (verbose_) log("filterbank_norm set to: " + filterbank_norm_);
    }
}

template<typename SampleType>
std::string MFCCProcessor<SampleType>::get_filterbank_norm() const { return filterbank_norm_; }

template<typename SampleType>
void MFCCProcessor<SampleType>::set_mel_norm_mode(contorchionist::core::ap_melspectrogram::MelNormMode mel_norm_mode) {
    if (mel_norm_mode_ != mel_norm_mode) {
        mel_norm_mode_ = mel_norm_mode;
        if (mel_processor_) mel_processor_->set_mel_norm_mode(mel_norm_mode_);
        if (verbose_) log("mel_norm_mode set to: " + contorchionist::core::ap_melspectrogram::mel_norm_mode_to_string(mel_norm_mode_));
    }
}

template<typename SampleType>
contorchionist::core::ap_melspectrogram::MelNormMode MFCCProcessor<SampleType>::get_mel_norm_mode() const { return mel_norm_mode_; }

// Add more template implementations as needed...

        } // namespace ap_mfcc
    } // namespace core
} // namespace contorchionist

#endif // CORE_AP_MFCC_H

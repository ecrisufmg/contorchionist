#ifndef CORE_AP_RMSOVERLAP_H
#define CORE_AP_RMSOVERLAP_H

#include <vector>
#include <string>
#include <cmath> // For std::sqrt, std::pow, std::sin, std::cos, std::log2, std::ceil
#include <numeric> // For std::accumulate (potentially)
#include <stdexcept> // For std::runtime_error
#include <thread> // Added for std::thread
#include <mutex> // Added for std::mutex
#include <condition_variable> // Added for std::condition_variable
#include <atomic> // Added for std::atomic
#include <algorithm> // For std::fill, std::copy, std::max, std::min
#include <torch/torch.h> // Added for torch::TensorOptions, etc.
#include <iostream>  // For potential debugging (post equivalent)
#include <cstring>   // For strcmp (if absolutely needed, prefer std::string)

#include "core_util_windowing.h" // Include the windowing header
#include "core_util_circbuffer.h" // Include the circular buffer header

// Forward declaration for testing purposes if needed
// class RMSAnalyzerTester;

// For temporary debugging of core library logic
#include <cstdio>  // For fprintf, stderr, fflush
#include <cstdarg> // For va_list, va_start, va_end, vfprintf

// --- Temporary Core Logging ---
// This will be used to ensure logs from core_ap_rmsoverlap are visible
// by printing to stderr, which usually appears in the PD console.
static inline void TempCoreLog(const char* format, ...) {
    fprintf(stderr, "[RMSOverlap_CORE_DEBUG] "); // Distinct prefix
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fprintf(stderr, "\n"); // Newline
    fflush(stderr);      // Ensure it's flushed immediately
}

// Undefine RMS_POST if it was defined before, then redefine to use TempCoreLog
#ifdef RMS_POST
#undef RMS_POST
#endif
#define RMS_POST(...) do { if (verbose_) TempCoreLog(__VA_ARGS__); } while(0)
// --- End Temporary Core Logging ---

namespace contorchionist {
namespace core {
namespace ap_rmsoverlap {

template<typename T = float>
class RMSOverlap {
public:
    // Keep NormalizationType here as it's specific to RMS calculation context
    enum class NormalizationType {
        WINDOW_OVERLAP_RMS,  // Corresponds to original case 0
        WINDOW_OVERLAP_MEAN, // Corresponds to original case 1
        WINDOW_OVERLAP_VALS, // Corresponds to original case 2 (1.0 / sum_of_overlapping_windows)
        OVERLAP_INVERSE,     // Corresponds to original case 3 (1.0 / num_overlaps)
        FIXED_MULTIPLIER,    // Corresponds to original case 4
        NONE                 // Corresponds to original case 5 (factor of 1.0)
    };

    RMSOverlap(int initialWindowSize = 1024,
               int initialHopSize = 512,
               contorchionist::core::util_windowing::Type initialWinType = contorchionist::core::util_windowing::Type::RECTANGULAR, // Changed type
               float zeroPaddingFactor = 0.0f,
               contorchionist::core::util_windowing::Alignment initialWinAlign = contorchionist::core::util_windowing::Alignment::LEFT, // Changed type
               NormalizationType initialNormType = NormalizationType::WINDOW_OVERLAP_RMS,
               float fixedNormMultiplier = 1.0f,
               int initialBlockSize = 64,
               bool verbose = false);

    ~RMSOverlap();

    // Main processing function
    // Input: block of audio samples
    // Output: block of RMS (or processed) samples
    // Returns true if output contains valid data, false otherwise (e.g. not enough samples yet)
    bool process(const T* input, T* output, int inputBlockSize);
    void post_input_data(const T* data, int num_samples);

    // Configuration methods
    void setWindowSize(int newWindowSize);
    void setHopSize(int newHopSize);
    void setWindowType(contorchionist::core::util_windowing::Type newType);
    void setZeroPadding(float factor, contorchionist::core::util_windowing::Alignment alignment);
    void setZeroPaddingSamples(int samples, contorchionist::core::util_windowing::Alignment alignment);
    void setNormalization(NormalizationType newType, float fixedMultiplier = 1.0f);
    void setBlockSize(int newBlockSize); // If block size can change dynamically

    // Getter methods (optional, for debugging or PD messages)
    int getWindowSize() const { return windowSize_; }
    int getHopSize() const { return hopSize_; }
    int getBlockSize() const { return internalBlockSize_; }
    int getNumFramesProducedInLastCall() const { return framesProducedInLastCall_; }
    int getNumOverlaps() const { return numOverlaps_; }
    const torch::Tensor& getWindowFunction() const { return window_; }
    const torch::Tensor& getWindowOverlapSum() const { return windowOverlapSum_; } // Changed to Tensor
    const torch::Tensor& getNormalizationBuffer() const { return normalizationBuffer_; } // Changed to Tensor
    contorchionist::core::util_windowing::Type getWindowType() const { return windowType_; }
    contorchionist::core::util_windowing::Alignment getWindowAlignment() const { return windowAlignment_; }
    int getZeroPaddingSamples() const { return zeroPaddingSamples_; } // Getter for zero padding samples
    NormalizationType getNormalizationType() const { return normalizationType_; } // Getter for normalization type
    int64_t getSamplesInCircularBuffer() const { return samples_in_buffer_.load(); } // Getter for samples in buffer

    // Helper function for enum to string conversion (for dumping parameters)
    static std::string toString(NormalizationType type);

private:
    // friend class RMSAnalyzerTester; // For allowing a test class to access private members

    void allocateBuffers();
    void freeBuffers();
    // void buildWindow(); // This logic is now in Windowing::generateWindow
    void calculateNormalizationFactors();
    static NormalizationType stringToNormalizationType(const std::string& strNorm); // Keep this one

    // Core parameters
    int windowSize_;
    int hopSize_;
    int numOverlaps_;
    int zeroPaddingSamples_;
    contorchionist::core::util_windowing::Type windowType_;
    contorchionist::core::util_windowing::Alignment windowAlignment_;
    NormalizationType normalizationType_;
    float fixedNormalizationMultiplier_;
    int internalBlockSize_; // Expected block size from environment (e.g., PD)
    int framesProducedInLastCall_; // To store the number of frames output by the last process() call

    // Buffers
    contorchionist::core::util_circbuffer::CircularBuffer<T> circularBuffer_; // Template-based declaration
    // int circularBufferWriteIndex_; // To be replaced by read_idx_ and write_idx_
    torch::Tensor window_;                 // The actual window function
    torch::Tensor layerRMSValues_;         // Stores RMS for each overlap layer
    torch::Tensor windowOverlapSum_;       // Sum of overlapping window values (for normalization type 2)
    torch::Tensor normalizationBuffer_;    // Per-sample normalization factors for output stage
    torch::Tensor sumBuffer_;              // Temporary buffer for summing windowed layer RMS values

    // State variables for processing
    int samplesSinceLastOutput_; // To manage outputting at hop rate if needed. May need review with new input model.
    int layerCountForInitialNorm_; // Counter for initial normalization adjustment like in C code
    int synthesis_reference_point_ = 0; // For aligning output synthesis

    // Threading-related members
    std::mutex circular_buffer_mutex_;
    std::condition_variable data_available_cv_;
    std::condition_variable space_available_cv_;

    int64_t read_idx_;
    int64_t write_idx_;
    std::atomic<int64_t> samples_in_buffer_;
    std::atomic<bool> running_; // To signal termination
    bool verbose_; // Enable verbose debug output

    // NEWLY ADDED MEMBERS for refactored process()
    T currentRMSValue_;
    int samplesUntilNextHop_;
    torch::Tensor analysisBuffer_; // Buffer to hold data for a single RMS calculation
    std::mutex process_mutex_;    // Mutex to protect processing logic if called concurrently (though likely not from PD)
    std::atomic<bool> isInitialized_;  // Flag to indicate if initial parameters are set

    // Helper for string conversion to enum
    static NormalizationType parseNormalizationType(const std::string& normStr); // Keep this one

    void reinitialize(); // Called when major parameters change

public:
    // Implementation of reset() is inline as this is a header-only class.
    void reset() {
        // Lock to ensure thread safety, although this is typically called from the main thread during DSP setup.
        std::lock_guard<std::mutex> lock(process_mutex_);

        // 1. Clear the main circular buffer of any leftover audio data.
        circularBuffer_.clear();

        // 2. Reset all processing state variables to their initial conditions.
        samplesSinceLastOutput_ = 0;
        layerCountForInitialNorm_ = 0;
        synthesis_reference_point_ = 0;
        currentRMSValue_ = 0.0f;
        samplesUntilNextHop_ = 0; // Process the first block of samples immediately.
        samples_in_buffer_.store(0);
        read_idx_ = 0;
        write_idx_ = 0;

        // 3. Zero out any temporary buffers (tensors) that hold intermediate values.
        if (layerRMSValues_.numel() > 0) {
            layerRMSValues_.fill_(0.0f);
        }
        if (sumBuffer_.numel() > 0) {
            sumBuffer_.fill_(0.0f);
        }
        if (analysisBuffer_.numel() > 0) {
            analysisBuffer_.fill_(0.0f);
        }

        // Note: Configuration-dependent buffers like window_, windowOverlapSum_,
        // and normalizationBuffer_ are NOT reset. They are only recalculated
        // in reinitialize() when parameters like window size are changed by the user.
    }
};

// ===================================================================
// TEMPLATE IMPLEMENTATION
// ===================================================================

// Constructor
template<typename T>
RMSOverlap<T>::RMSOverlap(
    int initialWindowSize,
    int initialHopSize,
    contorchionist::core::util_windowing::Type initialWinType,
    float zeroPaddingFactor,
    contorchionist::core::util_windowing::Alignment initialWinAlign,
    NormalizationType initialNormType,
    float fixedNormMultiplier,
    int initialBlockSize,
    bool verbose)
    : windowSize_(1024),
      hopSize_(512),
      numOverlaps_(0),
      zeroPaddingSamples_(0),
      windowType_(initialWinType),
      windowAlignment_(initialWinAlign),
      normalizationType_(initialNormType),
      fixedNormalizationMultiplier_(fixedNormMultiplier),
      internalBlockSize_(initialBlockSize),
      circularBuffer_(windowSize_ + internalBlockSize_),
      window_(torch::empty({0})),
      layerRMSValues_(torch::empty({0})),
      sumBuffer_(torch::empty({0})),
      windowOverlapSum_(torch::empty({0})),
      normalizationBuffer_(torch::empty({0})),
      read_idx_(0),
      write_idx_(0),
      samples_in_buffer_(0),
      running_(true),
      synthesis_reference_point_(0),
      samplesSinceLastOutput_(0),
      layerCountForInitialNorm_(0),
      framesProducedInLastCall_(0),
      currentRMSValue_(T(0)),
      samplesUntilNextHop_(0),
      analysisBuffer_(torch::empty({0})),
      isInitialized_(false),
      verbose_(verbose)
{
    setBlockSize(initialBlockSize);
    setWindowSize(initialWindowSize);
    setHopSize(initialHopSize);
    setZeroPadding(zeroPaddingFactor, windowAlignment_);
    setNormalization(initialNormType, fixedNormMultiplier);

    isInitialized_ = true;
    RMS_POST("RMSOverlap: Initialized. Window: %d, Hop: %d, Overlaps: %d, Block: %d. isInitialized: %s", 
             windowSize_, hopSize_, numOverlaps_, internalBlockSize_, isInitialized_ ? "true" : "false");
}

// Destructor
template<typename T>
RMSOverlap<T>::~RMSOverlap() {
    running_ = false;
    data_available_cv_.notify_all();
    space_available_cv_.notify_all();
    RMS_POST("RMSOverlap: Destroyed. Signaled running=false and notified CVs.");
}

// Reinitialize method
template<typename T>
void RMSOverlap<T>::reinitialize() {
    RMS_POST("RMSOverlap::reinitialize: Starting. WinSize: %d, HopSize: %d, BlockSize: %d", windowSize_, hopSize_, internalBlockSize_);
    isInitialized_ = false;

    if (windowSize_ <= 0 || hopSize_ <= 0 || internalBlockSize_ <= 0) {
        RMS_POST("RMSOverlap::reinitialize: Invalid sizes, aborting reinitialization.");
        return;
    }

    numOverlaps_ = windowSize_ / hopSize_;
    if (numOverlaps_ < 1) numOverlaps_ = 1;
    RMS_POST("RMSOverlap::reinitialize: numOverlaps_ = %d", numOverlaps_);

    allocateBuffers();
    
    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    bool periodic_window = true;
    window_ = contorchionist::core::util_windowing::generate_torch_window_aligned(
        windowSize_,
        windowType_,
        periodic_window,
        zeroPaddingSamples_,
        windowAlignment_,
        tensor_options
    );
    
    if (window_.defined()) {
        RMS_POST("RMSOverlap::reinitialize: window_ generated. Size: %lld", window_.numel());
    } else {
        RMS_POST("RMSOverlap::reinitialize: window_ IS NOT DEFINED after generation!");
    }
    
    calculateNormalizationFactors();

    circularBuffer_.clear();
    synthesis_reference_point_ = 0;
    samplesSinceLastOutput_ = 0;
    layerCountForInitialNorm_ = 0;
    framesProducedInLastCall_ = 0;
    currentRMSValue_ = T(0);
    samplesUntilNextHop_ = 0;

    if (layerRMSValues_.defined()) {
        layerRMSValues_.zero_();
    }
    if (sumBuffer_.defined()) {
        sumBuffer_.zero_();
    }
    if (analysisBuffer_.defined()) {
        analysisBuffer_.zero_();
    }

    isInitialized_ = true;
    RMS_POST("RMSOverlap: Reinitialized. Window: %d, Hop: %d, Overlaps: %d. isInitialized: %s", 
             windowSize_, hopSize_, numOverlaps_, isInitialized_ ? "true" : "false");
}

// post_input_data method
template<typename T>
void RMSOverlap<T>::post_input_data(const T* data, int num_samples) {
    if (!running_.load() || num_samples == 0) {
        return;
    }

    bool written = circularBuffer_.write_overwrite(data, num_samples);

    if (written) {
        // RMS_POST("RMSOverlap::post_input_data: Successfully wrote %d samples. Buffer now has %zu samples.", num_samples, circularBuffer_.getSamplesAvailable());
    } else {
        RMS_POST("RMSOverlap::post_input_data: Failed to write %d samples (buffer full or other issue). Buffer has %zu samples.", num_samples, circularBuffer_.getSamplesAvailable());
    }
}

// allocateBuffers method
template<typename T>
void RMSOverlap<T>::allocateBuffers() {
    RMS_POST("RMSOverlap::allocateBuffers: Starting. WinSize: %d, BlockSize: %d, NumOverlaps: %d", windowSize_, internalBlockSize_, numOverlaps_);
    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    if (!analysisBuffer_.defined() || analysisBuffer_.numel() != windowSize_) {
        analysisBuffer_ = torch::zeros({windowSize_}, tensor_options);
        RMS_POST("RMSOverlap::allocateBuffers: analysisBuffer_ (re)allocated. Size: %d", windowSize_);
    }

    if (!layerRMSValues_.defined() || layerRMSValues_.numel() != numOverlaps_) {
        layerRMSValues_ = torch::zeros({numOverlaps_}, tensor_options);
        RMS_POST("RMSOverlap::allocateBuffers: layerRMSValues_ (re)allocated. Size: %d", numOverlaps_);
    }

    if (!windowOverlapSum_.defined() || windowOverlapSum_.numel() != windowSize_) {
        windowOverlapSum_ = torch::zeros({windowSize_}, tensor_options);
        RMS_POST("RMSOverlap::allocateBuffers: windowOverlapSum_ (re)allocated. Size: %d", windowSize_);
    }

    if (!normalizationBuffer_.defined() || normalizationBuffer_.numel() != windowSize_) {
        normalizationBuffer_ = torch::zeros({windowSize_}, tensor_options);
        RMS_POST("RMSOverlap::allocateBuffers: normalizationBuffer_ (re)allocated. Size: %d", windowSize_);
    }

    if (!sumBuffer_.defined() || sumBuffer_.numel() != internalBlockSize_) {
        sumBuffer_ = torch::zeros({internalBlockSize_}, tensor_options);
        RMS_POST("RMSOverlap::allocateBuffers: sumBuffer_ (re)allocated. Size: %d", internalBlockSize_);
    }
    
    RMS_POST("RMSOverlap: Buffers allocated/checked. Circ Capacity: %zu, Analysis: %lld, Win: %lld, LayerRMS: %lld, Sum: %lld, WinOverlapSum: %lld, NormBuf: %lld",
        circularBuffer_.getCapacity(),
        analysisBuffer_.defined() ? analysisBuffer_.numel() : 0,
        window_.defined() ? window_.numel() : 0,
        layerRMSValues_.defined() ? layerRMSValues_.numel() : 0,
        sumBuffer_.defined() ? sumBuffer_.numel() : 0,
        windowOverlapSum_.defined() ? windowOverlapSum_.numel() : 0,
        normalizationBuffer_.defined() ? normalizationBuffer_.numel() : 0
    );
}

// stringToNormalizationType method
template<typename T>
typename RMSOverlap<T>::NormalizationType RMSOverlap<T>::stringToNormalizationType(const std::string& strNorm) {
    if (strNorm == "win_rms" || strNorm == "0") return NormalizationType::WINDOW_OVERLAP_RMS;
    if (strNorm == "win_mean" || strNorm == "1") return NormalizationType::WINDOW_OVERLAP_MEAN;
    if (strNorm == "win_vals" || strNorm == "2") return NormalizationType::WINDOW_OVERLAP_VALS;
    if (strNorm == "overlap_inverse" || strNorm == "overlap" || strNorm == "3") return NormalizationType::OVERLAP_INVERSE;
    if (strNorm == "fixed" || strNorm == "4") return NormalizationType::FIXED_MULTIPLIER;
    if (strNorm == "none" || strNorm == "5") return NormalizationType::NONE;
    // Unknown normalization type, defaulting to WINDOW_OVERLAP_RMS
    return NormalizationType::WINDOW_OVERLAP_RMS;
}

// calculateNormalizationFactors method
template<typename T>
void RMSOverlap<T>::calculateNormalizationFactors() {
    if (windowSize_ == 0 || numOverlaps_ == 0) {
        RMS_POST("RMSOverlap::calculateNormalizationFactors: Invalid windowSize or numOverlaps, aborting.");
        return;
    }

    if (!normalizationBuffer_.defined() || !windowOverlapSum_.defined()) {
        RMS_POST("RMSOverlap::calculateNormalizationFactors: Buffers not defined. Aborting.");
        return;
    }

    normalizationBuffer_.fill_(1.0f);
    windowOverlapSum_.zero_();

    if (window_.defined() && window_.numel() > 0) {
        auto wos_acc = windowOverlapSum_.accessor<float,1>();
        auto win_acc = window_.accessor<float,1>();
        for (int o = 0; o < numOverlaps_; ++o) {
            int win_first_index_for_sum = (windowSize_ - (o * hopSize_) + windowSize_) % windowSize_;
            for (int i = 0; i < windowSize_; ++i) {
                int w_idx = (win_first_index_for_sum + i) % windowSize_;
                if (w_idx < window_.numel()) {
                     wos_acc[i] += win_acc[w_idx];
                }
            }
        }
    } else {
        RMS_POST("RMSOverlap::calculateNormalizationFactors: Window tensor is not defined or empty for sum calculation.");
    }

    float overallNormFactor = 1.0f;

    if (!window_.defined() || window_.numel() == 0) {
        RMS_POST("RMSOverlap::calculateNormalizationFactors: Window tensor is not defined or empty. Using default norm factors.");
        if (normalizationType_ == NormalizationType::FIXED_MULTIPLIER) {
            normalizationBuffer_.fill_(fixedNormalizationMultiplier_);
        } else {
            normalizationBuffer_.fill_(1.0f);
        }
        if (normalizationType_ == NormalizationType::OVERLAP_INVERSE && numOverlaps_ > 0) {
             normalizationBuffer_.fill_(1.0f / static_cast<float>(numOverlaps_));
        }
        return;
    }

    switch (normalizationType_) {
        case NormalizationType::NONE:
            break;
        case NormalizationType::FIXED_MULTIPLIER:
            normalizationBuffer_.fill_(fixedNormalizationMultiplier_);
            break;
        case NormalizationType::OVERLAP_INVERSE:
            if (numOverlaps_ > 0) {
                overallNormFactor = 1.0f / static_cast<float>(numOverlaps_);
            }
            normalizationBuffer_.fill_(overallNormFactor);
            break;
        case NormalizationType::WINDOW_OVERLAP_VALS:
            {
                auto nb_acc = normalizationBuffer_.accessor<float,1>();
                auto wos_acc_read = windowOverlapSum_.accessor<float,1>();
                for (int i = 0; i < windowSize_; ++i) {
                    if (i < wos_acc_read.size(0)) {
                        if (std::abs(wos_acc_read[i]) > 1e-9) {
                            nb_acc[i] = 1.0f / wos_acc_read[i];
                        } else {
                            nb_acc[i] = 1.0f;
                        }
                    } else {
                        nb_acc[i] = 1.0f;
                    }
                }
            }
            break;
        case NormalizationType::WINDOW_OVERLAP_MEAN:
            {
                float sumAbsWin = window_.abs().sum().template item<float>();
                if (windowSize_ > 0 && numOverlaps_ > 0 && sumAbsWin > 1e-9) {
                    overallNormFactor = 1.0f / ((sumAbsWin / windowSize_) * numOverlaps_);
                }
                normalizationBuffer_.fill_(overallNormFactor);
            }
            break;
        case NormalizationType::WINDOW_OVERLAP_RMS:
            {
                float sumSqWin = window_.pow(2).sum().template item<float>();
                if (windowSize_ > 0 && numOverlaps_ > 0 && sumSqWin > 1e-9) {
                    overallNormFactor = 1.0f / (std::sqrt(sumSqWin / windowSize_) * numOverlaps_);
                }
                normalizationBuffer_.fill_(overallNormFactor);
            }
            break;
    }
    RMS_POST("RMSOverlap: Normalization factors calculated. Type: %d", static_cast<int>(normalizationType_));
}

// process method
template<typename T>
bool RMSOverlap<T>::process(const T* input_buffer_param, T* output_buffer, int inputBlockSize) {
    std::lock_guard<std::mutex> lock(process_mutex_);

    RMS_POST("RMSOverlap::process: Entered. inputBlockSize: %d, internalBlockSize_: %d, samples_in_buffer_: %zu, hopSize_: %d",
        inputBlockSize, internalBlockSize_, circularBuffer_.getSamplesAvailable(), hopSize_);

    if (!output_buffer || inputBlockSize <= 0 || !isInitialized_ || internalBlockSize_ <= 0) {
        RMS_POST("RMSOverlap::process: Critical check failed: null output_buffer OR non-positive inputBlockSize OR not initialized OR internalBlockSize_ invalid.");
        if (input_buffer_param == nullptr) {
             RMS_POST("RMSOverlap::process: (FYI: input_buffer_param is nullptr, intending to use circular buffer.)");
        } else {
             RMS_POST("RMSOverlap::process: (FYI: input_buffer_param is NOT nullptr. This mode is not currently expected for rmsoverlap~.)");
        }
        if (output_buffer && inputBlockSize > 0) {
            std::fill_n(output_buffer, inputBlockSize, T(0));
            RMS_POST("RMSOverlap::process: Output buffer zeroed due to critical check failure.");
        }
        return false;
    }

    std::fill_n(output_buffer, inputBlockSize, T(0));
    bool new_frame_processed_this_call = false;

    for (int i = 0; i < inputBlockSize; ++i) {
        if (samplesUntilNextHop_ == 0) {
            RMS_POST("RMSOverlap::process: samplesUntilNextHop_ is 0. Checking for new frame. Available: %zu, WindowSize: %d",
                     circularBuffer_.getSamplesAvailable(), windowSize_);

            if (circularBuffer_.getSamplesAvailable() >= static_cast<size_t>(windowSize_)) {
                RMS_POST("RMSOverlap::process: Enough samples for a new frame. Peeking %d samples.", windowSize_);
                
                size_t samples_read = circularBuffer_.peek_with_delay_and_fill(analysisBuffer_.data_ptr<float>(), windowSize_, 0);
                if (samples_read < windowSize_) {
                    RMS_POST("RMSOverlap::process: ERROR - circularBuffer_.peek_with_delay_and_fill returned only %zu samples, expected %d!", samples_read, windowSize_);
                    output_buffer[i] = currentRMSValue_;
                    samplesUntilNextHop_ = 0;
                    continue;
                }

                RMS_POST("RMSOverlap::process: Frame peeked. Applying window.");
                torch::Tensor windowed_frame = analysisBuffer_ * window_;

                RMS_POST("RMSOverlap::process: Calculating RMS.");
                torch::Tensor squared_sum = torch::sum(windowed_frame * windowed_frame);
                currentRMSValue_ = static_cast<T>(torch::sqrt(squared_sum / static_cast<float>(windowSize_)).template item<float>());
                
                RMS_POST("RMSOverlap::process: New RMS: %f. Discarding %d samples. Buffer before discard: %zu", 
                         static_cast<float>(currentRMSValue_), hopSize_, circularBuffer_.getSamplesAvailable());
                size_t discarded = circularBuffer_.discard(hopSize_);
                
                if (discarded != hopSize_) {
                    RMS_POST("RMSOverlap::process: WARNING - Only discarded %zu samples, expected %d", discarded, hopSize_);
                }
                
                samplesUntilNextHop_ = hopSize_;
                new_frame_processed_this_call = true;
                RMS_POST("RMSOverlap::process: Frame processed. Next hop in %d samples. Buffer after discard: %zu", 
                         samplesUntilNextHop_, circularBuffer_.getSamplesAvailable());
            } else {
                RMS_POST("RMSOverlap::process: Not enough samples for a new frame. Available: %zu, Needed: %d. Outputting current RMS: %f",
                         circularBuffer_.getSamplesAvailable(), windowSize_, static_cast<float>(currentRMSValue_));
            }
        }

        output_buffer[i] = currentRMSValue_;

        if (samplesUntilNextHop_ > 0) {
            samplesUntilNextHop_--;
        }
    }

    if (new_frame_processed_this_call) {
        RMS_POST("RMSOverlap::process: Finished block. At least one NEW frame was processed. Last RMS output: %f", static_cast<float>(currentRMSValue_));
    } else {
        RMS_POST("RMSOverlap::process: Finished block. NO new frame processed this call. Samples in buffer: %zu. Outputting RMS: %f", 
                 circularBuffer_.getSamplesAvailable(), static_cast<float>(currentRMSValue_));
    }
    
    return true;
}

// Configuration methods
template<typename T>
void RMSOverlap<T>::setWindowSize(int newWindowSize) {
    if (newWindowSize <= 0) {
        RMS_POST("RMSOverlap: Invalid window size %d, not changed.", newWindowSize);
        return;
    }
    windowSize_ = std::max(1, newWindowSize);
    if (hopSize_ > 0) {
        numOverlaps_ = windowSize_ / hopSize_;
        if (numOverlaps_ < 1) numOverlaps_ = 1;
    }
    RMS_POST("RMSOverlap: Window size set to %d. NumOverlaps: %d", windowSize_, numOverlaps_);
    reinitialize();
}

template<typename T>
void RMSOverlap<T>::setHopSize(int newHopSize) {
    if (newHopSize <= 0) {
        RMS_POST("RMSOverlap: Invalid hop size %d, not changed.", newHopSize);
        return;
    }
    hopSize_ = std::max(1, newHopSize);
    if (windowSize_ > 0) {
        numOverlaps_ = windowSize_ / hopSize_;
        if (numOverlaps_ < 1) numOverlaps_ = 1;
    }
    if (hopSize_ > windowSize_) {
        RMS_POST("RMSOverlap: Warning - hop size %d > window size %d. Setting hop to window size.", hopSize_, windowSize_);
        hopSize_ = windowSize_;
        numOverlaps_ = 1;
    }
    RMS_POST("RMSOverlap: Hop size set to %d. NumOverlaps: %d", hopSize_, numOverlaps_);
    reinitialize();
}

template<typename T>
void RMSOverlap<T>::setWindowType(contorchionist::core::util_windowing::Type newType) {
    windowType_ = newType;
    RMS_POST("RMSOverlap: Window type set to %s", contorchionist::core::util_windowing::torch_window_type_to_string(newType).c_str());
    reinitialize();
}

template<typename T>
void RMSOverlap<T>::setZeroPadding(float factor, contorchionist::core::util_windowing::Alignment alignment) {
    if (factor < 0.0f) factor = 0.0f;
    if (factor >= 1.0f) factor = 0.99999f;
    zeroPaddingSamples_ = static_cast<int>(factor * windowSize_);
    windowAlignment_ = alignment;
    RMS_POST("RMSOverlap: Zero padding factor %.2f (%d samples), alignment %s", factor, zeroPaddingSamples_, contorchionist::core::util_windowing::torch_window_alignment_to_string(alignment).c_str());
    reinitialize();
}

template<typename T>
void RMSOverlap<T>::setZeroPaddingSamples(int samples, contorchionist::core::util_windowing::Alignment alignment) {
    if (samples < 0) samples = 0;
    zeroPaddingSamples_ = std::min(samples, windowSize_ > 0 ? windowSize_ -1 : 0);
    windowAlignment_ = alignment;
    RMS_POST("RMSOverlap: Zero padding samples %d, alignment %s", zeroPaddingSamples_, contorchionist::core::util_windowing::torch_window_alignment_to_string(alignment).c_str());
    reinitialize();
}

template<typename T>
void RMSOverlap<T>::setNormalization(NormalizationType newType, float fixedMultiplier) {
    normalizationType_ = newType;
    if (newType == NormalizationType::FIXED_MULTIPLIER) {
        fixedNormalizationMultiplier_ = fixedMultiplier;
    }
    RMS_POST("RMSOverlap: Normalization type set to %d, fixed_mult: %.2f", static_cast<int>(newType), fixedNormalizationMultiplier_);
    reinitialize();
}

template<typename T>
void RMSOverlap<T>::setBlockSize(int newBlockSize) {
    if (newBlockSize <= 0) {
        RMS_POST("RMSOverlap: Invalid block size %d. Setting to 64.", newBlockSize);
        internalBlockSize_ = 64;
    } else {
        internalBlockSize_ = newBlockSize;
    }
    RMS_POST("RMSOverlap: Block size set to %d", internalBlockSize_);
    reinitialize();
}

// Static helper methods
template<typename T>
std::string RMSOverlap<T>::toString(NormalizationType type) {
    switch (type) {
        case NormalizationType::WINDOW_OVERLAP_RMS:  return "win_rms";
        case NormalizationType::WINDOW_OVERLAP_MEAN: return "win_mean";
        case NormalizationType::WINDOW_OVERLAP_VALS: return "win_vals";
        case NormalizationType::OVERLAP_INVERSE:     return "overlap_inverse";
        case NormalizationType::FIXED_MULTIPLIER:    return "fixed";
        case NormalizationType::NONE:                return "none";
        default:                                     return "unknown_norm_type";
    }
}

template<typename T>
typename RMSOverlap<T>::NormalizationType RMSOverlap<T>::parseNormalizationType(const std::string& normStr) {
    return RMSOverlap<T>::stringToNormalizationType(normStr);
}

// Type alias for common usage
using RMSOverlapFloat = RMSOverlap<float>;
using RMSOverlapDouble = RMSOverlap<double>;

} // namespace ap_rmsoverlap
} // namespace core
} // namespace contorchionist

#endif // CORE_AP_RMSOVERLAP_H

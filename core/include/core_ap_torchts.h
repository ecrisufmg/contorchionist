#ifndef CORE_AP_TORCHTS_H
#define CORE_AP_TORCHTS_H


#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <string>
#include <mutex>    
#include <atomic>   
#include <thread>   
#include <memory>  
#include "contorchionist_core/contorchionist_core_export.h" // Include the generated export header
#include "core_util_circbuffer.h"

namespace contorchionist {
    namespace core {
        namespace ap_torchts {

/**
 * @brief deal with output tensor from the model and write to the appropriate output circular buffer
 * @param tensor The output tensor from the model
 * @param channel_index The index of the output channel
 * @param buffer_size The size of the buffer
 * @param model_buffer_size The size of the model buffer
 * @param num_batches The number of batches processed
 * @param is_batched_processing Whether the processing is batched
 * @param out_buffers The vector of output circular buffers
 */
CONTORCHIONIST_CORE_EXPORT
bool process_output_tensor(
    torch::Tensor tensor,
    int channel_index,
    int buffer_size,
    int model_buffer_size,
    int num_batches,
    bool is_batched_processing,
    std::vector<std::unique_ptr<contorchionist::core::util_circbuffer::CircularBuffer<float>>>& out_buffers);


/**
 * @brief Process audio signal through a TorchScript model
 * @param model loaded TorchScript model
 * @param method method name to call on the model (e.g., "forward")
 * @param in_buffers Vector of pointers to input circular buffers
 * @param in_ch Number of input channels
 * @param out_ch Number of output channels
 * @param buffer_size buffer size in samples (circular buffers)
 * @param model_buffer_size model's buffer size
 * @param device device (CPU/CUDA/MPS)
 * @param model_mutex Mutex to protect concurrent model access
 * @param out_buffers Vector of pointers to output circular buffers
 */
CONTORCHIONIST_CORE_EXPORT
bool TorchTSProcessorSignal(
    torch::jit::script::Module& model,
    const std::string& method,
    const std::vector<const float*>& in_buffers,
    int in_ch,
    int out_ch,
    int buffer_size,
    int model_buffer_size,
    torch::Device device,
    std::mutex& model_mutex,
    std::vector<std::unique_ptr<contorchionist::core::util_circbuffer::CircularBuffer<float>>>& out_buffers);


// ----- buffer size validation utils -----
/**
 * @brief Checks if a number is a power of 2
 * @param n Number to check
 * @return true if n is power of 2, false otherwise
 */
CONTORCHIONIST_CORE_EXPORT
bool is_power_of_2(int n);

/**
 * @brief Suggests optimal buffer size based on constraints
 * @param requested_size Original requested size
 * @param block_size Current block size (minimum)
 * @param model_buffer_size Model's maximum buffer size (0 = no limit)
 * @return Optimal buffer size (power of 2, multiple of block_size)
 */
CONTORCHIONIST_CORE_EXPORT
int suggest_optimal_buffer_size(int requested_size, int block_size, int model_buffer_size);

/**
 * @brief Validates and optionally corrects buffer size
 * @param requested_size The requested buffer size
 * @param block_size Current block size (minimum allowed)
 * @param model_buffer_size Model's maximum buffer size (0 = no limit)
 * @param require_power_of_2 Whether to enforce power of 2 requirement
 * @param auto_correct Whether to auto-correct invalid sizes
 * @param corrected_size [OUT] The corrected size (if auto_correct = true)
 * @param error_message [OUT] Detailed error message if validation fails
 * @param warning_message [OUT] Warning message for suboptimal but valid configs
 * @return true if size is valid or was successfully corrected
 */
CONTORCHIONIST_CORE_EXPORT
bool validate_buffer_size(
    int requested_size,
    int block_size,
    int model_buffer_size,
    int max_buffer_size,
    bool require_power_of_2,
    bool auto_correct,
    int* corrected_size,
    std::string* error_message,
    std::string* warning_message = nullptr
);

 /**
 * @brief Creates circular buffers for input and output channels
 * @param in_buffers Vector to store input circular buffers
 * @param out_buffers Vector to store output circular buffers  
 * @param in_channel Number of input channels
 * @param out_channel Number of output channels
 * @param buffer_size Size of each buffer in samples
 */
CONTORCHIONIST_CORE_EXPORT
void create_circular_buffer(
    std::vector<std::unique_ptr<contorchionist::core::util_circbuffer::CircularBuffer<float>>>& in_buffers, // vector of input circular buffers
    std::vector<std::unique_ptr<contorchionist::core::util_circbuffer::CircularBuffer<float>>>& out_buffers, // vector of output circular buffers
    int in_channel, // input channels
    int out_channel, // output channels
    int buffer_size // buffer size
);

// enumerations structure for processing modes
enum ProcessingMode {
    SIMPLE, // = 0
    SLIDING_WINDOW, // = 1
    RECURRENT // = 2
};

    
        } // namespace ap_torchts
    } // namespace core
} // namespace contorchionist

#endif // A_MODEL_PROCESS_H  
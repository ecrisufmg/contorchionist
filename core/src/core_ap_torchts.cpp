#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <mutex>    
#include <atomic>   
#include <thread>   
#include <memory>  
#include "core_ap_torchts.h"
#include "core_util_circbuffer.h"


namespace contorchionist {
    namespace core {
        namespace ap_torchts {

// prepare the output tensor for signal output
bool process_output_tensor(
    torch::Tensor tensor,
    int channel_index,
    int buffer_size,
    int model_buffer_size,
    int num_batches,
    bool is_batched_processing,
    std::vector<std::unique_ptr<contorchionist::core::util_circbuffer::CircularBuffer<float>>>& out_buffers) {
    
    // Move tensor to CPU
    tensor = tensor.to(torch::kCPU);

    // case 1: batched tensors
    if (is_batched_processing) {
        // check the output tensors shape: [num_batches, 1, model_buffer_size] or similar
        if (tensor.dim() == 3 &&  // has 3 dimensions
            tensor.size(0) == num_batches && // first dimension size = num_batches
            tensor.size(1) == 1 && // second dimension size = 1 (channel)
            tensor.size(2) == model_buffer_size) { // third dimension size = model_buffer_size
            // reshape to [buffer_size]
            tensor = tensor.view({(int64_t)buffer_size}); 
        } else {
            // fallback for unexpected batch shapes, fill with zeros
            std::vector<float> zeros(buffer_size, 0.0f);
            out_buffers[channel_index]->write_overwrite(zeros.data(), buffer_size);
            return false; // Indicate failure/fallback
        }
    // case 2: non-batched tensors
    } else {
        // original logic for non-batched output
        if (tensor.dim() == 2 && tensor.size(0) == 1) {
            tensor = tensor.squeeze(0); // convert to 1D tensor [buffer_size]
        }
    }
    // check if tensor shape = [1, buffer_size] and write it to buffer
    if (tensor.dim() == 1 && tensor.size(0) == buffer_size) {
        out_buffers[channel_index]->write_overwrite(tensor.data_ptr<float>(), buffer_size);
        return true;
    } else {
        // if the shape doesn't match, fill with zeros
        std::vector<float> zeros(buffer_size, 0.0f);
        out_buffers[channel_index]->write_overwrite(zeros.data(), buffer_size);
        return false;
    }
}


// torch.ts~ signal processor
bool TorchTSProcessorSignal(
    torch::jit::script::Module& model, //pointer to the loaded model
    const std::string& method, // selected method to be used on the model
    const std::vector<const float*>& in_buffers, // vector of pointers to the input buffers
    int in_ch, // number of input channels
    int out_ch, // number of output channels
    int buffer_size, // size of the input and output circularbuffers
    int model_buffer_size, // model's buffer size
    torch::Device device, // device to run the model on (CPU or GPU)
    std::mutex& model_mutex, // mutex to protect the model from concurrent access
    std::vector<std::unique_ptr<contorchionist::core::util_circbuffer::CircularBuffer<float>>>& out_buffers) { //circular output buffers

    // disable the gradient computation for inference
    c10::InferenceMode guard;

    // checking if batching is possible
    bool can_batch = model_buffer_size > 0 && buffer_size > model_buffer_size && buffer_size % model_buffer_size == 0;
    int num_batches = can_batch ? buffer_size / model_buffer_size : 1;
    

    // prepare the input tensors
    std::vector<torch::jit::IValue> inputs; // vector to hold the input tensors

    // caso 1: batching is possible - prepare batched input tensors
    if (can_batch) {
        // create a batched input tensor for each input channel
        for (int ch = 0; ch < in_ch; ++ch) {
            // create a single tensor from the entire buffer making it contiguous
            auto t = torch::from_blob(const_cast<float*>(in_buffers[ch]), {(int64_t)buffer_size}, torch::kFloat32).clone();
            // reshape the tensor from [buffer_size] into a batched tensor [num_batches, model_buffer_size]
            t = t.view({num_batches, (int64_t)model_buffer_size});
            // add the channel dimension for the model [num_batches, 1, model_buffer_size]
            t = t.unsqueeze(1);
            // move to the specified device.
            t = t.to(device);
            inputs.push_back(t);
        }
    // caso 2: batching is not possible - prepare individual input tensors
    } else {
        for (int ch = 0; ch < in_ch; ++ch) {
            // create a tensor for each input buffer with shape [1, buffer_size]
            auto t = torch::from_blob(const_cast<float*>(in_buffers[ch]), {1, (int64_t)buffer_size}, torch::kFloat32).clone(); 
            t = t.to(device);
            inputs.push_back(t);
        }
    }
    // create a vector to hold the output tensors
    std::vector<torch::Tensor> out_tensors;
    torch::IValue output;

    // process the model
    try {
        // Use mutex to protect model access
        std::lock_guard<std::mutex> lock(model_mutex);
        output = model.get_method(method)(inputs);
    //catch any error during the model execution
    } catch (const c10::Error &e) {
        std::cerr << e.what() << '\n';
        // fill output tensors with zeros in case of error
        for (int ch = 0; ch < out_ch; ++ch) {
            out_tensors.push_back(torch::zeros({(int64_t)buffer_size}, torch::kFloat32));
            std::vector<float> zeros(buffer_size, 0.0f);
            out_buffers[ch]->write_overwrite(zeros.data(), buffer_size);
        }
        return false;
    }

    // prepare the output tensor for signal output
    if (output.isTuple() || output.isList()) {
        auto outs = output.isTuple() ? output.toTuple()->elements() : output.toListRef();
        for (int ch = 0; ch < out_ch; ++ch) {
            if (ch < outs.size() && outs[ch].isTensor()) {
                bool success = process_output_tensor(
                    outs[ch].toTensor(),
                    ch,
                    buffer_size,
                    model_buffer_size,
                    num_batches,
                    can_batch,
                    out_buffers
                );
                if (!success) {
                    std::cout << "TorchTS Warning: Output tensor for channel " << ch 
                                << " had unexpected shape. Filled with zeros." << std::endl;
                }
            } else {
                std::vector<float> zeros(buffer_size, 0.0f);
                out_buffers[ch]->write_overwrite(zeros.data(), buffer_size);
            }
        }
    // if the output is a single tensor
    } else if (output.isTensor()) {
        bool success = process_output_tensor(
            output.toTensor(),
            0,
            buffer_size,
            model_buffer_size,
            num_batches,
            can_batch,
            out_buffers
        );
        // log if fallback was used
        if (!success && can_batch) {
            std::cerr << "TorchTS Warning: Fallback to zeros for single tensor output" << std::endl;
        }
        // fill extra output channels with zeros
        for (int ch = 1; ch < out_ch; ++ch) {
            std::vector<float> zeros(buffer_size, 0.0f);
            out_buffers[ch]->write_overwrite(zeros.data(), buffer_size);
        }
    }
    return can_batch;
} 


// ----------- buffer size validation utils ----- //

// check if a number is a power of 2
bool is_power_of_2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Suggest an optimal buffer size based on various constraints
int suggest_optimal_buffer_size(int requested_size, int block_size, int model_buffer_size) {
    // Start with minimum valid size
    int min_size = std::max(requested_size, block_size);
    
    // Make it multiple of block_size
    int multiple_size = ((min_size / block_size) + (min_size % block_size ? 1 : 0)) * block_size;
    
    // Find next power of 2 that's >= multiple_size and multiple of block_size
    int power_of_2 = 1;
    while (power_of_2 < multiple_size) {
        power_of_2 <<= 1;
    }
    // Ensure power of 2 is multiple of block_size
    while (power_of_2 % block_size != 0) {
        power_of_2 <<= 1;
    }
    // Check model limit
    if (model_buffer_size > 0 && power_of_2 > model_buffer_size) {
        // Fallback to largest valid multiple
        power_of_2 = (model_buffer_size / block_size) * block_size;
        if (power_of_2 < block_size) {
            power_of_2 = block_size;
        }
    }
    return power_of_2;
}

// Validate the buffer size based on various constraints
bool validate_buffer_size(
    int requested_size,
    int block_size,
    int model_buffer_size,
    int max_buffer_size,
    bool require_power_of_2,
    bool auto_correct,
    int* corrected_size,
    std::string* error_message,
    std::string* warning_message) {
    // Clear outputs
    if (corrected_size){
        *corrected_size = requested_size;
    }
    if (error_message){
        error_message->clear();
    }
    if (warning_message) {
        warning_message->clear();
    }

    bool has_errors = false;
    bool has_warnings = false;
    
    // validation 1: Positive integer
    if (requested_size < 1) {
        if (error_message) {
            *error_message = "Buffer size must be a positive integer (got " + std::to_string(requested_size) + ")";
        }
        if (auto_correct && corrected_size) {
            *corrected_size = block_size;
        }
        has_errors = true;
    }
    
    // validation 2: Minimum size (model_buffer_size)
    else if (requested_size < model_buffer_size) {
        if (error_message) {
            *error_message = "Buffer size (" + std::to_string(requested_size) + 
                           ") must be >= model buffer size (" + std::to_string(model_buffer_size) + ")";
        }
        if (auto_correct && corrected_size) {
            *corrected_size = model_buffer_size;
        }
        has_errors = true;
    }

    // validation 3: Multiple of block_size
    else if (requested_size % block_size != 0) {
        if (error_message) {
            *error_message = "Buffer size (" + std::to_string(requested_size) + 
                           ") must be multiple of block size (" + std::to_string(block_size) + ")";
        }
        if (auto_correct && corrected_size) {
            // Round up to next multiple
            *corrected_size = ((requested_size / block_size) + 1) * block_size;
        }
        has_errors = true;
    }

    // validation 4: Maximum size (model limit)
    else if (max_buffer_size > 0 && requested_size > max_buffer_size) {
        if (error_message) {
            *error_message = "Buffer size (" + std::to_string(requested_size) +
                           ") cannot exceed model's max_buffer_size (" + std::to_string(max_buffer_size) + ")";
        }
        if (auto_correct && corrected_size) {
            *corrected_size = max_buffer_size;
        }
        has_errors = true;
    }
   
    // validation 5: Power of 2 (warning or error based on require_power_of_2)
    else if (!is_power_of_2(requested_size)) {
        std::string power_of_2_msg = "Buffer size (" + std::to_string(requested_size) + 
                                   ") is not a power of 2. Performance may be suboptimal.";
        
        if (require_power_of_2) {
            // Treat as error
            if (error_message) {
                *error_message = power_of_2_msg + " Required: power of 2 (64, 128, 256, 512, 1024...)";
            }
            if (auto_correct && corrected_size) {
                *corrected_size = suggest_optimal_buffer_size(requested_size, block_size, model_buffer_size);
            }
            has_errors = true;
        } else {
            // Treat as warning
            if (warning_message) {
                *warning_message = power_of_2_msg + " Consider using " + 
                                 std::to_string(suggest_optimal_buffer_size(requested_size, block_size, model_buffer_size)) + 
                                 " for optimal performance.";
            }
            has_warnings = true;
        }
    }
    
    return !has_errors; // Valid if no errors (warnings are ok)
}

void create_circular_buffer(
    std::vector<std::unique_ptr<contorchionist::core::util_circbuffer::CircularBuffer<float>>>& in_buffers, 
    std::vector<std::unique_ptr<contorchionist::core::util_circbuffer::CircularBuffer<float>>>& out_buffers, 
    int in_channel, 
    int out_channel, 
    int buffer_size){
    // resize the buffers
    in_buffers.clear();
    out_buffers.clear();

    // Reserve capacity for efficiency
    in_buffers.reserve(in_channel);
    out_buffers.reserve(out_channel);

    // initialize the input circular buffers (one for each input channel)
    for (int i = 0; i < in_channel; ++i){
        in_buffers.emplace_back(std::make_unique<contorchionist::core::util_circbuffer::CircularBuffer<float>>(buffer_size));
    }

    // initialize the output circular buffers (one for each output channel)
    for (int i = 0; i < out_channel; ++i){
        out_buffers.emplace_back(std::make_unique<contorchionist::core::util_circbuffer::CircularBuffer<float>>(buffer_size));
    }
}


    
        } // namespace ap_torchts
    } // namespace core
} // namespace contorchionist
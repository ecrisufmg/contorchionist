#pragma once
#include <torch/torch.h>
#include <vector>
#include <numeric>
#include <string>

namespace contorchionist {
    namespace core {
        namespace dp_ls2tensor {

// Configuration structure for formatting
struct FormatConfig {
    std::vector<float> data_buffer;
    std::vector<int64_t> tensor_shape;
    std::vector<int64_t> sample_shape;
    bool has_target = false;
    int64_t target_size = 0;
    torch::Device device = torch::kCPU;
};

// Result structure for formatting
struct FormatResult {
    bool success = false;
    std::string error_message;
    torch::Tensor dataset_tensor;
    torch::Tensor target_tensor;
    
    // Statistics
    int64_t num_samples = 0;
    int64_t sample_data_size = 0;
    int64_t target_size = 0;
};

// Result structure for appending to buffer
struct BufferAppendResult {
    bool success = false;
    std::string error_message;
    bool buffer_full = false;
    size_t elements_added = 0;
    size_t current_position = 0;
    size_t total_capacity = 0;
};

// shuffle result structure
struct ShuffleResult {
    bool success = false;
    std::string error_message;
    torch::Tensor shuffled_indices;
    int64_t num_samples = 0;
};


// Format the tensor with an optional target
inline FormatResult Ls2TensorProcessor(const FormatConfig& config) {
    FormatResult result;
    
    // validations
    if (config.data_buffer.empty()) {
        result.error_message = "Data buffer is empty, cannot create tensor";
        return result;
    }
    
    if (config.tensor_shape.empty()) {
        result.error_message = "Tensor shape is empty";
        return result;
    }
    
    if (config.sample_shape.empty()) {
        result.error_message = "Sample shape is empty";
        return result;
    }

   
    std::vector<float> dataset_flat;
    std::vector<float> target_flat;
    std::vector<int64_t> dataset_shape = config.tensor_shape;
    std::vector<int64_t> target_shape;

    if (config.has_target) {
        if (config.target_size <= 0) {
            result.error_message = "Target flag is true but target size is not set";
            return result;
        }

        int64_t sample_total_size = std::accumulate(
            config.sample_shape.begin(), 
            config.sample_shape.end(), 
            (int64_t)1, 
            std::multiplies<int64_t>()
        );
        
        int64_t sample_data_size = sample_total_size - config.target_size;
        if (sample_data_size <= 0) {
            result.error_message = "Target size (" + std::to_string(config.target_size) + 
                                  ") is >= sample size (" + std::to_string(sample_total_size) + ")";
            return result;
        }

        int64_t num_samples = config.tensor_shape[0];
        result.num_samples = num_samples;
        result.sample_data_size = sample_data_size;
        result.target_size = config.target_size;
    
        if (config.sample_shape.size() == 1) {
            // Caso 1D: [num_samples, features] → [num_samples, features-target_size]
            dataset_shape.back() = sample_data_size;
        } else {
            // Caso multi-D: Flatten as últimas dimensões após remover target
            dataset_shape = {num_samples, sample_data_size};
        }
        target_shape = {num_samples, config.target_size};

        for (int64_t i = 0; i < num_samples; ++i) {
            auto row_start = config.data_buffer.begin() + i * sample_total_size;
            
            // Input: primeiros sample_data_size elementos
            dataset_flat.insert(dataset_flat.end(), 
                               row_start, 
                               row_start + sample_data_size);
            
            // Target: últimos target_size elementos  
            target_flat.insert(target_flat.end(), 
                              row_start + sample_data_size, 
                              row_start + sample_total_size);
        }

    } else {
        dataset_flat = config.data_buffer;
        result.num_samples = config.tensor_shape[0];
        result.sample_data_size = std::accumulate(
            config.sample_shape.begin(), 
            config.sample_shape.end(), 
            (int64_t)1, 
            std::multiplies<int64_t>()
        );
        result.target_size = 0;
    }

    // create tensors
    try {
        result.dataset_tensor = torch::from_blob(
            dataset_flat.data(),
            torch::IntArrayRef(dataset_shape),
            torch::TensorOptions().dtype(torch::kFloat)
        ).clone().to(config.device);

        if (config.has_target && !target_flat.empty()) {
            result.target_tensor = torch::from_blob(
                target_flat.data(),
                torch::IntArrayRef(target_shape),
                torch::TensorOptions().dtype(torch::kFloat)
            ).clone().to(config.device);
        }

        result.success = true;
        
    } catch (const c10::Error& e) {
        result.error_message = "Failed to create tensors: " + std::string(e.what());
        return result;
    }

    return result;
}

//calculate the dataste shape
inline std::vector<int64_t> calculate_dataset_shape(
    const std::vector<int64_t>& tensor_shape,
    const std::vector<int64_t>& sample_shape,
    int64_t target_size) {
    
    std::vector<int64_t> dataset_shape = tensor_shape;
    int64_t num_samples = tensor_shape[0];
    
    int64_t sample_total_size = std::accumulate(
        sample_shape.begin(), 
        sample_shape.end(), 
        (int64_t)1, 
        std::multiplies<int64_t>()
    );
    
    int64_t sample_data_size = sample_total_size - target_size;
    
    if (sample_shape.size() == 1) {
        dataset_shape.back() = sample_data_size;
    } else {
        dataset_shape = {num_samples, sample_data_size};
    }
    
    return dataset_shape;
}

//calculate the target shape
inline std::vector<int64_t> calculate_target_shape(int64_t num_samples, int64_t target_size) {
    return {num_samples, target_size};
}

// convert shape to string
inline std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) result += ", ";
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
}


// Append new data to the buffer
inline BufferAppendResult append_to_buffer(
    std::vector<float>& data_buffer,
    size_t& write_position,
    const std::vector<int64_t>& tensor_shape,
    const std::vector<float>& new_data) {
    
    BufferAppendResult result;
    
    // validations
    if (tensor_shape.empty()) {
        result.error_message = "Shape not set. Use shape message first";
        return result;
    }
    
    if (new_data.empty()) {
        result.error_message = "No data provided";
        return result;
    }

    // compute total capacity
    size_t total_size = std::accumulate(
        tensor_shape.begin(), 
        tensor_shape.end(), 
        (size_t)1, 
        std::multiplies<size_t>()
    );
    
    result.total_capacity = total_size;
    result.current_position = write_position;
    
    // check capacity
    if (write_position + new_data.size() > total_size) {
        result.error_message = "Input data exceeds buffer capacity. Capacity: " + 
                              std::to_string(total_size) + 
                              ", Current: " + std::to_string(write_position) + 
                              ", Trying to add: " + std::to_string(new_data.size());
        return result;
    }
    
    // resize buffer if necessary
    if (data_buffer.size() != total_size) {
        data_buffer.resize(total_size, 0.0f);
    }
    
    // append data
    for (size_t i = 0; i < new_data.size(); ++i) {
        data_buffer[write_position + i] = new_data[i];
    }
    
    write_position += new_data.size();
    result.elements_added = new_data.size();
    result.current_position = write_position;
    result.buffer_full = (write_position == total_size);
    result.success = true;
    
    return result;
}


// shuffle tensors (dataset and target)
inline ShuffleResult shuffle_tensors(
    torch::Tensor& dataset_tensor,
    torch::Tensor& target_tensor, 
    const torch::Device& device,
    bool has_target = false) {
    
    ShuffleResult result;
    
    // check dataset tensor
    if (!dataset_tensor.defined() || dataset_tensor.numel() == 0) {
        result.error_message = "Dataset tensor is empty or not defined";
        return result;
    }
    //check data set dimensions
    if (dataset_tensor.dim() < 1) {
        result.error_message = "Dataset tensor must have at least 1 dimension";
        return result;
    }
    
    int64_t num_samples = dataset_tensor.size(0);
    result.num_samples = num_samples;
    
    if (num_samples <= 1) {
        // shuffle not necessary
        result.success = true;
        result.error_message = "Skipping shuffle - only " + std::to_string(num_samples) + " sample(s)";
        return result;
    }
    
    // if has target
    if (has_target) {
        if (!target_tensor.defined()) {
            result.error_message = "Target flag is true but target tensor is not defined";
            return result;
        }
        
        if (target_tensor.size(0) != num_samples) {
            result.error_message = "Sample count mismatch. Dataset: " + 
                                  std::to_string(num_samples) + 
                                  ", Target: " + std::to_string(target_tensor.size(0));
            return result;
        }
    }
    
    try {
        // generate unique permutation
        auto options = torch::TensorOptions().device(device).dtype(torch::kLong);
        auto shuffled_indices = torch::randperm(num_samples, options);
        result.shuffled_indices = shuffled_indices;

        // apply shuffle to dataset
        dataset_tensor = dataset_tensor.index_select(0, shuffled_indices);
        
        // apply shuffle to target
        if (has_target && target_tensor.defined()) {
            target_tensor = target_tensor.index_select(0, shuffled_indices);
        }
        
        result.success = true;
        
    } catch (const c10::Error& e) {
        result.error_message = "LibTorch error during shuffle: " + std::string(e.what());
        return result;
    } catch (const std::exception& e) {
        result.error_message = "Standard error during shuffle: " + std::string(e.what());
        return result;
    }
    
    return result;
}

// helper function to shuffle a single tensor
inline ShuffleResult shuffle_single_tensor(
    torch::Tensor& tensor,
    const torch::Device& device) {
    
    torch::Tensor dummy_target;
    return shuffle_tensors(tensor, dummy_target, device, false);
}




        } // namespace dp_ls2tensor
    } // namespace core
} // namespace contorchionist
#include "core_util_tensors.h"
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace contorchionist {
    namespace core {
        namespace util_tensors {

torch::Tensor normalize(
    const torch::Tensor& tensor,
    float min_val,
    float max_val
) {
    auto tensor_min = torch::min(tensor);
    auto tensor_max = torch::max(tensor);
    
    // Avoid division by zero
    auto range = tensor_max - tensor_min;
    if (range.item<float>() < 1e-7f) {
        return torch::full_like(tensor, (min_val + max_val) / 2.0f);
    }
    
    // Normalize to [0, 1] then scale to [min_val, max_val]
    auto normalized = (tensor - tensor_min) / range;
    return normalized * (max_val - min_val) + min_val;
}

torch::Tensor clamp(
    const torch::Tensor& tensor,
    float min_val,
    float max_val
) {
    return torch::clamp(tensor, min_val, max_val);
}

torch::Tensor toDevice(
    const torch::Tensor& tensor,
    const torch::Device& device
) {
    if (tensor.device() == device) {
        return tensor;
    }
    return tensor.to(device);
}

bool validateShape(
    const torch::Tensor& tensor,
    const std::vector<int64_t>& expected_shape,
    const std::string& tensor_name
) {
    auto actual_shape = tensor.sizes();
    
    if (actual_shape.size() != expected_shape.size()) {
        return false;
    }
    
    for (size_t i = 0; i < expected_shape.size(); ++i) {
        if (expected_shape[i] >= 0 && actual_shape[i] != expected_shape[i]) {
            return false;
        }
    }
    
    return true;
}

std::string shapeToString(const torch::Tensor& tensor) {
    auto sizes = tensor.sizes();
    return shapeToString(std::vector<int64_t>(sizes.begin(), sizes.end()));
}

std::string shapeToString(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i < shape.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

int64_t calculateNumElements(const std::vector<int64_t>& shape) {
    int64_t count = 1;
    for (int64_t dim : shape) {
        count *= dim;
    }
    return count;
}

bool isBroadcastCompatible(
    const std::vector<int64_t>& shape1,
    const std::vector<int64_t>& shape2
) {
    int64_t i = shape1.size() - 1;
    int64_t j = shape2.size() - 1;
    
    while (i >= 0 && j >= 0) {
        if (shape1[i] != shape2[j] && shape1[i] != 1 && shape2[j] != 1) {
            return false;
        }
        i--;
        j--;
    }
    
    return true;
}

std::vector<int64_t> computeBroadcastShape(
    const std::vector<int64_t>& shape1,
    const std::vector<int64_t>& shape2
) {
    if (!isBroadcastCompatible(shape1, shape2)) {
        return {};
    }
    
    size_t max_ndim = std::max(shape1.size(), shape2.size());
    std::vector<int64_t> result(max_ndim);
    
    for (size_t k = 0; k < max_ndim; ++k) {
        int64_t i = static_cast<int64_t>(shape1.size()) - 1 - k;
        int64_t j = static_cast<int64_t>(shape2.size()) - 1 - k;
        int64_t dim_idx = max_ndim - 1 - k;
        
        int64_t dim1 = (i >= 0) ? shape1[i] : 1;
        int64_t dim2 = (j >= 0) ? shape2[j] : 1;
        
        result[dim_idx] = std::max(dim1, dim2);
    }
    
    return result;
}

torch::Tensor createFilledTensor(
    const std::vector<int64_t>& shape,
    float value,
    torch::ScalarType dtype,
    const torch::Device& device
) {
    auto options = torch::TensorOptions().dtype(dtype).device(device);
    return torch::full(shape, value, options);
}

std::pair<torch::Tensor, bool> safeReshape(
    const torch::Tensor& tensor,
    const std::vector<int64_t>& new_shape,
    bool allow_copy
) {
    try {
        // First try view (no copy)
        auto reshaped = tensor.view(new_shape);
        return {reshaped, true};
    } catch (const std::exception&) {
        if (allow_copy) {
            try {
                // Fallback to reshape (may copy)
                auto reshaped = tensor.reshape(new_shape);
                return {reshaped, true};
            } catch (const std::exception&) {
                return {torch::Tensor(), false};
            }
        } else {
            return {torch::Tensor(), false};
        }
    }
}

std::vector<float> tensorToVector(const torch::Tensor& tensor) {
    if (tensor.dim() != 1) {
        throw std::invalid_argument("tensorToVector requires 1D tensor");
    }
    
    auto cpu_tensor = tensor.to(torch::kCPU).to(torch::kFloat32);
    auto accessor = cpu_tensor.accessor<float, 1>();
    
    std::vector<float> result;
    result.reserve(cpu_tensor.size(0));
    
    for (int64_t i = 0; i < cpu_tensor.size(0); ++i) {
        result.push_back(accessor[i]);
    }
    
    return result;
}

torch::Tensor vectorToTensor(
    const std::vector<float>& data,
    const torch::Device& device
) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto tensor = torch::from_blob(
        const_cast<float*>(data.data()),
        {static_cast<int64_t>(data.size())},
        options
    ).clone(); // Clone to own the memory
    
    return tensor.to(device);
}

void printTensorInfo(
    const torch::Tensor& tensor,
    const std::string& name,
    int max_elements
) {
    std::cout << "=== Tensor Info: " << name << " ===" << std::endl;
    std::cout << "  Shape: " << shapeToString(tensor) << std::endl;
    std::cout << "  Device: " << tensor.device() << std::endl;
    std::cout << "  Dtype: " << tensor.dtype() << std::endl;
    std::cout << "  Elements: " << tensor.numel() << std::endl;
    
    if (tensor.numel() > 0) {
        auto cpu_tensor = tensor.to(torch::kCPU).to(torch::kFloat32);
        auto flat = cpu_tensor.flatten();
        
        int64_t print_count = std::min(static_cast<int64_t>(max_elements), flat.size(0));
        
        std::cout << "  Values: [";
        for (int64_t i = 0; i < print_count; ++i) {
            std::cout << flat[i].item<float>();
            if (i < print_count - 1) std::cout << ", ";
        }
        if (flat.size(0) > print_count) {
            std::cout << ", ...";
        }
        std::cout << "]" << std::endl;
        
        // Print statistics
        auto stats = computeTensorStats(tensor);
        std::cout << "  Stats: min=" << stats.at("min") 
                  << ", max=" << stats.at("max")
                  << ", mean=" << stats.at("mean")
                  << ", std=" << stats.at("std") << std::endl;
    }
    
    std::cout << "===========================" << std::endl;
}

std::pair<bool, std::string> checkTensorValidity(const torch::Tensor& tensor) {
    if (tensor.numel() == 0) {
        return {true, "Empty tensor"};
    }
    
    auto has_nan = torch::isnan(tensor).any().item<bool>();
    auto has_inf = torch::isinf(tensor).any().item<bool>();
    
    if (has_nan && has_inf) {
        return {false, "Contains both NaN and infinite values"};
    } else if (has_nan) {
        return {false, "Contains NaN values"};
    } else if (has_inf) {
        return {false, "Contains infinite values"};
    } else {
        return {true, "Valid tensor"};
    }
}

std::map<std::string, float> computeTensorStats(const torch::Tensor& tensor) {
    std::map<std::string, float> stats;
    
    if (tensor.numel() == 0) {
        stats["min"] = 0.0f;
        stats["max"] = 0.0f;
        stats["mean"] = 0.0f;
        stats["std"] = 0.0f;
        return stats;
    }
    
    auto float_tensor = tensor.to(torch::kFloat32);
    
    stats["min"] = torch::min(float_tensor).item<float>();
    stats["max"] = torch::max(float_tensor).item<float>();
    stats["mean"] = torch::mean(float_tensor).item<float>();
    stats["std"] = torch::std(float_tensor).item<float>();
    
    return stats;
}

torch::Tensor prepend_leading_dims(
    const torch::Tensor& tensor,
    int64_t target_ndim
) {
    int64_t current_ndim = tensor.dim();
    if (current_ndim >= target_ndim) {
        return tensor;
    }

    int64_t dims_to_add = target_ndim - current_ndim;
    std::vector<int64_t> new_shape(dims_to_add, 1);
    auto original_shape = tensor.sizes();
    new_shape.insert(new_shape.end(), original_shape.begin(), original_shape.end());

    return tensor.view(new_shape);
}

        } // namespace util_tensors
    } // namespace core
} // namespace contorchionist

#ifndef CORE_UTIL_TENSORS_H
#define CORE_UTIL_TENSORS_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include "contorchionist_core/contorchionist_core_export.h"

namespace contorchionist {
    namespace core {
        namespace util_tensors {

/**
 * @brief Normalize tensor values to the range [-1, 1]
 * 
 * Scales tensor values to fit within the range [-1, 1], useful for
 * audio signals and some neural network applications.
 * 
 * @param tensor Input tensor to normalize
 * @param min_val Minimum value in output range (default: -1.0)
 * @param max_val Maximum value in output range (default: 1.0)
 * @return torch::Tensor Normalized tensor
 */
CONTORCHIONIST_CORE_EXPORT torch::Tensor normalize(
    const torch::Tensor& tensor,
    float min_val = -1.0f,
    float max_val = 1.0f
);

/**
 * @brief Clamp tensor values to a specific range
 * 
 * @param tensor Input tensor
 * @param min_val Minimum allowed value
 * @param max_val Maximum allowed value
 * @return torch::Tensor Clamped tensor
 */
CONTORCHIONIST_CORE_EXPORT torch::Tensor clamp(
    const torch::Tensor& tensor,
    float min_val,
    float max_val
);

/**
 * @brief Convert tensor to a specific device safely
 * 
 * @param tensor Input tensor
 * @param device Target device
 * @return torch::Tensor Tensor moved to target device
 */
CONTORCHIONIST_CORE_EXPORT torch::Tensor toDevice(
    const torch::Tensor& tensor,
    const torch::Device& device
);

/**
 * @brief Validate tensor shape matches expected shape
 * 
 * @param tensor Input tensor to validate
 * @param expected_shape Expected shape vector
 * @param tensor_name Name for error messages
 * @return true if shape matches, false otherwise
 */
CONTORCHIONIST_CORE_EXPORT bool validateShape(
    const torch::Tensor& tensor,
    const std::vector<int64_t>& expected_shape,
    const std::string& tensor_name = "tensor"
);

/**
 * @brief Get string representation of tensor shape
 * 
 * @param tensor Input tensor
 * @return std::string Shape as string (e.g., "[2, 3, 4]")
 */
CONTORCHIONIST_CORE_EXPORT std::string shapeToString(const torch::Tensor& tensor);

/**
 * @brief Get string representation of shape vector
 * 
 * @param shape Shape vector
 * @return std::string Shape as string (e.g., "[2, 3, 4]")
 */
CONTORCHIONIST_CORE_EXPORT std::string shapeToString(const std::vector<int64_t>& shape);

/**
 * @brief Calculate total number of elements in a shape
 * 
 * @param shape Shape vector
 * @return int64_t Total number of elements
 */
CONTORCHIONIST_CORE_EXPORT int64_t calculateNumElements(const std::vector<int64_t>& shape);

/**
 * @brief Check if two shapes are compatible for broadcasting
 * 
 * @param shape1 First shape
 * @param shape2 Second shape
 * @return true if shapes are broadcast compatible
 */
CONTORCHIONIST_CORE_EXPORT bool isBroadcastCompatible(
    const std::vector<int64_t>& shape1,
    const std::vector<int64_t>& shape2
);

/**
 * @brief Compute the broadcast result shape
 * 
 * @param shape1 First shape
 * @param shape2 Second shape
 * @return std::vector<int64_t> Resulting broadcast shape
 */
CONTORCHIONIST_CORE_EXPORT std::vector<int64_t> computeBroadcastShape(
    const std::vector<int64_t>& shape1,
    const std::vector<int64_t>& shape2
);

/**
 * @brief Create a tensor filled with a specific value
 * 
 * @param shape Desired tensor shape
 * @param value Fill value
 * @param dtype Tensor data type
 * @param device Target device
 * @return torch::Tensor Filled tensor
 */
CONTORCHIONIST_CORE_EXPORT torch::Tensor createFilledTensor(
    const std::vector<int64_t>& shape,
    float value,
    torch::ScalarType dtype = torch::kFloat32,
    const torch::Device& device = torch::kCPU
);

/**
 * @brief Safely reshape tensor with error checking
 * 
 * @param tensor Input tensor
 * @param new_shape Target shape
 * @param allow_copy Whether to allow copying if view is not possible
 * @return std::pair<torch::Tensor, bool> Reshaped tensor and success flag
 */
CONTORCHIONIST_CORE_EXPORT std::pair<torch::Tensor, bool> safeReshape(
    const torch::Tensor& tensor,
    const std::vector<int64_t>& new_shape,
    bool allow_copy = true
);

/**
 * @brief Convert tensor to CPU and extract as vector
 * 
 * @param tensor Input tensor (must be 1D)
 * @return std::vector<float> Vector containing tensor data
 */
CONTORCHIONIST_CORE_EXPORT std::vector<float> tensorToVector(const torch::Tensor& tensor);

/**
 * @brief Create tensor from vector
 * 
 * @param data Input vector
 * @param device Target device
 * @return torch::Tensor Created tensor
 */
CONTORCHIONIST_CORE_EXPORT torch::Tensor vectorToTensor(
    const std::vector<float>& data,
    const torch::Device& device = torch::kCPU
);

/**
 * @brief Print tensor information for debugging
 * 
 * @param tensor Input tensor
 * @param name Name for the tensor (used in output)
 * @param max_elements Maximum number of elements to print
 */
CONTORCHIONIST_CORE_EXPORT void printTensorInfo(
    const torch::Tensor& tensor,
    const std::string& name = "tensor",
    int max_elements = 10
);

/**
 * @brief Check if tensor contains NaN or infinite values
 * 
 * @param tensor Input tensor
 * @return std::pair<bool, std::string> Has invalid values and description
 */
CONTORCHIONIST_CORE_EXPORT std::pair<bool, std::string> checkTensorValidity(
    const torch::Tensor& tensor
);

/**
 * @brief Compute tensor statistics (mean, std, min, max)
 * 
 * @param tensor Input tensor
 * @return std::map<std::string, float> Statistics map
 */
CONTORCHIONIST_CORE_EXPORT std::map<std::string, float> computeTensorStats(
    const torch::Tensor& tensor
);

/**
 * @brief Prepends leading singleton dimensions to a tensor until it reaches a target number of dimensions.
 * 
 * If the tensor already has target_ndim or more dimensions, it is returned unchanged.
 * 
 * @param tensor The input tensor.
 * @param target_ndim The desired number of dimensions.
 * @return torch::Tensor The tensor with prepended dimensions, or the original tensor.
 */
CONTORCHIONIST_CORE_EXPORT torch::Tensor prepend_leading_dims(
    const torch::Tensor& tensor,
    int64_t target_ndim
);

        } // namespace util_tensors
    } // namespace core
} 

#endif // CORE_UTIL_TENSORS_H

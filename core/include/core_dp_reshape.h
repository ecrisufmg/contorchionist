#ifndef CORE_DP_RESHAPE_H
#define CORE_DP_RESHAPE_H

#include <torch/torch.h>
#include <string>
#include <vector>
#include <functional>
#include <map>
#include "contorchionist_core/contorchionist_core_export.h"



namespace contorchionist {
    namespace core {
        namespace dp_reshape {


struct ReshapeParams {
    // view/reshape
    std::vector<int64_t> shape;
    // permute
    std::vector<int64_t> dims;
    // squeeze/unsqueeze/transpose/flatten
    int64_t dim1 = 0;
    // flatten/transpose
    int64_t dim2 = -1;
};

struct ReshapeResult {
    bool success = false;
    at::Tensor output_tensor;
    std::string error_message;
};

struct ArgumentValidationResult {
    bool valid = false;
    std::string error_message;
    std::string method_name;
    ReshapeParams params;
};

struct ReshapeConfigurationResult {
    bool success = false;
    std::string error_message;
    std::string method_name;
    ReshapeParams params;
};

// apply reshape method 
CONTORCHIONIST_CORE_EXPORT
ReshapeResult ReshapeProcessor(
    const at::Tensor& input_tensor,
    const std::string& method,
    const ReshapeParams& params,
    const torch::Device& device);

// list available reshape methods
CONTORCHIONIST_CORE_EXPORT
std::vector<std::string> list_available_methods();

// check if method is valid
CONTORCHIONIST_CORE_EXPORT
bool is_valid_method(const std::string& method);

// compute output shape
CONTORCHIONIST_CORE_EXPORT
std::vector<int64_t> calculate_output_shape(
    const std::vector<int64_t>& input_shape,
    const std::string& method,
    const ReshapeParams& params
);

// parse arguments as a list
CONTORCHIONIST_CORE_EXPORT
ArgumentValidationResult parse_individual_arguments(
    const std::string& method,
    const std::vector<float>& args
);

// parse arguments individually
CONTORCHIONIST_CORE_EXPORT
ArgumentValidationResult parse_individual_arguments(
    const std::string& method,
    const std::vector<int64_t>& shape,   // shape for view/reshape
    int64_t dim1,                        // dim1 for flatten/squeeze/unsqueeze/transpose
    int64_t dim2,                        // dim2 for flatten/transpose (or -1)
    const std::vector<int64_t>& dims     // dims for permute
);

// setup reshape method and parameters
CONTORCHIONIST_CORE_EXPORT
ReshapeConfigurationResult set_reshape_method(
    const std::string& method_name,
    const std::vector<float>& args,
    const std::vector<int64_t>& shape,
    int64_t dim1,
    int64_t dim2,
    const std::vector<int64_t>& dims
);





        } // namespace dp_reshape
    } // namespace core
} // namespace contorchionist

#endif // CORE_DP_RESHAPE_H
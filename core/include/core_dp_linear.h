#include <torch/torch.h>
#include <memory>
#include "contorchionist_core/contorchionist_core_export.h"

namespace contorchionist {
    namespace core {
        namespace dp_linear {



// result structure for linear layer processing
struct LinearResult {
    bool success = false;
    std::string error_message;
    at::Tensor output;
};

/**
 * create a fully connected linear layer
* @param in_features Number of input features
* @param out_features Number of output features
* @param bias Whether to include a bias term
* @param device The device to create the layer on
*/
CONTORCHIONIST_CORE_EXPORT
std::shared_ptr<torch::nn::Linear> create_linear_layer(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    const torch::Device& device
);

 /**
 * forwards a fully connected linear layer
 * @param input Input tensor
 * @param layer Linear layer
 */
CONTORCHIONIST_CORE_EXPORT
LinearResult LinearProcessor(
    const at::Tensor& input,
    const std::shared_ptr<torch::nn::Linear>& layer);


        } // namespace dp_linear
    } // namespace core
} // namespace contorchionist
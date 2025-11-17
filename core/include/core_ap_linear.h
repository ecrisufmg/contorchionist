#ifndef CORE_AP_LINEAR_H
#define CORE_AP_LINEAR_H


#include <torch/torch.h>
#include "contorchionist_core/contorchionist_core_export.h" // Include the generated export header

namespace contorchionist {
    namespace core {
        namespace ap_linear {

/**
* Process signal blocks using a torch::nn::Linear layer
* @param linear Linear layer instance
* @param device Target device for computation
* @param in_buf Input buffer (audio samples)
* @param in_features Number of input features
* @return Output tensor from linear layer
*/
    // LinearAProcessor processes signal blocks using a torch::nn::Linear layer.
CONTORCHIONIST_CORE_EXPORT torch::Tensor LinearAProcessorSignal(
    torch::nn::Linear& linear,
    const torch::Device& device,
    const float* in_buf,
    int in_features
);


     } // namespace ap_linear
    } // namespace core
} // namespace contorchionist

#endif


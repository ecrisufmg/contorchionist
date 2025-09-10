#ifndef CORE_DP_TORCHTS_H
#define CORE_DP_TORCHTS_H

#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include "contorchionist_core/contorchionist_core_export.h"

namespace contorchionist {
    namespace core {
        namespace dp_torchts {

            /**
             * Execute model inference with input tensor
             * 
             * @param model TorchScript model (must be valid)
             * @param method_name Method to call (must exist in model)
             * @param input_tensor Input tensor (will be moved to model's device)
             * @param device Target device for computation
             * @return Output tensor from model inference
             * @throws c10::Error if any PyTorch operation fails
             */
            CONTORCHIONIST_CORE_EXPORT at::Tensor TorchTSProcessor(
                torch::jit::script::Module* model,
                const std::string& method_name,
                const at::Tensor& input_tensor,
                const torch::Device& device
            );

        } // namespace dp_torchts
    } // namespace core
} // namespace contorchionist

#endif // CORE_DP_TORCHTS_H
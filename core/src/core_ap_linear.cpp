#include "core_ap_linear.h"
#include <torch/torch.h>



namespace contorchionist {
    namespace core {
        namespace ap_linear {

// Processes signal blocks using a torch::nn::Linear layer.
torch::Tensor LinearAProcessorSignal(torch::nn::Linear& linear, const torch::Device& device, const float* in_buf, int in_features){

    try{

        // c10::InferenceMode guard;

        // create input tensor from the input buffer
        at::Tensor input = torch::from_blob((void*)in_buf, {1, in_features}, torch::TensorOptions().dtype(torch::kFloat32)).clone();

        // move the input tensor to the specified device
        input = input.to(device);

        // Forward
        // at::Tensor output = linear->forward(input);
        
        at::Tensor output = linear(input);

        // return the output tensor
        return output;
        
    } catch (const c10::Error& e) {
        // Re-throw PyTorch errors
        throw;
    } catch (const std::exception& e) {
    // Convert to runtime_error for consistency
        throw std::runtime_error("LinearAProcessor failed: " + std::string(e.what()));
    }
}
        } // namespace ap_linear
    } // namespace core
} // namespace contorchionist


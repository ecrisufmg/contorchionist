#include <torch/torch.h>
#include <torch/script.h>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include "core_dp_torchts.h"
#include <c10/util/Exception.h>
#include <ATen/core/ATenGeneral.h>




namespace contorchionist {
    namespace core {
        namespace dp_torchts {

at::Tensor TorchTSProcessor(
    torch::jit::script::Module* model, // Pointer to the TorchScript model
    const std::string& method_name, // Name of the method to call
    const at::Tensor& input_tensor, // Input tensor for the model
    const torch::Device& device) { // Device to run the model on (CPU or GPU)

    try {
        // Enable inference mode for performance
        c10::InferenceMode guard;
                    
        // Move input tensor to target device
         at::Tensor device_input = input_tensor.to(device);
                    
        // Prepare input for model
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(device_input);
                    
        // Execute model and return result
        torch::jit::IValue result = model->get_method(method_name)(inputs);
                    
        // Ensure result is a tensor
        if (!result.isTensor()) {
            AT_ERROR("Model output is not a tensor");
        }

        return result.toTensor();

    } catch (const c10::Error& e) {
        // Re-throw PyTorch errors com contexto extra
        AT_ERROR("Inference execution failed: ", e.what());
    } catch (const std::exception& e) {
        // Handle standard library exceptions
        AT_ERROR("Inference execution failed: ", e.what());
    } catch (...) {
        // Handle any other exceptions
        AT_ERROR("Inference execution failed: Unknown error");
    }
}

        } // namespace dp_torchts
    } // namespace core
} // namespace contorchionist
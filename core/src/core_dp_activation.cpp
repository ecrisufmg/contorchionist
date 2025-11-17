#include <torch/torch.h>
#include <string>
#include <map>
#include <vector>
#include <functional>
#include <stdexcept>
#include "core_dp_activation.h"




namespace contorchionist {
    namespace core {
        namespace dp_activation {

// maping string to activation functions
const std::map<std::string, ActivationFunction>& get_activation_function_map() {
    static const std::map<std::string, ActivationFunction> activation_map = {
        // Functions without parameters
        {"relu", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::relu(tensor);
        }},
        {"selu", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::selu(tensor);
        }},
        {"silu", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::silu(tensor);
        }},
        {"relu6", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::relu6(tensor);
        }},
        {"sigmoid", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::sigmoid(tensor);
        }},
        {"tanh", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::tanh(tensor);
        }},
        {"swish", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return tensor * torch::sigmoid(tensor);
        }},
        {"hardswish", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::hardswish(tensor);
        }},
        {"hardsigmoid", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::hardsigmoid(tensor);
        }},
        {"softplus", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::softplus(tensor);
        }},
        {"gelu", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::gelu(tensor, "none");
        }},
        {"logsigmoid", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::log_sigmoid(tensor);
        }},
        {"mish", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return tensor * torch::tanh(torch::softplus(tensor));
        }},
        {"tanhshrink", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return tensor - torch::tanh(tensor);
        }},

        // Functions with fixed parameters
        {"hardtanh", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::hardtanh(tensor, -1.0, 1.0);
        }},
        {"threshold", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::threshold(tensor, 0.0, 0.0);
        }},
        {"rrelu", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::rrelu(tensor, 0.24, 0.42);
        }},

        // Functions with alpha parameter
        {"leakyrelu", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::leaky_relu(tensor, alpha);
        }},
        {"elu", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::elu(tensor, alpha);
        }},
        {"celu", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::celu(tensor, alpha);
        }},

         // Functions with lambda parameter
        {"softshrink", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::softshrink(tensor, lambda);
        }},
        {"hardshrink", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::hardshrink(tensor, lambda);
        }},

        // Functions with dim parameter
        {"softmax", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::softmax(tensor, dim);
        }},
        {"logsoftmax", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::log_softmax(tensor, dim);
        }},
        {"glu", [](const at::Tensor& tensor, double alpha, double lambda, int64_t dim) {
            return torch::glu(tensor, dim);
        }}
    };
    return activation_map;
}

// get activation function by name
ActivationFunction get_activation_function(const std::string& activation_name) {
    const auto& activation_map = get_activation_function_map();
    auto it = activation_map.find(activation_name);
    return (it != activation_map.end()) ? it->second : nullptr;
}

// Activation processing function
at::Tensor ActivationProcessor(
    const at::Tensor& input_tensor, // Input tensor
    const std::string& activation_name, // Activation function name
    const torch::Device& device, // Target device
    double alpha, // Alpha parameter
    double lambda, // Lambda parameter
    int64_t dim) { // Dimension parameter

    try {
        // Enable inference mode for performance
        // c10::InferenceMode guard;
                    
        // Move input tensor to target device
        at::Tensor tensor = input_tensor.to(device);
                    
        // Get activation function from map
        auto func = get_activation_function(activation_name);
        if (!func) {
            throw std::invalid_argument("Unsupported activation function: " + activation_name);
        }
                    
        // Execute activation function
        at::Tensor result = func(tensor, alpha, lambda, dim);

        
        // return the result tensor
        return result;
                    
    } catch (const c10::Error& e) {
        throw;
    } catch (const std::exception& e) {
        throw std::runtime_error("ActivationProcessor failed: " + std::string(e.what()));
    }
}


// List all available activation functions
std::vector<std::string> list_available_activations() {
    const auto& activation_map = get_activation_function_map();
    std::vector<std::string> activation_names;
    activation_names.reserve(activation_map.size());
                
    for (const auto& pair : activation_map) {
        activation_names.push_back(pair.first);
    }
                
    return activation_names;
}
            










        } // namespace dp_activation
    } // namespace core
} // namespace contorchionist
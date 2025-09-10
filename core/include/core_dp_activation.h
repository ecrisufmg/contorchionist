#ifndef CORE_DP_ACTIVATION_H
#define CORE_DP_ACTIVATION_H


#include <torch/torch.h>
#include <string>
#include <map>
#include <vector>
#include <functional>
#include "contorchionist_core/contorchionist_core_export.h"



namespace contorchionist {
    namespace core {
        namespace dp_activation {

/**
* signature: (tensor, alpha, lambda, dim) -> tensor
*/
using ActivationFunction = std::function<at::Tensor(const at::Tensor&, double, double, int64_t)>;

/**
* get the activation function map containing all available activation functions
* @return reference to static map of activation functions
*/
CONTORCHIONIST_CORE_EXPORT
const std::map<std::string, ActivationFunction>& get_activation_function_map();

/**
* get activation function by name
* @param activation_name name of the activation function
* @return activationFunction lambda or nullptr if not found
*/
CONTORCHIONIST_CORE_EXPORT
ActivationFunction get_activation_function(const std::string& activation_name);


/**
* ActivationProcessor  to input tensor
* @param input_tensor input tensor to process
* @param activation_name name of activation function
* @param device target device for computation
* @param alpha alpha parameter for leaky_relu, elu, celu (default: 0.01)
* @param lambda lambda parameter for softshrink, hardshrink (default: 0.5)
* @param dim dimension parameter for softmax, logsoftmax, glu (default: -1)
* @return output tensor after activation on target device
*/
CONTORCHIONIST_CORE_EXPORT
at::Tensor ActivationProcessor(
    const at::Tensor& input_tensor,
    const std::string& activation_name,
    const torch::Device& device,
    double alpha = 0.01,
    double lambda = 0.5,
    int64_t dim = -1
);

/**
* list all available activation function names
* @return list of available activation function names
*/
CONTORCHIONIST_CORE_EXPORT
std::vector<std::string> list_available_activations();

        } // namespace dp_activation
    } // namespace core
} // namespace contorchionist

#endif // CORE_DP_ACTIVATION_H
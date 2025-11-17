#ifndef CORE_DP_SEQUENTIAL_H
#define CORE_DP_SEQUENTIAL_H

#include <torch/torch.h>
#include <string>
#include <vector>
#include <functional>
#include "contorchionist_core/contorchionist_core_export.h"



namespace contorchionist {
    namespace core {
        namespace dp_sequential {


/**
* process input tensor through sequential container
* @param input_tensor input tensor to process
* @param container sequential container to forward through
* @param device target device for computation
* @return output tensor after forward pass
*/

CONTORCHIONIST_CORE_EXPORT
at::Tensor SequentialProcessor(
    const at::Tensor& input_tensor,
    std::shared_ptr<torch::nn::Sequential> container,
    const torch::Device& device
);


/**
* Simple training result structure
*/
struct TrainingResult {
    bool success;
    float final_loss;
    std::string error_message;
};

/**
* Progress callback function type
* Parameters: (epoch, current_loss)
*/
using ProgressCallback = std::function<void(int64_t, float)>;

/**
* training loop 
* @param container Sequential container (already moved to device)
* @param dataset_tensor Training dataset (already moved to device)
* @param target_tensor Target tensor (already moved to device, can be empty)
* @param loss_function Loss function to use
* @param optimizer Optimizer for weight updates (already configured)
* @param num_epochs Number of training epochs
* @param device Target device for computation
* @param use_target Whether to use target tensor or input as target
* @param progress_callback Optional callback for progress updates (can be nullptr)
* @return TrainingResult with success status and final loss
*/
CONTORCHIONIST_CORE_EXPORT
TrainingResult SequentialTrainer(
    std::shared_ptr<torch::nn::Sequential> container,
    const at::Tensor& dataset_tensor,
    const at::Tensor& target_tensor,
    std::function<at::Tensor(const at::Tensor&, const at::Tensor&)> loss_function,
    torch::optim::Optimizer* optimizer,
    int64_t num_epochs,
    const torch::Device& device,
    bool use_target,
    ProgressCallback progress_callback = nullptr
);


/**
* Create optimizer for a given module
* @param opt_name optimizer name (adam, sgd, rmsprop, etc.)
* @param module module to optimize
* @param learning_rate learning rate for optimizer
* @param out_type output type for the optimizer
* @return optimizer
*/
CONTORCHIONIST_CORE_EXPORT
torch::optim::Optimizer* create_optimizer(
    const std::string& opt_name, 
    torch::nn::Module& module, 
    double learning_rate, 
    std::string& out_type
);


/**
* create loss function by name
* @param loss_name loss function name (mse, cross_entropy, bce, etc.)
* @return loss function or nullptr if invalid
*/
CONTORCHIONIST_CORE_EXPORT
std::function<at::Tensor(const at::Tensor&, const at::Tensor&)> create_loss_function(
    const std::string& loss_name
);

/**
* Validation result structure
*/
struct ValidationResult {
    bool is_valid;
    std::string error_message;
    std::string expected_type;
    std::string actual_type;
};

/**
* Check if target tensor is compatible with loss function
* @param loss_name Loss function name
* @param target Target tensor to validate
* @return ValidationResult with compatibility information
*/
CONTORCHIONIST_CORE_EXPORT
ValidationResult check_loss_target_compatibility(
    const std::string& loss_name, 
    const at::Tensor& target
);

// helper opcional para formatar shape como "1 3 4"
CONTORCHIONIST_CORE_EXPORT
std::string shape_to_string(const std::vector<int64_t>& shape);

// print layer information
struct LayerPrintInfo {
    std::string friendly_name;          // ex.: "Linear", "LeakyReLU", "Mha"
    std::vector<int64_t> input_shape;   // input shape
    std::vector<int64_t> output_shape;  // output shape
    bool is_activation = false; 
    bool print_shape  = true;
};

// summarize the layers in a sequential container
CONTORCHIONIST_CORE_EXPORT
std::vector<LayerPrintInfo> summarize_sequential(
    const std::shared_ptr<torch::nn::Sequential>& seq,
    const std::vector<int64_t>& input_shape);


// reset the parameters of all layers in a sequential container
CONTORCHIONIST_CORE_EXPORT
void reset_sequential_parameters(const std::shared_ptr<torch::nn::Sequential>& seq);


// weight initialization methods
enum class InitMethod {
    KaimingUniform,
    KaimingNormal,
    XavierUniform,
    XavierNormal,
    Uniform,
    Normal,
    Constant
};

// parameter initialization options
struct InitOptions {
    // for Kaiming
    double a = 0.0;                // negative_slope (LeakyReLU)
    std::string nonlinearity = "relu"; // "relu" ou "leaky_relu"
    // for Xavier/Uniform/Normal/Constant
    double gain = 1.0;
    double low = 0;  // Uniform
    double high = 1;  // Uniform
    double mean = 0.0;   // Normal
    double std = 0.02;   // Normal
    double constant = 0.0; // Constant
    // Bias
    bool zero_bias = true;
};

// parse initialization method from string
CONTORCHIONIST_CORE_EXPORT
InitMethod parse_init_method(const std::string& method_str, bool* ok = nullptr);

// convert enum to string
CONTORCHIONIST_CORE_EXPORT
std::string init_method_to_string(InitMethod m);

// initialize weights (Linear, MHA e Wrapper de MHA)
CONTORCHIONIST_CORE_EXPORT
void initialize_sequential_weights(const std::shared_ptr<torch::nn::Sequential>& seq, // sequential
                                      InitMethod method, // initialization method
                                      const InitOptions& opts = {} // options for initialization
); 

// set random seed for reproducibility
CONTORCHIONIST_CORE_EXPORT
void set_random_seed(int64_t seed, bool verbose = false);


        } // namespace dp_sequential
    } // namespace core
} // namespace contorchionist

#endif // CORE_DP_SEQUENTIAL_H
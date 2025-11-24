#include <torch/torch.h>
#include <string>
#include <map>
#include <vector>
#include <functional>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <torch/nn/init.h>
#include "core_dp_sequential.h"
#include "core_dp_mha.h"


namespace contorchionist {
    namespace core {
        namespace dp_sequential {


at::Tensor SequentialProcessor(
    const at::Tensor& input_tensor,
    std::shared_ptr<torch::nn::Sequential> container,
    const torch::Device& device) {
    
    try{
        c10::InferenceMode guard;

        // Move input tensor to target device
        at::Tensor tensor = input_tensor.to(device);
                    
        // Forward through sequential container
        at::Tensor output = container->get()->forward(tensor);
                    
        return output;  

    } catch (const c10::Error& e) {
        throw std::runtime_error("PyTorch error in SequentialProcessor: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw std::runtime_error("SequentialProcessor failed: " + std::string(e.what()));
    }
}

// training the sequential model
TrainingResult SequentialTrainer(
    std::shared_ptr<torch::nn::Sequential> container,
    const at::Tensor& dataset_tensor,
    const at::Tensor& target_tensor,
    std::function<at::Tensor(const at::Tensor&, const at::Tensor&)> loss_function,
    torch::optim::Optimizer* optimizer,
    int64_t num_epochs,
    const torch::Device& device,
    bool use_target,
    ProgressCallback progress_callback) {

    TrainingResult result;
    result.success = false;
    result.final_loss = 0.0f;

    try {
        // // Set container to training mode
        // container->train();

        // Get references and basic info
        auto& dataset = dataset_tensor;
        auto& targets = target_tensor;
        int64_t num_samples = dataset.size(0);

        // Training loop through all epochs
        for (int64_t epoch = 0; epoch < num_epochs; ++epoch) {
            float epoch_loss = 0.0f;

            // For each epoch, iterate through all examples in dataset
            for (int64_t i = 0; i < num_samples; ++i) {
                // Get sample i (shape: input_shape)
                at::Tensor input = dataset[i];
                at::Tensor target;

                // If the target tensor was defined, it will be used as a target (classification/regression, etc.); if not, the input will be used as a target (autoencoder/autoatenção, etc.).
                if (use_target && targets.defined()) {
                    target = targets[i];
                } else {
                    target = input.clone();
                }

                // Move tensors to device
                input = input.to(device);
                target = target.to(device);

                // Forward pass through container
                at::Tensor output = container->get()->forward(input);

                // Validate output shape matches target shape
                if (!output.sizes().equals(target.sizes())) {
                    result.error_message = "Output shape doesn't match target shape at epoch " + 
                                            std::to_string(epoch) + ", sample " + std::to_string(i);
                    return result;
                }

                // Calculate loss
                at::Tensor loss = loss_function(output, target);
                epoch_loss += loss.item<float>();

                // Backpropagation and weight update
                optimizer->zero_grad();
                loss.backward();
                optimizer->step();
            }

            // Calculate average loss for this epoch
            float avg_loss = epoch_loss / num_samples;

            // Call progress callback if provided (every 10 epochs)
            if (progress_callback && epoch % 10 == 0) {
                progress_callback(epoch, avg_loss);
            }

            // Store final loss
            result.final_loss = avg_loss;
        }

        result.success = true;
        return result;

    } catch (const c10::Error& e) {
        result.error_message = "PyTorch error: " + std::string(e.what());
        return result;
    } catch (const std::exception& e) {
        result.error_message = "Training error: " + std::string(e.what());
        return result;
    }
}

// create optimizer
torch::optim::Optimizer* create_optimizer(
    const std::string& opt_name,
    torch::nn::Module& module,
    double learning_rate,
    std::string& out_type) {

    try{
        if (opt_name == "adam") {
            out_type = "adam";
            return new torch::optim::Adam(module.parameters(), torch::optim::AdamOptions(learning_rate));
        } else if (opt_name == "adamw") {
            out_type = "adamw";
            return new torch::optim::AdamW(module.parameters(), torch::optim::AdamWOptions(learning_rate));
        } else if (opt_name == "adagrad") {
            out_type = "adagrad";
            return new torch::optim::Adagrad(module.parameters(), torch::optim::AdagradOptions(learning_rate));
        } else if (opt_name == "lbfgs") {
            out_type = "lbfgs";
            return new torch::optim::LBFGS(module.parameters(), torch::optim::LBFGSOptions(learning_rate));
        } else if (opt_name == "rmsprop") {
            out_type = "rmsprop";
            return new torch::optim::RMSprop(module.parameters(), torch::optim::RMSpropOptions(learning_rate));
        } else if (opt_name == "sgd") {
            out_type = "sgd";
            return new torch::optim::SGD(module.parameters(), torch::optim::SGDOptions(learning_rate));
        } else {
            throw std::invalid_argument("Unknown optimizer: " + opt_name);
        }
    } catch (const c10::Error& e) {
        throw std::runtime_error("PyTorch error creating optimizer: " + std::string(e.what()));
    }
}

//list all available optimizers
std::vector<std::string> list_available_optimizers() {
    return {
        "adam", "adamw", "adagrad", "lbfgs", "rmsprop", "sgd"
    };
}


// creat loss function
std::function<at::Tensor(const at::Tensor&, const at::Tensor&)> create_loss_function(
    const std::string& loss_name
) {
   
    if (loss_name == "mse" || loss_name == "mse_loss") {
        return [](const at::Tensor& output, const at::Tensor& target) {
            return torch::mse_loss(output, target);
        };
    } else if (loss_name == "l1" || loss_name == "l1_loss") {
        return [](const at::Tensor& output, const at::Tensor& target) {
            return torch::l1_loss(output, target);
        };
    } else if (loss_name == "smoothl1" || loss_name == "sl1") {
        return [](const at::Tensor& output, const at::Tensor& target) {
            return torch::smooth_l1_loss(output, target);
        };
        // } else if (loss_name == "poisson_nll" || loss_name == "poisson_nll_loss") {
        //     return [](const at::Tensor& output, const at::Tensor& target) {
        //         return torch::poisson_nll_loss(output, target);
        //     };
    } else if (loss_name == "cross_entropy" || loss_name == "ce") {
        return [](const at::Tensor& output, const at::Tensor& target) {
            // target deve ser long/int64 com índices das classes
            return torch::nn::functional::cross_entropy(output, target);
        };
    } else if (loss_name == "nll" || loss_name == "nll_loss") {
        return [](const at::Tensor& output, const at::Tensor& target) {
            return torch::nn::functional::nll_loss(output, target);
        };
    } else if (loss_name == "bce" || loss_name == "binary_cross_entropy") {
        return [](const at::Tensor& output, const at::Tensor& target) {
            return torch::nn::functional::binary_cross_entropy(output, target);
        };
    } else if (loss_name == "bcelogits" || loss_name == "binary_cross_entropy_with_logits") {
        return [](const at::Tensor& output, const at::Tensor& target) {
            return torch::nn::functional::binary_cross_entropy_with_logits(output, target);
        };
    } else if (loss_name == "kldiv" || loss_name == "kl_divergence") {
        return [](const at::Tensor& output, const at::Tensor& target) {
            return torch::nn::functional::kl_div(output, target);
        };
        // } else if (loss_name == "cosine_embedding") {
        //     return [](const at::Tensor& output, const at::Tensor& target) {
        //         // target deve ser 1 ou -1
        //         return torch::nn::functional::cosine_embedding_loss(output, target);
        //     };
    } else if (loss_name == "hinge_embedding" || loss_name == "hinge") {
        return [](const at::Tensor& output, const at::Tensor& target) {
            return torch::nn::functional::hinge_embedding_loss(output, target);
        };
        // } else if (loss_name == "triplet_margin") {
        //     // Triplet margin loss requer três tensores: anchor, positive, negative
        //     if (pd_obj) pd_error(pd_obj, "triplet_margin_loss requer três tensores: anchor, positive, negative.");
        //     return [](const at::Tensor&, const at::Tensor&) {
        //         return at::Tensor();
        //     };
    } else if (loss_name == "multi_margin" || loss_name == "mm") {
        return [](const at::Tensor& output, const at::Tensor& target) {
            return torch::nn::functional::multi_margin_loss(output, target);
        };
    } else if (loss_name == "multilabelmargin" || loss_name == "mlm") {
        return [](const at::Tensor& output, const at::Tensor& target) {
            return torch::nn::functional::multilabel_margin_loss(output, target);
        };
    } else if (loss_name == "multilabel_soft_margin" || loss_name == "mlsm") {
        return [](const at::Tensor& output, const at::Tensor& target) {
            return torch::nn::functional::multilabel_soft_margin_loss(output, target);
        };
    } else if (loss_name == "soft_margin" || loss_name == "sm") {
        return [](const at::Tensor& output, const at::Tensor& target) {
            return torch::nn::functional::soft_margin_loss(output, target);
        };
    } else {
        throw std::invalid_argument("Unknown loss function: " + loss_name);
    }
}

std::vector<std::string> list_available_loss_functions() {
    return {
        "mse", "mse_loss", "l1", "l1_loss", "smooth_l1", "sl1", "smooth_l1_loss",
        "cross_entropy", "ce", "nll", "nll_loss", "bce", "binary_cross_entropy",
        "bce_logits", "binary_cross_entropy_with_logits", "kldiv", "kl_divergence",
        "hinge_embedding", "multi_margin", "mm", "multilabel_margin",
        "multilabel_soft_margin", "mlsm", "soft_margin", "sm"
    };
}


// check compatibility of target tensor with loss function
ValidationResult check_loss_target_compatibility(const std::string& loss_name, const at::Tensor& target) {
    ValidationResult result; 
    result.is_valid = true;
    result.error_message = "";

    if (!target.defined()) {
        result.is_valid = false;
        result.error_message = "Target tensor is not defined";
        return result;
    }

    try {
        if (loss_name == "cross_entropy" || loss_name == "ce") {
            if (target.dtype() != torch::kLong) {
                result.is_valid = false;
                result.expected_type = "Long (class indices)";
                result.actual_type = torch::toString(target.dtype());
                result.error_message = "cross_entropy expects target of type Long (class indices)";
            }
        } else if (loss_name == "nll" || loss_name == "nll_loss") {
            if (target.dtype() != torch::kLong) {
                result.is_valid = false;
                result.expected_type = "Long (class indices)";
                result.actual_type = torch::toString(target.dtype());
                result.error_message = "nll expects target of type Long (class indices)";
            }
        } else if (loss_name == "mse" || loss_name == "mse_loss" ||
                   loss_name == "l1" || loss_name == "l1_loss" ||
                   loss_name == "smooth_l1" || loss_name == "smooth_l1_loss") {
            if (target.dtype() != torch::kFloat && target.dtype() != torch::kDouble) {
                result.is_valid = false;
                result.expected_type = "Float or Double";
                result.actual_type = torch::toString(target.dtype());
                result.error_message = loss_name + " expects target of type Float or Double"; 
            }
        } else if (loss_name == "bce" || loss_name == "binary_cross_entropy" ||
                   loss_name == "bce_logits" || loss_name == "binary_cross_entropy_with_logits" || loss_name == "bcelogits") {
            // Check dtype
            if (target.dtype() != torch::kFloat && target.dtype() != torch::kDouble) {
                result.is_valid = false;
                result.expected_type = "Float or Double";
                result.actual_type = torch::toString(target.dtype());
                result.error_message = loss_name + " expects target of type Float or Double";
            }
            // Check value range [0, 1]
            else if (target.defined() && (target.min().item<float>() < 0.0f || target.max().item<float>() > 1.0f)) {
                result.is_valid = false;
                result.expected_type = "Values in range [0, 1]";
                result.actual_type = "Values outside [0, 1] range";
                result.error_message = loss_name + " expects target values in the interval [0, 1]";  
            }
        } else if (loss_name == "multi_margin" || loss_name == "multilabel_margin" || loss_name == "mm") {
            if (target.dtype() != torch::kLong) {
                result.is_valid = false;
                result.expected_type = "Long";
                result.actual_type = torch::toString(target.dtype());
                result.error_message = loss_name + " expects target of type Long"; 
            }
        }
        // Add other checks for other loss functions here

    } catch (const c10::Error& e) {
        result.is_valid = false;
        result.error_message = "PyTorch error during validation: " + std::string(e.what());
    } catch (const std::exception& e) {
        result.is_valid = false;
        result.error_message = "Error during validation: " + std::string(e.what());
    }

    return result;
} 

    
// get shape and convert to string
std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) oss << " ";
        oss << shape[i];
    }
    return oss.str();
}

// map torch::nn::Module types to friendly layer names
static std::string layer_name(const std::shared_ptr<torch::nn::Module>& child) {
    // basic layers
    if (std::dynamic_pointer_cast<torch::nn::LinearImpl>(child))      return "Linear";
    if (std::dynamic_pointer_cast<contorchionist::core::dp_mha::MHAWrapperImpl>(child)) return "Mha";

    // activation layers
    if (std::dynamic_pointer_cast<torch::nn::ReLUImpl>(child))        return "ReLU";
    if (std::dynamic_pointer_cast<torch::nn::SELUImpl>(child))        return "SELU";
    if (std::dynamic_pointer_cast<torch::nn::SiLUImpl>(child))        return "SiLU";
    if (std::dynamic_pointer_cast<torch::nn::ReLU6Impl>(child))       return "ReLU6";
    if (std::dynamic_pointer_cast<torch::nn::SigmoidImpl>(child))     return "Sigmoid";
    if (std::dynamic_pointer_cast<torch::nn::TanhImpl>(child))        return "Tanh";
    // if (std::dynamic_pointer_cast<torch::nn::HardswishImpl>(child))   return "Hardswish";
    // if (std::dynamic_pointer_cast<torch::nn::HardsigmoidImpl>(child)) return "Hardsigmoid";
    if (std::dynamic_pointer_cast<torch::nn::SoftplusImpl>(child))    return "Softplus";
    if (std::dynamic_pointer_cast<torch::nn::GELUImpl>(child))        return "GELU";
    if (std::dynamic_pointer_cast<torch::nn::LogSigmoidImpl>(child))  return "LogSigmoid";
    if (std::dynamic_pointer_cast<torch::nn::MishImpl>(child))        return "Mish";
    if (std::dynamic_pointer_cast<torch::nn::TanhshrinkImpl>(child))  return "Tanhshrink";
    if (std::dynamic_pointer_cast<torch::nn::HardtanhImpl>(child))    return "Hardtanh";
    if (std::dynamic_pointer_cast<torch::nn::ThresholdImpl>(child))   return "Threshold";
    if (std::dynamic_pointer_cast<torch::nn::RReLUImpl>(child))       return "RReLU";
    if (std::dynamic_pointer_cast<torch::nn::LeakyReLUImpl>(child))   return "LeakyReLU";
    if (std::dynamic_pointer_cast<torch::nn::ELUImpl>(child))         return "ELU";
    if (std::dynamic_pointer_cast<torch::nn::CELUImpl>(child))        return "CELU";
    if (std::dynamic_pointer_cast<torch::nn::SoftshrinkImpl>(child))  return "Softshrink";
    if (std::dynamic_pointer_cast<torch::nn::HardshrinkImpl>(child))  return "Hardshrink";
    if (std::dynamic_pointer_cast<torch::nn::SoftmaxImpl>(child))     return "Softmax";
    if (std::dynamic_pointer_cast<torch::nn::LogSoftmaxImpl>(child))  return "LogSoftmax";
    if (std::dynamic_pointer_cast<torch::nn::GLUImpl>(child))         return "GLU";

    // Fallback: usa o nome do módulo sem sufixo Impl
    std::string n = child->name();
    if (n.size() > 4 && n.substr(n.size() - 4) == "Impl") n.resize(n.size() - 4);
    return n;
}

// check if the module is an activation
static bool is_activation_module(const std::shared_ptr<torch::nn::Module>& child) {
    return
        std::dynamic_pointer_cast<torch::nn::ReLUImpl>(child)      ||
        std::dynamic_pointer_cast<torch::nn::SELUImpl>(child)      ||
        std::dynamic_pointer_cast<torch::nn::SiLUImpl>(child)      ||
        std::dynamic_pointer_cast<torch::nn::ReLU6Impl>(child)     ||
        std::dynamic_pointer_cast<torch::nn::SigmoidImpl>(child)   ||
        std::dynamic_pointer_cast<torch::nn::TanhImpl>(child)      ||
        std::dynamic_pointer_cast<torch::nn::SoftplusImpl>(child)  ||
        std::dynamic_pointer_cast<torch::nn::GELUImpl>(child)      ||
        std::dynamic_pointer_cast<torch::nn::LogSigmoidImpl>(child)||
        std::dynamic_pointer_cast<torch::nn::MishImpl>(child)      ||
        std::dynamic_pointer_cast<torch::nn::TanhshrinkImpl>(child)||
        std::dynamic_pointer_cast<torch::nn::HardtanhImpl>(child)  ||
        std::dynamic_pointer_cast<torch::nn::ThresholdImpl>(child) ||
        std::dynamic_pointer_cast<torch::nn::RReLUImpl>(child)     ||
        std::dynamic_pointer_cast<torch::nn::LeakyReLUImpl>(child) ||
        std::dynamic_pointer_cast<torch::nn::ELUImpl>(child)       ||
        std::dynamic_pointer_cast<torch::nn::CELUImpl>(child)      ||
        std::dynamic_pointer_cast<torch::nn::SoftshrinkImpl>(child)||
        std::dynamic_pointer_cast<torch::nn::HardshrinkImpl>(child)||
        std::dynamic_pointer_cast<torch::nn::SoftmaxImpl>(child)   ||
        std::dynamic_pointer_cast<torch::nn::LogSoftmaxImpl>(child)||
        std::dynamic_pointer_cast<torch::nn::GLUImpl>(child);
}


// summarize the sequential model layers
std::vector<LayerPrintInfo> summarize_sequential(
    const std::shared_ptr<torch::nn::Sequential>& seq,
    const std::vector<int64_t>& input_shape) {

    std::vector<LayerPrintInfo> out;
    if (!seq) return out;

    auto curr = input_shape;

    for (size_t i = 0; i < seq->get()->size(); ++i) {
        auto child = seq->get()->ptr(i);
        LayerPrintInfo info;
        info.friendly_name = layer_name(child);
        info.input_shape = curr;
        // linear
        if (auto linear = std::dynamic_pointer_cast<torch::nn::LinearImpl>(child)) {
            info.output_shape = curr;
            if (!info.output_shape.empty()) {
                info.output_shape.back() = linear->options.out_features();
            }
        } else if (std::dynamic_pointer_cast<contorchionist::core::dp_mha::MHAWrapperImpl>(child)) {
            // MHA shape [seq_len, batch, embed]
            info.output_shape = curr;
        } else {
            // activations
            info.output_shape = curr;
            if (is_activation_module(child)) {
                info.input_shape.clear();
            }
        }

        curr = info.output_shape;
        out.push_back(std::move(info));
    }
    return out;
}

// reset the parameters of all layers in a sequential container
void reset_sequential_parameters(const std::shared_ptr<torch::nn::Sequential>& seq) {
    if (!seq) return;
    seq->get()->apply([](torch::nn::Module& m) {
        // Linear
        if (auto* lin = dynamic_cast<torch::nn::LinearImpl*>(&m)) {
            lin->reset_parameters();
            return;
        }
        // MultiheadAttention 
        if (auto* mha = dynamic_cast<torch::nn::MultiheadAttentionImpl*>(&m)) {
            mha->_reset_parameters();
            return;
        }
        // SelfAttention
        if (auto* wrap = dynamic_cast<contorchionist::core::dp_mha::MHAWrapperImpl*>(&m)) {
            if (wrap->mha) {
                wrap->mha->_reset_parameters();
            }
            return;
        }
        // add here other layers with parameters
        // if (auto* bn = dynamic_cast<torch::nn::BatchNorm1dImpl*>(&m)) { bn->reset_parameters(); return; }
    });
}


// initialize tensor weights
static void init_tensor(torch::Tensor& w, InitMethod m, const InitOptions& o) {
    if (!w.defined()) return;
    using namespace torch::nn::init;
    switch (m) {
        case InitMethod::KaimingUniform: {
            torch::nn::init::NonlinearityType nl;
            if (o.nonlinearity == "leaky_relu") {
                nl = torch::kLeakyReLU;
            } else {
                nl = torch::kReLU;
            }
            kaiming_uniform_(w, /*a=*/o.a, torch::kFanIn, nl);
            break;
        }
        case InitMethod::KaimingNormal: {
            torch::nn::init::NonlinearityType nl;
            if (o.nonlinearity == "leaky_relu") {
                nl = torch::kLeakyReLU;
            } else {
                nl = torch::kReLU;
            }
            kaiming_normal_(w, /*a=*/o.a, torch::kFanIn, nl);
            break;
        }
        case InitMethod::XavierUniform:
            xavier_uniform_(w, o.gain);
            break;
        case InitMethod::XavierNormal:
            xavier_normal_(w, o.gain);
            break;
        case InitMethod::Uniform:
            uniform_(w, o.low, o.high);
            break;
        case InitMethod::Normal:
            normal_(w, o.mean, o.std);
            break;
        case InitMethod::Constant:
            constant_(w, o.constant);
            break;
    }
}

// initialize linear layer weights
static void init_linear(torch::nn::LinearImpl* lin, InitMethod m, const InitOptions& o) {
    if (!lin) return;
    auto w = lin->weight;
    init_tensor(w, m, o);
    if (lin->bias.defined() && o.zero_bias) {
        torch::nn::init::zeros_(lin->bias);
    }
}

// initialize multi-head attention weights
static void init_mha(torch::nn::MultiheadAttentionImpl* mha, InitMethod m, const InitOptions& o) {
    if (!mha) return;
    if (mha->in_proj_weight.defined()) init_tensor(mha->in_proj_weight, m, o);
    if (mha->in_proj_bias.defined() && o.zero_bias) torch::nn::init::zeros_(mha->in_proj_bias);
    // out_proj é Linear
    init_linear(mha->out_proj.get(), m, o);
}


// initialize sequential parameters
void initialize_sequential_weights(const std::shared_ptr<torch::nn::Sequential>& seq,
                                      InitMethod method,
                                      const InitOptions& opts) {
    if (!seq) return;
    seq->get()->apply([&](torch::nn::Module& m) {
        if (auto* lin = dynamic_cast<torch::nn::LinearImpl*>(&m)) {
            init_linear(lin, method, opts);
            return;
        }
        if (auto* mha = dynamic_cast<torch::nn::MultiheadAttentionImpl*>(&m)) {
            init_mha(mha, method, opts);
            return;
        }
        if (auto* wrap = dynamic_cast<contorchionist::core::dp_mha::MHAWrapperImpl*>(&m)) {
            if (wrap->mha) init_mha(wrap->mha.get(), method, opts);
            return;
        }
    });
}

// convert string to lowercase
static std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)std::tolower(c); });
    return s;
}

// parse initialization method from string
InitMethod parse_init_method(const std::string& method_str, bool* ok) {
    const std::string s = to_lower_copy(method_str);
    if (ok){
        *ok = true;
    }

    if (s == "kaiming_uniform" || s == "he_uniform"){
        return InitMethod::KaimingUniform;
    }
    if (s == "kaiming_normal"  || s == "he_normal"){  
        return InitMethod::KaimingNormal;
    }
    if (s == "xavier_uniform"  || s == "glorot_uniform"){ 
        return InitMethod::XavierUniform;
    }
    if (s == "xavier_normal"   || s == "glorot_normal") {
        return InitMethod::XavierNormal;
    }
    if (s == "uniform"){  
        return InitMethod::Uniform;
    }
    if (s == "normal"){   
        return InitMethod::Normal;
    }
    if (s == "constant") { 
        return InitMethod::Constant;
    }

    if (ok){
        *ok = false;
    }
    // fallback 
    return InitMethod::Uniform;
}

// convert enum to string
std::string init_method_to_string(InitMethod m) {
    switch (m) {
        case InitMethod::KaimingUniform: return "kaiming_uniform";
        case InitMethod::KaimingNormal:  return "kaiming_normal";
        case InitMethod::XavierUniform:  return "xavier_uniform";
        case InitMethod::XavierNormal:   return "xavier_normal";
        case InitMethod::Uniform:        return "uniform";
        case InitMethod::Normal:         return "normal";
        case InitMethod::Constant:       return "constant";
    }
    return "unknown";
}
 
// set random seed for reproductibility
void set_random_seed(int64_t seed, bool verbose) {
    if (seed >= 0) {
        // set seed for pytorch
        torch::manual_seed(seed);

        // set seed for CUDA if available
        if (torch::cuda::is_available()) {
            torch::cuda::manual_seed_all(seed);
        }
        
        #ifdef TORCH_BACKEND_MPS
        // set seed for MPS (Apple Silicon) if available
        if (torch::backends::mps::is_available()) {
            torch::mps::manual_seed(seed);
        }
        #endif
        
        if (verbose) {
            std::cout << "[contorchionist::core] Random seed set to " << seed 
                        << " for weight initialization" << std::endl;
        }
    } else if (verbose) {
        std::cout << "[contorchionist::core] Using random seed for weight initialization" 
                    << std::endl;
    }
}


        } // namespace dp_sequential
    } // namespace core
} // namespace contorchionist
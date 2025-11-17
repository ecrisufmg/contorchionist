#include "pd_global_state.h"
#include "pd_torch_utils.h"
#include "pd_torch_types.h"
#include <torch/script.h>
#include <cmath>       // For M_PI, cos
#include <algorithm>
#include "m_pd.h"      // For post() and pd_error()
#include "../../../core/include/core_dp_mha.h"



namespace PDGlobalState {


//*---------------------------- global variables -----------------------------*//    

//registry for storing module names and their corresponding torch::nn::Sequential containers
std::map<std::string, std::shared_ptr<torch::nn::Sequential>> module_registry;

//registry for storing which instances are associated with each module
std::map<std::string, std::vector<t_torch_mha*>> mha_registry; 
std::map<std::string, std::vector<t_torch_activation*>> activation_registry; 
std::map<std::string, std::vector<t_torch_linear*>> linear_registry;
std::map<std::string, std::vector<t_torch_reshape*>> reshape_registry;
std::map<std::string, std::vector<t_torch_conv*>> conv_registry;
std::map<std::string, t_torch_ls2tensor*> ls2tensor_registry;

// global pointer for the class of torch.sequential and torch.load to other objects can access it and send messages using pd_findbyclass
t_class *torch_sequential_class = nullptr;
t_class *torch_ls2tensor_class = nullptr;

// t_class *torch_load_class = nullptr;



//* ----------------------------------- global functions -------------------------------------- //

// ----------------- removes the module from a container (module) in the registry ----------------- //
void remove_layer(void* x, const std::string& layer_type, bool verbose) {
    if (layer_type == "mha") {
        t_torch_mha* mha = static_cast<t_torch_mha*>(x);

        // Check if the instance was added to a module
        if (!mha || !mha->added_to_module || !mha->added_to_module_name || !mha->added_layer_name) {
            if (verbose) {
                pd_error(mha, "torch.mha: This instance was not added to a module.");
            }
            return;
        }
        // Look for the module in the global registry
        auto it = PDGlobalState::module_registry.find(mha->added_to_module_name->s_name);
        if (it == PDGlobalState::module_registry.end()){
            pd_error(mha, "torch.mha: Module '%s' not found.", mha->added_to_module_name->s_name);
            return;
        }
        // Get the current container (module) from the registry
        std::shared_ptr<torch::nn::Sequential> old_container = it->second;
        if (!old_container){
            pd_error(mha, "torch.mha: Internal error: module '%s' is nullptr.", mha->added_to_module_name->s_name);
            return;
        }
        // create a new container to store the layers that will be kept
        auto new_container = std::make_shared<torch::nn::Sequential>();
        // Copy all the layers from the old container to the new container, except the MHA instance that is being removed
        for (size_t i = 0; i < old_container->get()->size(); ++i) {
            auto child = old_container->get()->ptr(i);
            auto mha_ptr = std::dynamic_pointer_cast<torch::nn::MultiheadAttentionImpl>(child);
            // remove the MHA instance from the container 
            if (auto mha_wrapper = std::dynamic_pointer_cast<contorchionist::core::dp_mha::MHAWrapperImpl>(child)) {
                if (mha_wrapper.get() == mha->mha_wrapper.get()) continue; // Remove só o wrapper correto
                new_container->get()->push_back(mha_wrapper);
            } else if (auto linear = std::dynamic_pointer_cast<torch::nn::LinearImpl>(child)) {
                new_container->get()->push_back(torch::nn::Linear(linear));
            } else if (auto relu = std::dynamic_pointer_cast<torch::nn::ReLUImpl>(child)) {
                new_container->get()->push_back(torch::nn::ReLU(relu));
            } else if (auto selu = std::dynamic_pointer_cast<torch::nn::SELUImpl>(child)) {
                new_container->get()->push_back(torch::nn::SELU(selu));
            } else if (auto silu = std::dynamic_pointer_cast<torch::nn::SiLUImpl>(child)) {
                new_container->get()->push_back(torch::nn::SiLU(silu));
            } else if (auto relu6 = std::dynamic_pointer_cast<torch::nn::ReLU6Impl>(child)) {
                new_container->get()->push_back(torch::nn::ReLU6(relu6));
            } else if (auto sigmoid = std::dynamic_pointer_cast<torch::nn::SigmoidImpl>(child)) {
                new_container->get()->push_back(torch::nn::Sigmoid(sigmoid));
            } else if (auto tanh = std::dynamic_pointer_cast<torch::nn::TanhImpl>(child)) {
                new_container->get()->push_back(torch::nn::Tanh(tanh));
            } else if (auto leakyrelu = std::dynamic_pointer_cast<torch::nn::LeakyReLUImpl>(child)) {
                new_container->get()->push_back(torch::nn::LeakyReLU(leakyrelu));
            } else if (auto elu = std::dynamic_pointer_cast<torch::nn::ELUImpl>(child)) {
                new_container->get()->push_back(torch::nn::ELU(elu));
            } else if (auto celu = std::dynamic_pointer_cast<torch::nn::CELUImpl>(child)) {
                new_container->get()->push_back(torch::nn::CELU(celu));
            } else if (auto softmax = std::dynamic_pointer_cast<torch::nn::SoftmaxImpl>(child)) {
                new_container->get()->push_back(torch::nn::Softmax(softmax));
            } else if (auto logsoftmax = std::dynamic_pointer_cast<torch::nn::LogSoftmaxImpl>(child)) {
                new_container->get()->push_back(torch::nn::LogSoftmax(logsoftmax));
            } else if (auto softshrink = std::dynamic_pointer_cast<torch::nn::SoftshrinkImpl>(child)) {
                new_container->get()->push_back(torch::nn::Softshrink(softshrink));
            } else if (auto hardshrink = std::dynamic_pointer_cast<torch::nn::HardshrinkImpl>(child)) {
                new_container->get()->push_back(torch::nn::Hardshrink(hardshrink));
            } else if (auto hardtanh = std::dynamic_pointer_cast<torch::nn::HardtanhImpl>(child)) {
                new_container->get()->push_back(torch::nn::Hardtanh(hardtanh));
            } else if (auto threshold = std::dynamic_pointer_cast<torch::nn::ThresholdImpl>(child)) {
                new_container->get()->push_back(torch::nn::Threshold(threshold));
            } else if (auto rrelu = std::dynamic_pointer_cast<torch::nn::RReLUImpl>(child)) {
                new_container->get()->push_back(torch::nn::RReLU(rrelu));
            } else if (auto gelu = std::dynamic_pointer_cast<torch::nn::GELUImpl>(child)) {
                new_container->get()->push_back(torch::nn::GELU(gelu));
            } else if (auto softplus = std::dynamic_pointer_cast<torch::nn::SoftplusImpl>(child)) {
                new_container->get()->push_back(torch::nn::Softplus(softplus));
            } else if (auto logsigmoid = std::dynamic_pointer_cast<torch::nn::LogSigmoidImpl>(child)) {
                new_container->get()->push_back(torch::nn::LogSigmoid(logsigmoid));
            } //add other types of layer here
        }
        // Erase the old container from the global registry
        PDGlobalState::module_registry[mha->added_to_module_name->s_name] = new_container;
        // Erase the MHA instance from the global registry of instances
        const char* module_name = mha->added_to_module_name ? mha->added_to_module_name->s_name : "(null)";
        auto &mha_list = PDGlobalState::mha_registry[module_name];
        mha_list.erase(std::remove(mha_list.begin(), mha_list.end(), mha), mha_list.end());
        // Clean the instance state
        mha->added_to_module = false;
        mha->added_layer_name = nullptr;
        mha->added_to_module_name = nullptr;
        mha->added_layer_index = (size_t)-1;
        if (verbose) {
            post("torch.mha: Removed layer from module '%s'", module_name);
        }


    // activation
    } else if (layer_type == "activation") {
        t_torch_activation* act = static_cast<t_torch_activation*>(x);
        // Check if the instance was added to a module
        if (!act || !act->added_to_module || !act->added_to_module_name || !act->added_layer_name) {
            if (verbose) {
                pd_error(act, "torch.activation: This instance was not added to a module.");
            }
            return;
        }
        // Look for the module in the global registry
        auto it = PDGlobalState::module_registry.find(act->added_to_module_name->s_name);
        if (it == PDGlobalState::module_registry.end()){
            pd_error(act, "torch.activation: Module '%s' not found.", act->added_to_module_name->s_name);
            return;
        }
        // Get the current container (module) from the registry
        std::shared_ptr<torch::nn::Sequential> old_container = it->second;
        if (!old_container){
            pd_error(act, "torch.activation: Internal error: module '%s' is nullptr.", act->added_to_module_name->s_name);
            return;
        }

        // create a new container to store the layers that will be kept
        auto new_container = std::make_shared<torch::nn::Sequential>();
        // Copy all the layers from the old container to the new container, except the activation instance that is being removed
        for (size_t i = 0; i < old_container->get()->size(); ++i) {
            if (i == act->added_layer_index) continue; // removes activation instance
            auto child = old_container->get()->ptr(i);
            if (auto mha_wrapper = std::dynamic_pointer_cast<contorchionist::core::dp_mha::MHAWrapperImpl>(child)) {
                new_container->get()->push_back(mha_wrapper);
            } else if (auto linear = std::dynamic_pointer_cast<torch::nn::LinearImpl>(child)) {
                new_container->get()->push_back(torch::nn::Linear(linear));
            } else if (auto relu = std::dynamic_pointer_cast<torch::nn::ReLUImpl>(child)) {
                new_container->get()->push_back(torch::nn::ReLU(relu));
            } else if (auto selu = std::dynamic_pointer_cast<torch::nn::SELUImpl>(child)) {
                new_container->get()->push_back(torch::nn::SELU(selu));
            } else if (auto silu = std::dynamic_pointer_cast<torch::nn::SiLUImpl>(child)) {
                new_container->get()->push_back(torch::nn::SiLU(silu));
            } else if (auto relu6 = std::dynamic_pointer_cast<torch::nn::ReLU6Impl>(child)) {
                new_container->get()->push_back(torch::nn::ReLU6(relu6));
            } else if (auto sigmoid = std::dynamic_pointer_cast<torch::nn::SigmoidImpl>(child)) {
                new_container->get()->push_back(torch::nn::Sigmoid(sigmoid));
            } else if (auto tanh = std::dynamic_pointer_cast<torch::nn::TanhImpl>(child)) {
                new_container->get()->push_back(torch::nn::Tanh(tanh));
            } else if (auto leakyrelu = std::dynamic_pointer_cast<torch::nn::LeakyReLUImpl>(child)) {
                new_container->get()->push_back(torch::nn::LeakyReLU(leakyrelu));
            } else if (auto elu = std::dynamic_pointer_cast<torch::nn::ELUImpl>(child)) {
                new_container->get()->push_back(torch::nn::ELU(elu));
            } else if (auto celu = std::dynamic_pointer_cast<torch::nn::CELUImpl>(child)) {
                new_container->get()->push_back(torch::nn::CELU(celu));
            } else if (auto softmax = std::dynamic_pointer_cast<torch::nn::SoftmaxImpl>(child)) {
                new_container->get()->push_back(torch::nn::Softmax(softmax));
            } else if (auto logsoftmax = std::dynamic_pointer_cast<torch::nn::LogSoftmaxImpl>(child)) {
                new_container->get()->push_back(torch::nn::LogSoftmax(logsoftmax));
            } else if (auto softshrink = std::dynamic_pointer_cast<torch::nn::SoftshrinkImpl>(child)) {
                new_container->get()->push_back(torch::nn::Softshrink(softshrink));
            } else if (auto hardshrink = std::dynamic_pointer_cast<torch::nn::HardshrinkImpl>(child)) {
                new_container->get()->push_back(torch::nn::Hardshrink(hardshrink));
            } else if (auto hardtanh = std::dynamic_pointer_cast<torch::nn::HardtanhImpl>(child)) {
                new_container->get()->push_back(torch::nn::Hardtanh(hardtanh));
            } else if (auto threshold = std::dynamic_pointer_cast<torch::nn::ThresholdImpl>(child)) {
                new_container->get()->push_back(torch::nn::Threshold(threshold));
            } else if (auto rrelu = std::dynamic_pointer_cast<torch::nn::RReLUImpl>(child)) {
                new_container->get()->push_back(torch::nn::RReLU(rrelu));
            } else if (auto gelu = std::dynamic_pointer_cast<torch::nn::GELUImpl>(child)) {
                new_container->get()->push_back(torch::nn::GELU(gelu));
            } else if (auto softplus = std::dynamic_pointer_cast<torch::nn::SoftplusImpl>(child)) {
                new_container->get()->push_back(torch::nn::Softplus(softplus));
            } else if (auto logsigmoid = std::dynamic_pointer_cast<torch::nn::LogSigmoidImpl>(child)) {
                new_container->get()->push_back(torch::nn::LogSigmoid(logsigmoid));
            }
            // add other types of layers here
        }
        // Erase the old container from the global registry
        PDGlobalState::module_registry[act->added_to_module_name->s_name] = new_container;
        const char* module_name = act->added_to_module_name ? act->added_to_module_name->s_name : "(null)";
        auto &act_list = PDGlobalState::activation_registry[module_name];
        act_list.erase(std::remove(act_list.begin(), act_list.end(), act), act_list.end());
        // Clean the instance state
        act->added_to_module = false;
        act->added_layer_name = nullptr;
        act->added_to_module_name = nullptr;
        act->added_layer_index = (size_t)-1;
        if (verbose) {
            post("torch.activation: Removed layer from module '%s'", module_name);
        }

    // linear
    } else if (layer_type == "linear") {
        t_torch_linear* linear = static_cast<t_torch_linear*>(x);
        // Verifica se a instância foi adicionada a um módulo
        if (!linear || !linear->added_to_module || !linear->added_to_module_name || !linear->added_layer_name) {
            if (verbose) {
                pd_error(linear, "torch.linear: This instance was not added to a module.");
            }
            return;
        }
        // Procura o módulo no registry global
        auto it = PDGlobalState::module_registry.find(linear->added_to_module_name->s_name);
        if (it == PDGlobalState::module_registry.end()) {
            pd_error(linear, "torch.linear: Module '%s' not found.", linear->added_to_module_name->s_name);
            return;
        }
        // get the current container (module) from the registry
        std::shared_ptr<torch::nn::Sequential> old_container = it->second;
        if (!old_container) {
            pd_error(linear, "torch.linear: Internal error: module '%s' is nullptr.", linear->added_to_module_name->s_name);
            return;
        }
        // create a new container to store the layers that will be kept
        auto new_container = std::make_shared<torch::nn::Sequential>();
        // Copy all the layers from the old container to the new container, except the linear instance that is being removed
        for (size_t i = 0; i < old_container->get()->size(); ++i) {
            if (i == linear->added_layer_index) continue; // remove a camada linear
            auto child = old_container->get()->ptr(i);
            if (auto mha_wrapper = std::dynamic_pointer_cast<contorchionist::core::dp_mha::MHAWrapperImpl>(child)) {
                new_container->get()->push_back(mha_wrapper);
            } else if (auto linear_impl = std::dynamic_pointer_cast<torch::nn::LinearImpl>(child)) {
                new_container->get()->push_back(torch::nn::Linear(linear_impl));
            } else if (auto relu = std::dynamic_pointer_cast<torch::nn::ReLUImpl>(child)) {
                new_container->get()->push_back(torch::nn::ReLU(relu));
            } else if (auto selu = std::dynamic_pointer_cast<torch::nn::SELUImpl>(child)) {
                new_container->get()->push_back(torch::nn::SELU(selu));
            } else if (auto silu = std::dynamic_pointer_cast<torch::nn::SiLUImpl>(child)) {
                new_container->get()->push_back(torch::nn::SiLU(silu));
            } else if (auto relu6 = std::dynamic_pointer_cast<torch::nn::ReLU6Impl>(child)) {
                new_container->get()->push_back(torch::nn::ReLU6(relu6));
            } else if (auto sigmoid = std::dynamic_pointer_cast<torch::nn::SigmoidImpl>(child)) {
                new_container->get()->push_back(torch::nn::Sigmoid(sigmoid));
            } else if (auto tanh = std::dynamic_pointer_cast<torch::nn::TanhImpl>(child)) {
                new_container->get()->push_back(torch::nn::Tanh(tanh));
            } else if (auto leakyrelu = std::dynamic_pointer_cast<torch::nn::LeakyReLUImpl>(child)) {
                new_container->get()->push_back(torch::nn::LeakyReLU(leakyrelu));
            } else if (auto elu = std::dynamic_pointer_cast<torch::nn::ELUImpl>(child)) {
                new_container->get()->push_back(torch::nn::ELU(elu));
            } else if (auto celu = std::dynamic_pointer_cast<torch::nn::CELUImpl>(child)) {
                new_container->get()->push_back(torch::nn::CELU(celu));
            } else if (auto softmax = std::dynamic_pointer_cast<torch::nn::SoftmaxImpl>(child)) {
                new_container->get()->push_back(torch::nn::Softmax(softmax));
            } else if (auto logsoftmax = std::dynamic_pointer_cast<torch::nn::LogSoftmaxImpl>(child)) {
                new_container->get()->push_back(torch::nn::LogSoftmax(logsoftmax));
            } else if (auto softshrink = std::dynamic_pointer_cast<torch::nn::SoftshrinkImpl>(child)) {
                new_container->get()->push_back(torch::nn::Softshrink(softshrink));
            } else if (auto hardshrink = std::dynamic_pointer_cast<torch::nn::HardshrinkImpl>(child)) {
                new_container->get()->push_back(torch::nn::Hardshrink(hardshrink));
            } else if (auto hardtanh = std::dynamic_pointer_cast<torch::nn::HardtanhImpl>(child)) {
                new_container->get()->push_back(torch::nn::Hardtanh(hardtanh));
            } else if (auto threshold = std::dynamic_pointer_cast<torch::nn::ThresholdImpl>(child)) {
                new_container->get()->push_back(torch::nn::Threshold(threshold));
            } else if (auto rrelu = std::dynamic_pointer_cast<torch::nn::RReLUImpl>(child)) {
                new_container->get()->push_back(torch::nn::RReLU(rrelu));
            } else if (auto gelu = std::dynamic_pointer_cast<torch::nn::GELUImpl>(child)) {
                new_container->get()->push_back(torch::nn::GELU(gelu));
            } else if (auto softplus = std::dynamic_pointer_cast<torch::nn::SoftplusImpl>(child)) {
                new_container->get()->push_back(torch::nn::Softplus(softplus));
            } else if (auto logsigmoid = std::dynamic_pointer_cast<torch::nn::LogSigmoidImpl>(child)) {
                new_container->get()->push_back(torch::nn::LogSigmoid(logsigmoid));
            }
            // add other types of layers here
        }
        // update the global registry with the new container
        PDGlobalState::module_registry[linear->added_to_module_name->s_name] = new_container;
        // remove the linear instance from the global registry of instances
        const char* module_name = linear->added_to_module_name ? linear->added_to_module_name->s_name : "(null)";
        auto &linear_list = PDGlobalState::linear_registry[module_name];
        linear_list.erase(std::remove(linear_list.begin(), linear_list.end(), linear), linear_list.end());
        // clenan the instance state
        linear->added_to_module = false;
        linear->added_layer_name = nullptr;
        linear->added_to_module_name = nullptr;
        linear->added_layer_index = (size_t)-1;
        if (verbose) {
            post("torch.linear: Removed layer from module '%s'", module_name);
        }

    // reshape
    } else if (layer_type == "reshape") {
        t_torch_reshape* reshape = static_cast<t_torch_reshape*>(x);
        //verify if the instance was added to a module
        if (!reshape || !reshape->added_to_module || !reshape->added_to_module_name || !reshape->added_layer_name) {
            if (verbose) {
                pd_error(reshape, "torch.reshape: This instance was not added to a module.");
            }
            return;
        }
        // look for the module in the global registry
        auto it = PDGlobalState::module_registry.find(reshape->added_to_module_name->s_name);
        if (it == PDGlobalState::module_registry.end()) {
            pd_error(reshape, "torch.reshape: Module '%s' not found.", reshape->added_to_module_name->s_name);
            return;
        }
        // Obtém o container atual do registry
        std::shared_ptr<torch::nn::Sequential> old_container = it->second;
        if (!old_container) {
            pd_error(reshape, "torch.reshape: Internal error: module '%s' is nullptr.", reshape->added_to_module_name->s_name);
            return;
        }
        // create a new container to store the layers that will be kept
        auto new_container = std::make_shared<torch::nn::Sequential>();
        // copy all the layers from the old container to the new container, except the reshape instance that is being removed
        for (size_t i = 0; i < old_container->get()->size(); ++i) {
            if (i == reshape->added_layer_index) continue; // remove a camada reshape
            auto child = old_container->get()->ptr(i);
            // remove the reshape instance from the container
            if (auto mha_wrapper = std::dynamic_pointer_cast<contorchionist::core::dp_mha::MHAWrapperImpl>(child)) {
                new_container->get()->push_back(mha_wrapper);
            } else if (auto linear_impl = std::dynamic_pointer_cast<torch::nn::LinearImpl>(child)) {
                new_container->get()->push_back(torch::nn::Linear(linear_impl));
            } else if (auto relu = std::dynamic_pointer_cast<torch::nn::ReLUImpl>(child)) {
                new_container->get()->push_back(torch::nn::ReLU(relu));
            } else if (auto selu = std::dynamic_pointer_cast<torch::nn::SELUImpl>(child)) {
                new_container->get()->push_back(torch::nn::SELU(selu));
            } else if (auto silu = std::dynamic_pointer_cast<torch::nn::SiLUImpl>(child)) {
                new_container->get()->push_back(torch::nn::SiLU(silu));
            } else if (auto relu6 = std::dynamic_pointer_cast<torch::nn::ReLU6Impl>(child)) {
                new_container->get()->push_back(torch::nn::ReLU6(relu6));
            } else if (auto sigmoid = std::dynamic_pointer_cast<torch::nn::SigmoidImpl>(child)) {
                new_container->get()->push_back(torch::nn::Sigmoid(sigmoid));
            } else if (auto tanh = std::dynamic_pointer_cast<torch::nn::TanhImpl>(child)) {
                new_container->get()->push_back(torch::nn::Tanh(tanh));
            } else if (auto leakyrelu = std::dynamic_pointer_cast<torch::nn::LeakyReLUImpl>(child)) {
                new_container->get()->push_back(torch::nn::LeakyReLU(leakyrelu));
            } else if (auto elu = std::dynamic_pointer_cast<torch::nn::ELUImpl>(child)) {
                new_container->get()->push_back(torch::nn::ELU(elu));
            } else if (auto celu = std::dynamic_pointer_cast<torch::nn::CELUImpl>(child)) {
                new_container->get()->push_back(torch::nn::CELU(celu));
            } else if (auto softmax = std::dynamic_pointer_cast<torch::nn::SoftmaxImpl>(child)) {
                new_container->get()->push_back(torch::nn::Softmax(softmax));
            } else if (auto logsoftmax = std::dynamic_pointer_cast<torch::nn::LogSoftmaxImpl>(child)) {
                new_container->get()->push_back(torch::nn::LogSoftmax(logsoftmax));
            } else if (auto softshrink = std::dynamic_pointer_cast<torch::nn::SoftshrinkImpl>(child)) {
                new_container->get()->push_back(torch::nn::Softshrink(softshrink));
            } else if (auto hardshrink = std::dynamic_pointer_cast<torch::nn::HardshrinkImpl>(child)) {
                new_container->get()->push_back(torch::nn::Hardshrink(hardshrink));
            } else if (auto hardtanh = std::dynamic_pointer_cast<torch::nn::HardtanhImpl>(child)) {
                new_container->get()->push_back(torch::nn::Hardtanh(hardtanh));
            } else if (auto threshold = std::dynamic_pointer_cast<torch::nn::ThresholdImpl>(child)) {
                new_container->get()->push_back(torch::nn::Threshold(threshold));
            } else if (auto rrelu = std::dynamic_pointer_cast<torch::nn::RReLUImpl>(child)) {
                new_container->get()->push_back(torch::nn::RReLU(rrelu));
            } else if (auto gelu = std::dynamic_pointer_cast<torch::nn::GELUImpl>(child)) {
                new_container->get()->push_back(torch::nn::GELU(gelu));
            } else if (auto softplus = std::dynamic_pointer_cast<torch::nn::SoftplusImpl>(child)) {
                new_container->get()->push_back(torch::nn::Softplus(softplus));
            } else if (auto logsigmoid = std::dynamic_pointer_cast<torch::nn::LogSigmoidImpl>(child)) {
                new_container->get()->push_back(torch::nn::LogSigmoid(logsigmoid));
            } else if (auto functional = std::dynamic_pointer_cast<torch::nn::FunctionalImpl>(child)) {
                new_container->get()->push_back(torch::nn::Functional(functional));
            }
            // add other types of layers here
        }
        // update the global registry with the new container
        PDGlobalState::module_registry[reshape->added_to_module_name->s_name] = new_container;
        // Remove a instância do registry de reshape
        const char* module_name = reshape->added_to_module_name ? reshape->added_to_module_name->s_name : "(null)";
        auto &reshape_list = PDGlobalState::reshape_registry[module_name];
        reshape_list.erase(std::remove(reshape_list.begin(), reshape_list.end(), reshape), reshape_list.end());
        // clean the instance state
        reshape->added_to_module = false;
        reshape->added_layer_name = nullptr;
        reshape->added_to_module_name = nullptr;
        reshape->added_layer_index = (size_t)-1;
        if (verbose) {
            post("torch.reshape: Removed layer from module '%s'", module_name);
        }
    } else {
        pd_error(nullptr, "torch: Unknown layer type '%s' for removal.", layer_type.c_str());
    }
}

} // end namespace PDGlobalState
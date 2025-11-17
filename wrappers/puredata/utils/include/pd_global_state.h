#include "m_pd.h"
#include <torch/torch.h>
#include <string>
#include <vector> // Include if needed for other utils later
#include <map>
#include "pd_torch_types.h"
#pragma once // Avoid multiple inclusions


namespace PDGlobalState {

//* ------------------------ global variables ------------------------ *//

// global registry for storing sequential module names and their corresponding torch::nn::Sequential containers
extern std::map<std::string, std::shared_ptr<torch::nn::Sequential>> module_registry;

// global registry for storing which objects/layers instances are associated with each module
extern std::map<std::string, std::vector<t_torch_mha*>> mha_registry; 
extern std::map<std::string, std::vector<t_torch_activation*>> activation_registry;
extern std::map<std::string, std::vector<t_torch_linear*>> linear_registry;
extern std::map<std::string, std::vector<t_torch_reshape*>> reshape_registry;
extern std::map<std::string, std::vector<t_torch_conv*>> conv_registry;
extern std::map<std::string, t_torch_ls2tensor*> ls2tensor_registry;


// global pointer to the class of objects used for other objects access instances and send messages to it using pd_findbyclass
extern t_class *torch_sequential_class;
extern t_class *torch_ls2tensor_class;
// extern t_class *torch_load_class;

//* ----------------------------------- global functions -------------------------------------- //

// Function to remove a layer from a container (module) in the registry
void remove_layer(void* x, const std::string& layer_type, bool verbose);

//template to add a layer to a torch.sequential module (template function must be implemented in the header file)
template <typename ModuleType, typename RegistryType, typename ObjectType> 
bool add_layer_to_module(
    ObjectType* x, 
    const std::string& layer_type, 
    ModuleType&& module, 
    t_symbol* module_name, 
    RegistryType& registry, 
    size_t& out_layer_index, 
    t_symbol*& out_layer_name, 
    bool& out_added_to_module, 
    t_symbol*& out_added_to_module_name) {

    // Check if the layer has already been added to a module
    if (out_added_to_module) {
        pd_error(x, "torch.%s: This instance has already been added to a module.", layer_type.c_str());
        return false;
    }
    // look up the module name in the registry
    auto it = PDGlobalState::module_registry.find(module_name->s_name);
    if (it == PDGlobalState::module_registry.end()) {
        pd_error(x, "torch.%s: Module '%s' not found.", layer_type.c_str(), module_name->s_name);
        return false;
    }
    // check if the module is a sequential container
    std::shared_ptr<torch::nn::Sequential> container = it->second;
    if (!container) {
        pd_error(x, "torch.%s: Internal error: module '%s' is nullptr.", layer_type.c_str(), module_name->s_name);
        return false;
    }
    // check if the layer type is valid
    if (layer_type == "activation" && container->get()->size() == 0) {
        pd_error(x, "torch.activation: Activation cannot be the first layer in the module '%s'.", module_name->s_name);
        return false;
    }
    //store the layer name and index
    out_layer_index = container->get()->size();
    // create a unique name for the layer
    std::string unique_name = layer_type + "_" + std::to_string(reinterpret_cast<uintptr_t>(x));
    //add the layer to the torch.sequential module
    container->get()->push_back(std::forward<ModuleType>(module));
    out_layer_name = gensym(unique_name.c_str());
    out_added_to_module = true;
    out_added_to_module_name = module_name;
    registry[module_name->s_name].push_back(x);

    post("torch.%s: Added to module '%s' with layer name '%s' as layer %zu",
         layer_type.c_str(), module_name->s_name, unique_name.c_str(), container->get()->size());
    return true;
}

// Template function to notify all instances removed from a module when the module is cleaned (template function must be implemented in the header file)
template <typename T>
void notify_and_clear_registry(std::map<std::string, std::vector<T*>> &registry, const std::string& module_name) {
    auto &layer_list = registry[module_name];
    for (auto *layer : layer_list) {
        layer->added_to_module = false;
        layer->added_to_module_name = nullptr;
        layer->added_layer_name = nullptr;
    }
    layer_list.clear();
}




} // namespace PDGlobalState


#include "m_pd.h"
#include <torch/torch.h>
#include <string>
#include <map>
#include <vector>
#include <functional>
#include "../utils/include/pd_torch_types.h" // Include the object struct definition
#include "../utils/include/pd_torch_utils.h" // Utility functions
#include "../utils/include/pd_global_state.h" // manages global state for pd modules and layers
#include "../../../core/include/core_dp_activation.h"


/*
TODO:
- suporte a in-place
- suporte a batch
- suporte a argmax, argmin, etc. (?)
*/


static t_class *torch_activation_class;


//* ------------------------- set alpha for leaky relu  ------------------------- *//
static void alpha_value (t_torch_activation *x, t_floatarg al){
    if(al > 0 && al < 10){
        x->alpha = al;
        post("alpha: %0.4f", x->alpha);
    } else {
        pd_error(x, "torch.activation: Alpha must be positive value between 0 and 10.");
    }
}

//* -------------------------- set lambda for leaky relu -------------------------- *//
static void lambda_value(t_torch_activation *x, t_floatarg lambda) {
    // check if lambda is positive
    if (lambda >= 0) {
        x->lambda = lambda; // update lambda
        post("torch.activation: Lambda: %0.4f", x->lambda); 
    } else {
        pd_error(x, "torch.activation: Lambda must be positive."); 
    }
}

//* --------------------- set dim value for softmax, logsoftmax and softmin --------------------- *//
static void torch_activation_set_dim(t_torch_activation *x, t_floatarg dim) {
    if (dim < -1) {
        pd_error(x, "torch.activation: Dimension (dim) must be >= -1.");
        return;
    }
    // check if the shape is set
    if (x->shape.empty()) {
        pd_error(x, "torch.activation: Shape must be set before setting dim.");
        return;
    }
    // check if the dimension is valid
    int64_t actual_dim = static_cast<int64_t>(dim);
    if (actual_dim >= 0 && actual_dim >= static_cast<int64_t>(x->shape.size())) {
        pd_error(x, "torch.activation: Dimension (dim) must be less than the number of dimensions in the shape (%zu).", x->shape.size());
        return;
    }
    x->dim = actual_dim;
    post("torch.activation: Dimension (dim) set to %lld", x->dim);
}


//* ----------------------- sets the device (cpu or cuda) --------------------- *//
static void torch_activation_device(t_torch_activation *x, t_symbol *s, int argc, t_atom *argv) {
    // Check if the first argument is a symbol
    if (argc != 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.activation: Please provide a device (cpu or cuda).");
        return;
    }
    // get the device name received
    t_symbol *device_name = atom_getsymbol(&argv[0]);
    std::string dev = device_name->s_name;

    // set the device
    x->device = PdTorchUtils::select_device(dev, (t_pd*)x);
    // post("torch.activation: Device set to %s", dev.c_str());
}



//* ------------------------ adds the activation layer to a specific container (module) created by pdtoch.sequential ------------------------ *//
static void torch_add_to_module(t_torch_activation *x, t_symbol *s, int argc, t_atom *argv) {
    // check if the module is already added
    if (x->added_to_module) {
        pd_error(x, "torch.activation: This instance has already been added to a module.");
        return;
    }
    // check if a module name was provided
    if (argc != 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.activation: Please provide a module name to add to.");
        return;
    }
    // get the module name provided
    t_symbol *module_name = atom_getsymbol(argv);
    
    // adds the activation layer chosen by the user
    std::string act = x->activation ? x->activation->s_name : "";
    bool added = false;
    if (act == "relu") {
        added = PDGlobalState::add_layer_to_module(
            x, "activation", torch::nn::ReLU(), module_name,
            PDGlobalState::activation_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (act == "selu") {
        added = PDGlobalState::add_layer_to_module(
            x, "activation", torch::nn::SELU(), module_name,
            PDGlobalState::activation_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (act == "silu") {
        added = PDGlobalState::add_layer_to_module(
            x, "activation", torch::nn::SiLU(), module_name,
            PDGlobalState::activation_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (act == "relu6") {
        added = PDGlobalState::add_layer_to_module(
            x, "activation", torch::nn::ReLU6(), module_name,
            PDGlobalState::activation_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (act == "sigmoid") {
        added = PDGlobalState::add_layer_to_module(
            x, "activation", torch::nn::Sigmoid(), module_name,
            PDGlobalState::activation_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (act == "tanh") {
        added = PDGlobalState::add_layer_to_module(
            x, "activation", torch::nn::Tanh(), module_name,
            PDGlobalState::activation_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (act == "leakyrelu") {
        added = PDGlobalState::add_layer_to_module(
            x, "activation", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(x->alpha)), module_name,
            PDGlobalState::activation_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (act == "softshrink") {
        added = PDGlobalState::add_layer_to_module(
            x, "activation", torch::nn::Softshrink(torch::nn::SoftshrinkOptions().lambda(x->lambda)), module_name,
            PDGlobalState::activation_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (act == "hardshrink") {
        added = PDGlobalState::add_layer_to_module(
            x, "activation", torch::nn::Hardshrink(torch::nn::HardshrinkOptions().lambda(x->lambda)), module_name,
            PDGlobalState::activation_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (act == "hardtanh") {
        added = PDGlobalState::add_layer_to_module(
            x, "activation", torch::nn::Hardtanh(), module_name,
            PDGlobalState::activation_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (act == "threshold") {
        added = PDGlobalState::add_layer_to_module(
            x, "activation", torch::nn::Threshold(0.0, 0.0), module_name,
            PDGlobalState::activation_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (act == "rrelu") {
        added = PDGlobalState::add_layer_to_module(
            x, "activation", torch::nn::RReLU(torch::nn::RReLUOptions().lower(0.24).upper(0.42)), module_name,
            PDGlobalState::activation_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (act == "elu") {
        added = PDGlobalState::add_layer_to_module(
            x, "activation", torch::nn::ELU(torch::nn::ELUOptions().alpha(x->alpha)), module_name,
            PDGlobalState::activation_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (act == "celu") {
        added = PDGlobalState::add_layer_to_module(
            x, "activation", torch::nn::CELU(torch::nn::CELUOptions().alpha(x->alpha)), module_name,
            PDGlobalState::activation_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (act == "softplus") {
        added = PDGlobalState::add_layer_to_module(
            x, "activation", torch::nn::Softplus(), module_name,
            PDGlobalState::activation_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (act == "gelu") {
        added = PDGlobalState::add_layer_to_module(
            x, "activation", torch::nn::GELU(), module_name,
            PDGlobalState::activation_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (act == "logsigmoid") {
        added = PDGlobalState::add_layer_to_module(
            x, "activation", torch::nn::LogSigmoid(), module_name,
            PDGlobalState::activation_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (act == "softmax") {
        added = PDGlobalState::add_layer_to_module(
            x, "activation", torch::nn::Softmax(torch::nn::SoftmaxOptions(x->dim)), module_name,
            PDGlobalState::activation_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (act == "logsoftmax") {
        added = PDGlobalState::add_layer_to_module(
            x, "activation", torch::nn::LogSoftmax(torch::nn::LogSoftmaxOptions(x->dim)), module_name,
            PDGlobalState::activation_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else {
        pd_error(x, "torch.activation: Activation '%s' not supported for module addition.", act.c_str());
        return;
    }
    if (!added) return;
}

//* ------------------------ removes the activation layer from a specific container (module) created by pdtoch.sequential ------------------------ *//
static void torch_activation_remove (t_torch_activation *x, t_symbol *s, int argc, t_atom *argv){
    if (x->added_to_module) {
        PDGlobalState::remove_layer(x, "activation", x->verbose);
    } else {
        pd_error(x, "torch.activation: This instance was not added to a module.");
    }
}

//* ----------------------------- receives an activation function name ----------------------------- *//
static void activation_functions(t_torch_activation *x, t_symbol *s, int argc, t_atom *argv) {
    // Check if the first argument is a symbol
    if (argc != 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.activation: Please provide an activation name, e.g., 'activation relu'.");
        return;
    }
    // set the activation function to activation_name
    t_symbol *activation_name = atom_getsymbol(&argv[0]);
    std::string act = activation_name->s_name;

    // Check if the activation function is valid or is leaky_relu/elu/celu and set alpha
    auto func = contorchionist::core::dp_activation::get_activation_function(act);
    bool is_valid = (func != nullptr);

    // Check if the activation function is valid
    if (!is_valid) {
        pd_error(x, "torch.activation: Invalid activation function: %s", act.c_str());
        return;
    }

    // Set the activation function
    x->activation = activation_name;
    post("Activation function set to: %s", activation_name->s_name);
}


//* -------------------------- receives a shape for the tensor  -------------------------- *//
static void torch_activation_shape(t_torch_activation *x, t_symbol *s, int argc, t_atom *argv) {
    std::vector<int64_t> new_shape;

    // Check if the shape is valid
    for (int i = 0; i < argc; ++i) {
        if (argv[i].a_type != A_FLOAT) {
            pd_error(x, "torch.activation: Shape must be a list of float values representing dimensions.");
            return;
        }
        // Check if the value is a positive integer
        float val = atom_getfloat(argv + i);
        if (val <= 0 || std::floor(val) != val) {
            pd_error(x, "torch.activation: Shape dimensions must be positive integers.");
            return;
        }
        // Convert float to int64_t and add to new_shape (conversion is needed for libtorch compatibility)
        new_shape.push_back(static_cast<int64_t>(val));
    }

    // Check if the shape is empty
    if (new_shape.empty()) {
        pd_error(x, "torch.activation: Shape cannot be empty.");
        return;
    }
    // set the new shape
    x->shape = new_shape;

    x->dim = static_cast<int64_t>(x->shape.size()) - 1;

    // Print the new shape
    std::string shape_str = "[";
    for (size_t i = 0; i < x->shape.size(); ++i) {
        shape_str += std::to_string(x->shape[i]);
        if (i != x->shape.size() - 1) shape_str += ", ";
    }
    shape_str += "]";
    post("torch.activation: Shape set to: %s", shape_str.c_str());
}


//* ----------------------------- list all available activation functions ----------------------------- *//
static void list_activation_functions(t_torch_activation *x) {
    post("torch.activation: Available activation functions:");
    
    // list all activation functions from the map
    auto activations = contorchionist::core::dp_activation::list_available_activations();
    for (const auto& name : activations) {
        post("- %s", name.c_str());
    }
}


//* ------------------------------- receives a flattened tensor, reshape it, apply the activation function, flatten the result and send it to the outlet ------------------ *//
static void torch_activation_process(t_torch_activation *x, t_symbol *s, int argc, t_atom *argv) {
    // check if the activation function is set
    if (x->activation == nullptr) {
        pd_error(x, "torch.activation: No activation function set.");
        return;
    }

    // check if the shape is set
    if (x->shape.empty()) {
        pd_error(x, "torch.activation: Tensor shape not set. Use 'shape' method to define it.");
        return;
    }

    // check if the input is a list of floats
    std::vector<float> values;
    for (int i = 0; i < argc; ++i) {
        if (argv[i].a_type != A_FLOAT) {
            pd_error(x, "torch.activation: Only float values are accepted.");
            return;
        }
        values.push_back(atom_getfloat(argv + i));
    }

    // check if the input size matches the expected shape size
    int64_t expected_size = 1;
    for (auto d : x->shape) {
        expected_size *= d;
    }

    if (expected_size != static_cast<int64_t>(values.size())) {
        pd_error(x, "torch.activation: Input size (%d) does not match expected shape size (%lld).", (int)values.size(), expected_size);
        return;
    }

    //creates a tensor with the shape received
    at::Tensor tensor = torch::from_blob(values.data(), x->shape, at::kFloat).clone().to(x->device);

    // apply the activation function
    at::Tensor activated;

    // ActivationProcessor 
    try {
        activated = contorchionist::core::dp_activation::ActivationProcessor(
            tensor,                    // input tensor
            x->activation->s_name,     // activation name
            x->device,                 // target device
            x->alpha,                  // alpha parameter
            x->lambda,                 // lambda parameter
            x->dim                     // dim parameter
        );
    } catch (const std::runtime_error& e) {
        pd_error(x, "torch.activation: %s", e.what());
        return;
    } catch (const c10::Error& e) {
        pd_error(x, "torch.activation: PyTorch error: %s", e.what());
        return;
    }

    // move the tensor to CPU if it is on CUDA or MPS
    if (activated.device().is_cuda() || activated.device().is_mps()) {
        activated = activated.to(torch::kCPU);
    }

    // flatten the output tensor and convert to std::vector<float>
    activated = activated.flatten();
    int64_t out_size = activated.size(0);
    t_atom *out_atoms = (t_atom *)getbytes(sizeof(t_atom) * out_size);

    // convert the tensor to a list of atoms
    for (int64_t i = 0; i < out_size; ++i) {
        SETFLOAT(&out_atoms[i], activated[i].item<float>());
    }

    // send the output to the outlet
    outlet_list(x->x_obj.ob_outlet, &s_list, out_size, out_atoms);
    freebytes(out_atoms, sizeof(t_atom) * out_size);
}


//* -------------------------- Constructor -------------------------- *//
void *torch_activation_new(t_symbol *s, int argc, t_atom *argv) {
    t_torch_activation *x = (t_torch_activation *)pd_new(torch_activation_class);

    // post("torch.activation: libtorch version: %s", TORCH_VERSION);

    // ----- set the default parameters ------//
    //alpha default value
    x->alpha = 0.01;
    x->dim = - 1; //default value for softmax, logsoftmax and softmin
    x->lambda = 0.5; //default value for softshrink and hardshrink 
    size_t added_layer_index = -1; // store the index of the added layer 
    post("torch.activation: alpha = %.4f, lambda = %.4f, dim = %lld", x->alpha, x->lambda, x->dim);

    //shape default value
    std::vector<int64_t> default_shape; //default shape declared
    default_shape.push_back(static_cast<int64_t>(1)); //convert float to int64_t due to libtorch compatibility
    x->shape = default_shape; //set default shape to 1
    
    //default device
    x->device = torch::kCPU; // Default to CPU
    std::string requested_device_str = "cpu"; // Default string representation

    //auxiliary variables
    std::string device_str = "cpu";
    int shape_start_index = 1;
    int shape_end_index = argc;

    // ----- parse the creation argument: activation function, shape and device -----//
    // 1. Activation function
    if (argc >= 1 && argv[0].a_type == A_SYMBOL) {
        std::string act = argv[0].a_w.w_symbol->s_name;

        auto func = contorchionist::core::dp_activation::get_activation_function(act);
        if (func != nullptr) {
            x->activation = gensym(act.c_str());
            post("torch.activation: Activation set to %s", act.c_str());
        } else {
        // If invalid activation, use default and show warning
        x->activation = gensym("sigmoid"); //default activation function
        post("torch.activation: Invalid activation '%s'. Defaulting to sigmoid.", act.c_str());
        }
    } else {
        x->activation = gensym("sigmoid"); // default activation function
        post("torch.activation: No activation specified via creation argument. Defaulting to sigmoid");
    }
    

    // 2. Check if last argument is device
    if (argc >= 2 && argv[argc - 1].a_type == A_SYMBOL) {
        std::string arg = argv[argc - 1].a_w.w_symbol->s_name;
        if (arg == "cpu" || (arg == "cuda" && torch::cuda::is_available()) || (arg == "mps" && torch::mps::is_available())) {
            device_str = arg;
            shape_end_index = argc - 1; // exclude device from shape
        }
    }
    //set device
    x->device = PdTorchUtils::select_device(device_str, (t_pd*)x);

    // 3. Shape
    std::vector<int64_t> shape;
    for (int i = shape_start_index; i < shape_end_index; ++i) {
        if (argv[i].a_type == A_FLOAT) {
            float val = atom_getfloat(argv + i);
            if (val <= 0) {
                pd_error(x, "torch.activation: Shape values must be positive.");
                return nullptr;
            }
            shape.push_back(static_cast<int64_t>(val));
        } else {
            pd_error(x, "torch.activation: Invalid shape argument at position %d. Must be float.", i);
            return nullptr;
        }
    }
    // Check if the shape is empty
    if (!shape.empty()) {
        x->shape = shape;
        x->dim = static_cast<int64_t>(x->shape.size()) - 1;
        std::string shape_str = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            shape_str += std::to_string(shape[i]);
            if (i < shape.size() - 1) shape_str += ", ";
        }
        shape_str += "]";
        post("torch.activation: Shape set to %s", shape_str.c_str());
    }

    // Create outlet and output buffer
    x->x_out1 = outlet_new(&x->x_obj, &s_list);
    return (void *)x;
    
}

//* -------------------------- Destructor -------------------------- *//
void torch_activation_destroy(t_torch_activation *x) {

    if (x->added_to_module) {
        PDGlobalState::remove_layer(x, "activation", x->verbose);
    }
    
    // freebytes(x->out, 1 * sizeof(t_atom)); //free output buffer
    outlet_free(x->x_out1); //free outlet
}


//* -------------------------- setup function -------------------------- *//
extern "C" void setup_torch0x2eactivation(void) { //change 0x2e to "."
    torch_activation_class = class_new(
        gensym("torch.activation"), 
        (t_newmethod)torch_activation_new, 
        (t_method)torch_activation_destroy, 
        sizeof(t_torch_activation), 
        CLASS_DEFAULT, 
        A_GIMME, //Argument type: Optional Symbol (defaults to &s_)
        0);


    class_addlist(torch_activation_class, (t_method)torch_activation_process); //receive a list of values
    class_addmethod(torch_activation_class, (t_method)activation_functions, gensym("activation"), A_GIMME, 0); //receive an activation function name
    class_addmethod(torch_activation_class, (t_method)torch_activation_device, gensym("device"), A_GIMME, 0); //set the device (cpu or cuda)
    class_addmethod(torch_activation_class, (t_method)torch_activation_shape, gensym("shape"), A_GIMME, 0); //receive a shape for the tensor
    class_addmethod(torch_activation_class, (t_method)alpha_value, gensym("alpha"), A_FLOAT, 0); //receive alpha value
    class_addmethod(torch_activation_class, (t_method)torch_activation_set_dim, gensym("dim"), A_FLOAT, 0); //receive dim value
    class_addmethod(torch_activation_class, (t_method)lambda_value, gensym("lambda"), A_FLOAT, 0); //receive lambda value
    class_addmethod(torch_activation_class, (t_method)list_activation_functions, gensym("info"), A_GIMME, 0); //list all available activation functions
    class_addmethod(torch_activation_class, (t_method) torch_add_to_module, gensym("add"), A_GIMME, 0); // add to module
    class_addmethod(torch_activation_class, (t_method) torch_activation_remove, gensym("remove"), A_GIMME, 0); // add to module
}

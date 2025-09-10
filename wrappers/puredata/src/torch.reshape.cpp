#include "m_pd.h"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <functional>
#include <sstream>
#include "../../../core/include/core_dp_reshape.h"
#include "../utils/include/pd_torch_device_adapter.h"
#include "../utils/include/pd_arg_parser.h"
#include "../utils/include/pd_torch_types.h" // Include the object struct definition
#include "../utils/include/pd_torch_utils.h" // Utility functions
#include "../utils/include/pd_global_state.h" // shared library to manage global state for pd modules and layers


static t_class *torch_reshape_class;


//* ----------------------------- receives resehape method name ----------------------------- *//
static void torch_reshape_methods(t_torch_reshape *x, t_symbol *s, int argc, t_atom *argv) {
    // Check if the first argument is a symbol
    if (argc < 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.reshape: Please provide a reshape method name, e.g., 'transpose 1 2'.");
        return;
    }

    // set the reshape method to reshape_name
    t_symbol *reshape_name = atom_getsymbol(&argv[0]); // get the reshape method name provided
    const std::string method = reshape_name->s_name;

    // converts arguments to the appropriate types
    std::vector<float> args;
    args.reserve(argc > 1 ? (argc - 1) : 0);
    for (int i = 1; i < argc; ++i) {
        args.push_back(atom_getfloat(argv + i));
    }

    // set the reshape method configuration
    auto config = contorchionist::core::dp_reshape::set_reshape_method(method, args, {}, 0, -1, {});
    if (!config.success) {
        pd_error(x, "torch.reshape: %s", config.error_message.c_str());
        return;
    }

    // Set the reshape parameters
    x->reshape_method = reshape_name;
    x->shape = config.params.shape;
    x->dims  = config.params.dims;
    x->dim1  = config.params.dim1;
    x->dim2  = config.params.dim2;
    // post("Reshape method set to: %s", reshape_name->s_name);
    PdTorchUtils::print_reshape_config(x, method, x->verbose);
}

//* ------------------ receives the input shape ------------------ */
static void torch_reshape_shape(t_torch_reshape *x, t_symbol *s, int argc, t_atom *argv) {
    x->input_shape.clear();
    for (int i = 0; i < argc; ++i){
        x->input_shape.push_back((int64_t)atom_getfloat(argv + i));
        if (x->verbose) {
            post("torch.reshape: input shape[%d] = %lld", i, x->input_shape[i]);
        }
    }
    
}

//* ------------------ sends the in/out shape to torch.sequential ------------------ *//
static void torch_reshape_send_shapes(t_torch_reshape *x, t_symbol *module_name) {
    //create t_atom array to store the input shape
    int shape_size = x->input_shape.size();
    if (shape_size == 0) {
        pd_error(x, "torch.reshape: No input shape set to send.");
        return;
    }
    t_atom *shape = (t_atom *)getbytes(sizeof(t_atom) * shape_size);
    for (int i = 0; i < shape_size; ++i) {
        SETFLOAT(&shape[i], x->input_shape[i]);
    }
    /*
    Find a torch.sequential module instance by the module name associated with it and the global class pointer (PDGlobalState::torch_sequential_class). 
    This allows torch.linear to send messages directly to a specific instance of torch.sequential. 
    torch_sequential_class need to be defined in the header file
    */
    t_pd *target = (t_pd *)pd_findbyclass(module_name, PDGlobalState::torch_sequential_class);

   // look for the correct instance of torch.sequential and send the input shape
   if (target) {
       pd_typedmess(target, gensym("input_shape"), shape_size, shape);

       post("torch.reshape input shape=[%d, %d, %d] sent to module '%s'",
            (int)atom_getfloat(&shape[0]), //in_fetures
            (int)atom_getfloat(&shape[1]), //out_features
            (int)atom_getfloat(&shape[2]), //out_features
            module_name->s_name); // torch.sequential module name 
   } else {
       pd_error(x, "torch.reshape: Could not find module '%s' to send input shape.", module_name->s_name);
   }
   // freebytes buffer
   freebytes(shape, sizeof(t_atom) * shape_size);
}


//* ------------------ add a reshape layer to torch.sequential ------------------ */
static void torch_reshape_add_to_module(t_torch_reshape *x, t_symbol *s, int argc, t_atom *argv) {
    if (x->added_to_module) {
        pd_error(x, "torch.reshape: This instance has already been added to a module.");
        return;
    }
    if (argc != 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.reshape: Please provide a module name to add to.");
        return;
    }

    t_symbol *module_name = atom_getsymbol(argv);

    // create the reshape layer
    std::string method = x->reshape_method ? x->reshape_method->s_name : "";
    bool added = false;

    if (method == "view" || method == "reshape") {
        auto shape = x->shape;
        added = PDGlobalState::add_layer_to_module(
            x, "reshape",
            torch::nn::Functional([shape](const at::Tensor& t) {
                return t.view(shape);
            }),
            module_name,
            PDGlobalState::reshape_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (method == "flatten") {
        int64_t start_dim = x->dim1;
        int64_t end_dim = x->dim2;
        added = PDGlobalState::add_layer_to_module(
            x, "reshape",
            torch::nn::Functional([start_dim, end_dim](const at::Tensor& t) {
                return torch::flatten(t, start_dim, end_dim);
            }),
            module_name,
            PDGlobalState::reshape_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (method == "squeeze") {
        int64_t dim = x->dim1;
        added = PDGlobalState::add_layer_to_module(
            x, "reshape",
            torch::nn::Functional([dim](const at::Tensor& t) {
                return t.squeeze(dim);
            }),
            module_name,
            PDGlobalState::reshape_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (method == "unsqueeze") {
        int64_t dim = x->dim1;
        added = PDGlobalState::add_layer_to_module(
            x, "reshape",
            torch::nn::Functional([dim](const at::Tensor& t) {
                return t.unsqueeze(dim);
            }),
            module_name,
            PDGlobalState::reshape_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (method == "permute") {
        auto dims = x->dims;
        added = PDGlobalState::add_layer_to_module(
            x, "reshape",
            torch::nn::Functional([dims](const at::Tensor& t) {
                return t.permute(dims);
            }),
            module_name,
            PDGlobalState::reshape_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else if (method == "transpose") {
        int64_t dim1 = x->dim1;
        int64_t dim2 = x->dim2;
        added = PDGlobalState::add_layer_to_module(
            x, "reshape",
            torch::nn::Functional([dim1, dim2](const at::Tensor& t) {
                return t.transpose(dim1, dim2);
            }),
            module_name,
            PDGlobalState::reshape_registry,
            x->added_layer_index, x->added_layer_name,
            x->added_to_module, x->added_to_module_name
        );
    } else {
        pd_error(x, "torch.reshape: Unknown reshape method '%s'", method.c_str());
        return;
    }

    if (!added) {
        pd_error(x, "torch.reshape: Failed to add reshape layer to module.");
    }

    //if is the first layer sends the in/out shape to a instance of torch.sequential that matches the module name only if this is the first layer
    auto it = PDGlobalState::module_registry.find(module_name->s_name);
    if (it != PDGlobalState::module_registry.end()) {
        std::shared_ptr<torch::nn::Sequential> container = it->second;
        if (container && container->get()->size() == 1) {
            torch_reshape_send_shapes(x, module_name);
        }
    }
}

//* ------------------------ removes the reshape layer from a specific container (module) created by pdtoch.sequential ------------------------ *//
static void torch_reshape_remove (t_torch_reshape *x, t_symbol *s, int argc, t_atom *argv){
    if (x->added_to_module) {
        PDGlobalState::remove_layer(x, "reshape", x->verbose);
    } else {
        pd_error(x, "torch.reshape: This instance was not added to a module.");
    }
}



//* ------------------ reshape the input tensor ------------------ */
static void torch_reshape_list(t_torch_reshape *x, t_symbol *s, int argc, t_atom *argv) {
    
    std::vector<float> input; // vector to store the input values received
    // get the input values from the list received
    for (int i = 0; i < argc; ++i){
        input.push_back(atom_getfloat(argv + i));
    }
    
    // use the input shape if it is set, otherwise use the number of elements in the list
    std::vector<int64_t> input_shape = x->input_shape.empty() ? std::vector<int64_t>{(int64_t)argc} : x->input_shape;

    // check if the input shape is valid
    int64_t shape_prod = 1;
    for (auto v : input_shape) shape_prod *= v;
    if (shape_prod != argc) {
        pd_error(x, "torch.reshape: Input shape product (%lld) does not match number of elements in list (%d)", shape_prod, argc);
        return;
    }

    // create a tensor with the shape received
    at::Tensor tensor = torch::from_blob(input.data(), input_shape, torch::kFloat).clone();

    // get the parameters
    contorchionist::core::dp_reshape::ReshapeParams params;
    params.shape = x->shape;
    params.dims  = x->dims;
    params.dim1  = x->dim1;
    params.dim2  = x->dim2;

    // get the selected method
    const std::string method = (x->reshape_method && x->reshape_method->s_name)
        ? std::string{x->reshape_method->s_name}
        : std::string{};

    // apply reshape
    auto result = contorchionist::core::dp_reshape::ReshapeProcessor(tensor, method, params, x->device);
    if (!result.success) {
        pd_error(x, "torch.reshape: %s", result.error_message.c_str());
        return;
    }

    // get the output tensor
    at::Tensor out = result.output_tensor.to(torch::kCPU);

    if (x->recursive) {
        PdTorchUtils::send_tensor_recursive(out, x->x_out1, 0);
    }
    else{
        // flatten the output tensor and convert to std::vector<float>
        at::Tensor flat = out.contiguous().view({-1});
        PdTorchUtils::send_tensor_1d(flat, x->x_out1);
    }

    // free buffers
    if (x->out_tensor) {
        freebytes(x->out_tensor, sizeof(t_atom) * x->out_tensor_size);
        x->out_tensor = nullptr;
        x->out_tensor_size = 0;
    }
}



//* ------------------ constructor ------------------ */
void *torch_reshape_new(t_symbol *s, int argc, t_atom *argv) {
    t_torch_reshape *x = (t_torch_reshape *)pd_new(torch_reshape_class);

    post("torch.reshape: libtorch version: %s", TORCH_VERSION);

     // default method
    x->reshape_method = gensym("reshape");
    x->dim1 = 0;
    x->dim2 = 0;
    x->out_tensor = nullptr;
    x->out_shape = nullptr;
    x->out_tensor_size = 0;
    x->out_shape_size = 0;
    x->recursive = false;
    x->shape.clear();
    x->dims.clear();

    //arg parse
    pd_utils::ArgParser parser(argc, argv, (t_object*)x);
    x->verbose = parser.has_flag("verbose v");

    x->recursive = parser.has_flag("multioutput mout");
    if (x->verbose) {
        post("torch.reshape: output mode=%s", x->recursive ? "structured" : "flattened");
    } 
    
    bool device_flag_present = parser.has_flag("device d");
    std::string device_arg_str = parser.get_string("device d", "cpu");
    // Get device from string
    auto device_result = get_device_from_string(device_arg_str);
    // x->device = device_result.first;
    bool device_parse_success = device_result.second;

    pd_parse_and_set_torch_device(
        (t_object*)x,
        x->device,
        device_arg_str,
        x->verbose,
        "torch.reshape",
        device_flag_present
    );

    // 1. detect flags for methods
    const char* method_flags[] = {
        "reshape", "view", "flatten", "squeeze", "unsqueeze", "permute", "transpose"
    };
    std::string chosen_method;
        for (const auto& method : method_flags) {
            if (parser.has_flag(method)) {
                if (!chosen_method.empty()) {
                    pd_error(x, "torch.reshape: Múltiplos métodos especificados (-%s e -%s). Usando %s.", 
                             chosen_method.c_str(), method, chosen_method.c_str());
                    continue;
                }
                chosen_method = method;
            }
        }
        // fallback
        if (chosen_method.empty()) {
            chosen_method = "reshape";
            if (x->verbose) {
                post("torch.reshape: No method specified, using 'reshape' [-1]");
            }
        }

        // 2. get specific parameters for each method
        std::vector<int64_t> shape_i64;
        std::vector<int64_t> dims_i64;
        int64_t dim1 = 0, dim2 = -1;

        if (chosen_method == "reshape" || chosen_method == "view" || chosen_method == "r" || chosen_method == "rshp") {
            // -shape
            auto shape_f = parser.get_float_list("shape s", {});
            if (shape_f.empty() && parser.has_flag(chosen_method)) {
                shape_f = parser.get_float_list(chosen_method, {-1});
            }
            if (shape_f.empty()) {
                shape_f = {-1}; // fallback
            }
            for (auto v : shape_f) shape_i64.push_back((int64_t)v);

        } else if (chosen_method == "flatten" || chosen_method == "f" || chosen_method == "flat") {
            dim1 = (int64_t)parser.get_float("dim1 start", 0);
            dim2 = (int64_t)parser.get_float("dim2 end", -1);

        } else if (chosen_method == "squeeze" || chosen_method == "sq" || chosen_method == "s") {
            dim1 = (int64_t)parser.get_float("dim", 0);

        } else if (chosen_method == "unsqueeze" || chosen_method == "unsq" || chosen_method == "uns") {
            dim1 = (int64_t)parser.get_float("dim", 0);

        } else if (chosen_method == "permute" || chosen_method == "perm" || chosen_method == "p") {
            auto dims_f = parser.get_float_list("dims", {});
            if (dims_f.empty() && parser.has_flag(chosen_method)) {
                dims_f = parser.get_float_list(chosen_method, {});
            }
            for (auto v : dims_f) dims_i64.push_back((int64_t)v);

        } else if (chosen_method == "transpose" || chosen_method == "t" || chosen_method == "trans") {
            dim1 = (int64_t)parser.get_float("dim0", 0);
            dim2 = (int64_t)parser.get_float("dim1", 1);
        }

        // 3. get input shape if provided
        auto input_shape_f = parser.get_float_list("input_shape inshape", {});
        x->input_shape.clear();
        for (auto v : input_shape_f) {
            x->input_shape.push_back((int64_t)v);
        }

        // 4. set the configuration
        auto cfg = contorchionist::core::dp_reshape::set_reshape_method(
            chosen_method, {}, shape_i64, dim1, dim2, dims_i64
        );

        if (!cfg.success) {
            pd_error(x, "torch.reshape: %s (fallback para reshape [-1])", cfg.error_message.c_str());
            if (x->verbose) {
                post("torch.reshape: set to default 'reshape' [-1]");
            }
        } else {
            x->reshape_method = gensym(chosen_method.c_str());
            x->shape = cfg.params.shape;
            x->dims  = cfg.params.dims;
            x->dim1  = cfg.params.dim1;
            x->dim2  = cfg.params.dim2;
        }
    PdTorchUtils::print_reshape_config(x, chosen_method, x->verbose);
    x->x_out1 = outlet_new(&x->x_obj, &s_anything);
    return (void *)x;
}


//* ------------------ destructor ------------------ *//
void torch_reshape_destroy(t_torch_reshape *x) {

    PDGlobalState::remove_layer(x, "reshape", x->verbose); // remove the layer from the module
    freebytes(x->out_tensor, x->out_tensor_size * sizeof(t_atom));
    freebytes(x->out_shape, x->out_shape_size * sizeof(t_atom));
    outlet_free(x->x_out1);
}


//* ------------------ setup ------------------ */
extern "C" void setup_torch0x2ereshape(void) {
    torch_reshape_class = class_new(
        gensym("torch.reshape"),
        (t_newmethod)torch_reshape_new, 
        (t_method)torch_reshape_destroy, 
        sizeof(t_torch_reshape), 
        CLASS_DEFAULT, 
        A_GIMME, //Argument type: Optional Symbol (defaults to &s_)
        0);

    class_addlist(torch_reshape_class, (t_method)torch_reshape_list); //receive a list of values
    class_addmethod(torch_reshape_class, (t_method)torch_reshape_methods, gensym("method"), A_GIMME, 0); // set the reshape method
    class_addmethod(torch_reshape_class, (t_method)torch_reshape_shape, gensym("input_shape"), A_GIMME, 0); //receive a shape for the tensor
    class_addmethod(torch_reshape_class, (t_method)torch_reshape_add_to_module, gensym("add"), A_GIMME, 0); // add to a module
    class_addmethod(torch_reshape_class, (t_method)torch_reshape_remove, gensym("remove"), A_GIMME, 0); // remove from a module
}
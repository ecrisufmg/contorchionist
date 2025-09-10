#include "m_pd.h"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <functional>
#include "../utils/include/pd_torch_device_adapter.h"
#include "../utils/include/pd_torch_types.h" // Include the object struct definition
#include "../utils/include/pd_arg_parser.h"
#include "../utils/include/pd_torch_utils.h" // Utility functions
#include "../utils/include/pd_global_state.h" // manages global state for pd modules and layers
#include "../../../core/include/core_dp_linear.h"
#include "../../../core/include/core_util_tensors.h"


static t_class *torch_linear_class;


//* ------------------ rebuild the linear layer with new parameters ------------------ */
static void torch_linear_rebuild(t_torch_linear *x) {
    // free module
    x->linear = nullptr; // <--- important!

    //creates the linear layer
    x->linear = contorchionist::core::dp_linear::create_linear_layer(
        x->in_features, x->out_features, x->bias, x->device);

    std::string device_name = pd_torch_device_friendly_name(x->device);

    if(x->verbose){
        post("torch.linear: Created with in_features=%d, out_features=%d, bias=%s, device=%s", x->in_features, x->out_features, x->bias ? "true" : "false", device_name.c_str());
    }
}


//* ------------------ in_features/out_features ------------------ */
static void torch_linear_size(t_torch_linear *x, t_symbol *s, int argc, t_atom *argv) {
    //check if the list received is valid
    if (argc < 1 || argv[0].a_type != A_FLOAT || argv[1].a_type != A_FLOAT) {
        pd_error(x, "torch.linear: Provide in_features and out_features.");
        return;
    }
    //set the in_features and out_features values
    x->in_features = (int)atom_getfloat(argv + 0);
    x->out_features = (int)atom_getfloat(argv + 1);
    // rebuild the linear layer with new parameters
    torch_linear_rebuild(x); 
    
}

//* ------------------ enable/disable bias ------------------ */
static void torch_linear_bias(t_torch_linear *x, t_floatarg b) {
    if(b < 0 || b > 1) {
        pd_error(x, "torch.linear: bias must be 0 or 1.");
        return;
    }
    x->bias = (b != 0);
    // rebuild the linear layer with new parameters
    torch_linear_rebuild(x); 
}


//* -------------------- batch_size --------------------- *//
static void torch_linear_batch_size(t_torch_linear *x, t_floatarg b) {
    int batch = (int)b;
    if (batch <= 0) {
        pd_error(x, "torch.linear: batch_size must be positive integer.");
        return;
    }
    x->batch_size = batch;
    post("torch.linear: batch_size = %d", x->batch_size);
}


//* ----------------------- sets the device (cpu or cuda) --------------------- *//
static void torch_linear_device(t_torch_linear *x, t_symbol *s, int argc, t_atom *argv) {
    // Check if the first argument is a symbol
    if (argc != 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.linear: Please provide a device (cpu or cuda).");
        return;
    }
    
    std::string dev = atom_getsymbol(argv)->s_name;
    pd_parse_and_set_torch_device(
        (t_object*)x,
        x->device,
        dev,
        x->verbose,
        "torch.linear",
        true
    );
    // rebuild the linear layer with new parameters
    torch_linear_rebuild(x); 
}


//* ------------------ sends the in/out shape to pdtorch.sequential ------------------ *//
static void torch_linear_send_shapes(t_torch_linear *x, t_symbol *module_name) {
    //create t_atom array to store the input shape
    t_atom shape[2];
    SETFLOAT(&shape[0], x->in_features); //torch.linear needs to send only the in_features to pdtorch.sequential create the input shape
    // SETFLOAT(&shape[1], x->in_features);
    /*
    Find a pdtorch.sequential module instance by the module name associated with it and the global class pointer (PDGlobalState::pdtorch_sequential_class). 
    This allows torch.linear to send messages directly to a specific instance of pdtorch.sequential. 
    pdtorch_sequential_class need to be defined in the header file
    */
   t_pd *target = (t_pd *)pd_findbyclass(module_name, PDGlobalState::torch_sequential_class);

   // look for the correct instance of pdtorch.sequential and send the input shape
   if (target) {
       pd_typedmess(target, gensym("input_shape"), 1, shape);

       post("torch.linear: input shape=[%d] sent to module '%s'",
            (int)atom_getfloat(&shape[0]), //in_fetures
            // (int)atom_getfloat(&shape[1]), //batch_size
            module_name->s_name); // pdtorch.sequential module name 
   } else {
       pd_error(x, "torch.linear: Could not find module '%s' to send input shape.", module_name->s_name);
   }
}



//* ------------------ adds the linear layer to a specific container (module) created by pdtoch.sequential  ------------------ *//
static void torch_linear_add_to_module(t_torch_linear *x, t_symbol *s, int argc, t_atom *argv) {
    //check if this instance of torch.linear is already added
    if (x->added_to_module) {
        pd_error(x, "torch.linear: This instance has already been added to a module.");
        return;
    }
    //check if a module name was provided
    if (argc != 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.linear: Please provide a module name to add to.");
        return;
    }
    //get the module name provided
    t_symbol *module_name = atom_getsymbol(argv);
 
    //add the layer to the pdtorch.sequential module
    bool added = PDGlobalState::add_layer_to_module(
        x, "linear", *x->linear, module_name,
        PDGlobalState::linear_registry,
        x->added_layer_index, x->added_layer_name,
        x->added_to_module, x->added_to_module_name
    );

    if (!added){
        return;
    }

    //sends the in/out shape to a instance of pdtorch.sequential that matches the module name only if this is the first layer
    auto it = PDGlobalState::module_registry.find(module_name->s_name);
    if (it != PDGlobalState::module_registry.end()) {
        std::shared_ptr<torch::nn::Sequential> container = it->second;
        if (container && container->get()->size() == 1) {
            torch_linear_send_shapes(x, module_name);
        }
    }
}

//* ------------------------ removes the linear layer from a specific container (module) created by pdtoch.sequential ------------------------ *//
static void torch_linear_remove(t_torch_linear *x, t_symbol *s, int argc, t_atom *argv) {
    if (x->added_to_module) {
        PDGlobalState::remove_layer(x, "linear", x->verbose);
    } else {
        pd_error(x, "torch.linear: This instance was not added to a module.");
    }
}


//* ------------------ forwards a fully connected linear layer ------------------ *//
static void torch_linear_forward(t_torch_linear *x, t_symbol *s, int argc, t_atom *argv) {
    //check if the layer is initialized
    if (!x->linear) {
        pd_error(x, "torch.linear: layer not initialized.");
        return;
    }

    //check if the input list is compatible with the expected input shape
    if (argc != x->in_features) {
        pd_error(x, "torch.linear: expected %lld input values, got %d.", static_cast<long long>(x->in_features), argc);
        return;
    }

    // get the input values
    std::vector<float> values;
    values.reserve(argc);
    for (int i = 0; i < argc; ++i) {
        if (argv[i].a_type != A_FLOAT) {
            pd_error(x, "torch.linear: all inputs must be floats.");
            return;
        }
        values.push_back(atom_getfloat(argv + i));
    }

    // create a tensor with the input values and send it to the device
    auto input = contorchionist::core::util_tensors::vectorToTensor(values, x->device).view({1, x->in_features});

    // forward the input tensor through the fully connected layer
    auto output = contorchionist::core::dp_linear::LinearProcessor(input, x->linear);
    if (!output.success) {
        pd_error(x, "torch.linear: %s", output.error_message.c_str());
        return;
    }

    // move the output tensor to cpu
    auto out_cpu = output.output.to(torch::kCPU).contiguous().view({x->out_features});
    
    // allocate memory for the output buffer
    x->out = (t_atom*)getbytes(x->out_features * sizeof(t_atom));

    // copy the output values to the output buffer
    for (int i = 0; i < x->out_features; ++i) {
        SETFLOAT(&x->out[i], out_cpu[i].item<float>());
    }
    outlet_list(x->x_out1, &s_list, x->out_features, x->out);
}


//* -------------------- constructor -------------------- *//
void *torch_linear_new(t_symbol *s, int argc, t_atom *argv) {
    t_torch_linear *x = (t_torch_linear *)pd_new(torch_linear_class);

    post("torch.linear: libtorch version: %s", TORCH_VERSION);

    //default parameters
    x->in_features = 8; // number of input features
    x->out_features = 4; // number of output features (neurons)
    x->batch_size = 1; // batch size
    x->bias = false; // if bias is used
    x->added_to_module = false; 
    x->device = torch::kCPU;

    // Parse arguments
    if (argc > 0) {
        pd_utils::ArgParser parser(argc, argv, (t_object*)x);

        x->verbose = parser.has_flag("verbose v");

        std::vector<float> in_out = parser.get_float_list("size s I/O", {});
        if (in_out.size() == 2) {
            x->in_features = (int)in_out[0];
            x->out_features = (int)in_out[1];
        }

        x->in_features =  static_cast<int>(parser.get_float("in in_feat", 1));
        x->out_features =  static_cast<int>(parser.get_float("out out_feat", 1));
        x->batch_size =  static_cast<int>(parser.get_float("batch btz", 1));
        

        if (parser.has_flag("bias b")) {
            x->bias = true;
        }

        // parse device 
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
            "torch.linear",
            device_flag_present
        );
    }

    // creates a fully connected linear layer
    x->linear = contorchionist::core::dp_linear::create_linear_layer(
        x->in_features, x->out_features, x->bias, x->device
    );

    if (x->verbose) {
        post("torch.linear: Created with in_features=%d, out_features=%d, bias=%s", x->in_features, x->out_features, x->bias ? "true" : "false");
    }

    x->x_out1 = outlet_new(&x->x_obj, &s_list);
    return (void *)x;
}

//* ------------------ destructor ------------------ *//
void torch_linear_destroy(t_torch_linear *x) {

    PDGlobalState::remove_layer(x, "linear", x->verbose); // remove the layer from the modul
    freebytes(x->out, 1 * sizeof(t_atom));
    outlet_free(x->x_out1);
}

//* ------------------ setup function ------------------ *//
extern "C" void setup_torch0x2elinear(void) {
    torch_linear_class = class_new(
        gensym("torch.linear"),
        (t_newmethod)torch_linear_new, 
        (t_method)torch_linear_destroy, 
        sizeof(t_torch_linear), 
        CLASS_DEFAULT, 
        A_GIMME, //Argument type: Optional Symbol (defaults to &s_)
        0);

        class_addlist(torch_linear_class, (t_method)torch_linear_forward); // process the input list
        class_addmethod(torch_linear_class, (t_method)torch_linear_add_to_module, gensym("add"), A_GIMME, 0); // add to a module
        class_addmethod(torch_linear_class, (t_method)torch_linear_remove, gensym("remove"), A_GIMME, 0); // add to a module
        class_addmethod(torch_linear_class, (t_method)torch_linear_size, gensym("size"), A_GIMME, 0); //receive in_features and out_features
        class_addmethod(torch_linear_class, (t_method)torch_linear_bias, gensym("bias"), A_FLOAT, 0); // bias
        class_addmethod(torch_linear_class, (t_method)torch_linear_device, gensym("device"), A_GIMME, 0); // device
        class_addmethod(torch_linear_class, (t_method)torch_linear_batch_size, gensym("batch_size"), A_FLOAT, 0); // batch size

}
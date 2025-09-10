#include "m_pd.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include "../utils/include/pd_torch_device_adapter.h"
#include "../utils/include/pd_arg_parser.h"
#include "../utils/include/pd_torch_types.h" // Include the object struct definition
#include "../utils/include/pd_torch_utils.h" // Utility functions
#include "../../../core/include/core_nn_ts_manager.h" // Include the model manager header for loading models, select methods, extract shape, etc.
#include "../../../core/include/core_dp_torchts.h"



static t_class *torch_ts_class;


//* ----------------------- sets the device (cpu or cuda) --------------------- *//
static void torch_ts_device(t_torch_ts *x, t_symbol *s, int argc, t_atom *argv) {
    // Check if the first argument is a symbol
    if (argc != 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.ts: Please provide a device (cpu or cuda).");
        return;
    }
    // get the device name received
    t_symbol *device_name = atom_getsymbol(&argv[0]);
    std::string device_arg_str = device_name->s_name;

    // Get device from string
    auto device_result = get_device_from_string(device_arg_str);
    
    // set the device
    pd_parse_and_set_torch_device(
        (t_object*)x,
        x->device,
        device_arg_str,
        x->verbose,
        "torch.ts",
        true
    );

    // post("torch.ts: Device set to %s", dev.c_str());
    if (x->model){
         x->model->to(x->device);
    }

}


// * ------------------ load the model from a file ------------------ */
static void torch_ts_model(t_torch_ts *x,t_symbol *filename, t_symbol *format) {
    char dirresult[MAXPDSTRING]; //buffer to store the directory address
    char *nameresult; //pointer to store the name of the file
    int fd; //file descriptor
    // get the filename
    std::string model_name(filename->s_name);
    const char *canvas_dir = canvas_getdir(x->m_canvas)->s_name;

    // try to open the file
    fd = open_via_path(canvas_dir, model_name.c_str(), "", dirresult, &nameresult, MAXPDSTRING, 1);
        //if the file is found and the model is not empty try to load it
        if (fd >= 0) {
            sys_close(fd);
            char fullpath[MAXPDSTRING];
            snprintf(fullpath, MAXPDSTRING, "%s/%s", dirresult, nameresult);
            char normalized[MAXPDSTRING]; // buffer to store the normalized path of the file (without special characters)
            sys_unbashfilename(fullpath, normalized);

        //store the error message if the model fails to load
        std::string load_error_message; 
        // load the model from the path normalized and send it to device
        bool loaded = contorchionist::core::nn_ts_manager::load_torchscript_model(normalized, x->model, x->device, &load_error_message);

        // check if the model was loaded successfully
        if (!loaded) {
            x->loaded_model = false;
            pd_error(x, "torch.ts~: %s", load_error_message.c_str());
            return;
        } else {
            x->loaded_model = true;
        }
        
        // get the available methods and attributes from the loaded model
        std::string methods_log;
        std::string attributes_log;
        //methods
        x->available_methods = contorchionist::core::nn_ts_manager::get_string_list_from_method(x->model.get(), "get_methods", &methods_log);
        if (!methods_log.empty() && x->verbose) {
            post("torch.ts~: methods = %s", methods_log.c_str());
        }
        //attributes
        x->available_attributes = contorchionist::core::nn_ts_manager::get_string_list_from_method(x->model.get(), "get_attributes", &attributes_log);
        if (!attributes_log.empty() && x->verbose) {
            post("torch.ts~: attributes = %s", attributes_log.c_str());
        }

        //set the default method to be used on the loaded model as "forward"
        x->selected_method = "forward";

        std::string error_message;
        std::vector<std::string> log_messages;

        bool shapes_extracted = contorchionist::core::nn_ts_manager::extract_shape_method(
                x->model.get(),
                x->selected_method,
                &x->input_shape_model,
                &x->output_shape_model,
                &x->tensors_struct->input_tensor_model,
                &error_message,
                &log_messages
            );
        
        // Check for errors and display them
        if (!shapes_extracted) {
            pd_error(x, "torch.ts: %s", error_message.c_str());
        }

        if (x->verbose) {
            // Display log messages
            for (const auto& msg : log_messages) {
                post("torch.ts: %s", msg.c_str());
            }
        }
        
        //if the file is not found, send an error
    } else {
        pd_error(x, "torch.ts: File not found: %s", model_name.c_str());
        x->loaded_model = false; //model is not loaded
    }
}

// * ------------------ set the model method to be used on loaded model ------------------ */
static void torch_ts_select_method(t_torch_ts *x, t_symbol *s, int argc, t_atom *argv) {
    if (argc < 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.ts: Provide the method name to select.");
        return;
    }
    // get the method name from the first argument
    std::string method_name = atom_getsymbol(argv)->s_name;

    // check if the method name is valid
    auto it = std::find(x->available_methods.begin(), x->available_methods.end(), method_name);
    if (it == x->available_methods.end()) {
        pd_error(x, "torch.ts: Method '%s' not found in loaded model.", method_name.c_str());
        return;
    }
    // set the selected method to the method name received
    x->selected_method = method_name;

    std::string error_message;
    std::vector<std::string> log_messages;

    bool shapes_extracted = contorchionist::core::nn_ts_manager::extract_shape_method(
            x->model.get(),
            x->selected_method,
            &x->input_shape_model,
            &x->output_shape_model,
            &x->tensors_struct->input_tensor_model,
            &error_message,
            &log_messages
        );
        
     // Check for errors and display them
    if (!shapes_extracted) {
        pd_error(x, "torch.ts: %s", error_message.c_str());
    }

    if(x->verbose) {
        // Display log messages
        for (const auto& msg : log_messages) {
            post("torch.ts: %s", msg.c_str());
        }
        post("torch.ts: Selected method set to '%s'", method_name.c_str());
    }
}



//* ------------------ forward the input tensor through the model loaded ------------------ */
static void torch_ts_forward(t_torch_ts *x, t_symbol *s, int argc, t_atom *argv) {

    // Check if the input is a list of floats
    if (argc == 0) {
        pd_error(x, "torch.ts: No input list received.");
        return;
    }

    // if a model is loaded, check if the input shape is setted 
    if (x->loaded_model && x->model) {
        if (x->input_shape_model.empty()) { 
            pd_error(x, "torch.ts: input_shape_model not set or incomplete.");
            return;
        }
        // check if the selected method is valid
        if (!x->model->find_method(x->selected_method)) {
            pd_error(x, "Method '%s' not found!", x->selected_method.c_str());
            return;
        }
        // Check if the input shape is set
        if (x->input_shape_model.size() == 0) {
            pd_error(x, "torch.ts: input_shape not set or incomplete.");
            return;
        }
    } else {
        // If no model is loaded, we cannot proceed
        pd_error(x, "torch.ts: No model loaded.");
        return;
    }

    // check if the input shape is compatible with the model input shape
    int64_t expected_size = 1;

    for (auto v : x->input_shape_model){
        expected_size *= v; //multiply the shape dimension to get the expected size
    } 

    if (argc < expected_size) {
        pd_error(x, "torch.ts: Input has %d elements, expected at least %lld (input_shape_model).", argc, expected_size);
        return;
    }

    // prepare input tensor

    //convert atoms to float vector
    std::vector<float> values;
    for (int i = 0; i < expected_size; ++i) {
        if (argv[i].a_type != A_FLOAT) {
            pd_error(x, "torch.ts: Only float values are accepted in list.");
            return;
        }
        values.push_back(atom_getfloat(argv + i));
    }
    // create input tensor
    at::Tensor input_tensor;
    try {
        input_tensor = torch::from_blob(
            values.data(),
            torch::IntArrayRef(x->input_shape_model),
            torch::TensorOptions().dtype(torch::kFloat)
        ).clone();
    } catch (const c10::Error& e) {
        pd_error(x, "torch.ts: Error creating input tensor: %s", e.what());
        return;
    }

    // process torchscript model (TorchTSProcessor)
    at::Tensor output_tensor;
    try {
        output_tensor = contorchionist::core::dp_torchts::TorchTSProcessor(
            x->model.get(),           // model pointer
            x->selected_method,       // method name
            input_tensor,             // input tensor
            x->device                 // target device
        );
    } catch (const c10::Error& e) {
        pd_error(x, "torch.ts: %s", e.what());
        return;
    }

    // prepare output tensor for Pure Data outlet
    try {
        // Flatten output and move to CPU for Pure Data
        output_tensor = output_tensor.view({-1}).cpu();
        int64_t out_size = output_tensor.size(0);
            
        // Allocate Pure Data atoms
        x->out = (t_atom*)getbytes(sizeof(t_atom) * out_size);
        if (!x->out) {
            pd_error(x, "torch.ts: Failed to allocate output memory");
            return;
        }
        // Convert tensor to Pure Data atoms
        for (int64_t i = 0; i < out_size; ++i) {
            SETFLOAT(&x->out[i], output_tensor[i].item<float>());
        }
        // Send to outlet and free memory
        outlet_list(x->x_out1, &s_list, out_size, x->out);
        freebytes(x->out, sizeof(t_atom) * out_size);
        
    } catch (const c10::Error& e) {
        pd_error(x, "torch.ts: Error converting output: %s", e.what());
        return;
    }
}


//* -------------------------- constructor -------------------------- //
void *torch_ts_new(t_symbol *s, int argc, t_atom *argv) {
    t_torch_ts *x = (t_torch_ts *)pd_new(torch_ts_class);

    x->loaded_model = false; // model not loaded by default
    x->tensors_struct = new PdTorchTensors(); // create the tensors struct
    x->m_canvas = canvas_getcurrent(); // get the current canvas
    x->name = gensym("torch.ts"); // set the name of the object
    x->out = nullptr; // output buffer

    //---- argument and flag parsing ---- //
    pd_utils::ArgParser parser(argc, argv, (t_object*)x);

    // verbose
    x->verbose = parser.has_flag("verbose v");

    // method
    x->selected_method = parser.get_string("method m", "forward");

    // device
    bool device_flag_present = parser.has_flag("device d");
    std::string device_arg_str = parser.get_string("device d", "cpu");
    // Get device from string
    auto device_result = get_device_from_string(device_arg_str);
    
    // set the device
    pd_parse_and_set_torch_device(
        (t_object*)x,
        x->device,
        device_arg_str,
        x->verbose,
        "torch.ts",
        device_flag_present
    );

    x->x_out1 = outlet_new(&x->x_obj, &s_list);
    return (void *)x;
}


//* ------------------------- destructor ------------------------- //
void torch_ts_free(t_torch_ts *x) {

    delete x->tensors_struct; // free the tensors struct
    x->tensors_struct = nullptr;
    // free the output buffer and outlet
    // freebytes(x->out, 1 * sizeof(t_atom));
    outlet_free(x->x_out1);
}


//* -------------------------- setup function -------------------------- //
extern "C" void setup_torch0x2ets(void) {
    torch_ts_class = class_new(
        gensym("torch.ts"),
        (t_newmethod)torch_ts_new,
        (t_method)torch_ts_free,
        sizeof(t_torch_ts),
        CLASS_DEFAULT, 
        A_GIMME, //Argument type: Optional Symbol (defaults to &s_)
        0);

    class_addlist(torch_ts_class, (t_method)torch_ts_forward); //receive a list of input values
    class_addmethod(torch_ts_class, (t_method)torch_ts_device, gensym("device"), A_GIMME, 0); //set the device (cpu, cuda or mps)
    class_addmethod(torch_ts_class, (t_method)torch_ts_model, gensym("load"), A_SYMBOL, 0); //load a model from a file
    class_addmethod(torch_ts_class, (t_method)torch_ts_select_method, gensym("method"), A_GIMME, 0); //set the method to be used on the loaded model
}

#include "m_pd.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include "../utils/include/pd_torch_device_adapter.h"
#include "../utils/include/pd_torch_types.h" // Include the object struct definition
#include "../utils/include/pd_torch_utils.h" // Utility functions
#include "../../../core/include/core_nn_ts_manager.h" // Include the model manager header for loading models, select methods, etc.
#include "../../../core/include/core_dp_sequential.h"
#include "../utils/include/pd_global_state.h" // shared library to manage global state for pd modules and layers
#include "../utils/include/pd_arg_parser.h"



 
static t_class *torch_sequential_class;

typedef struct _torch_sequential {
    t_object x_obj;

    double learning_rate; // learning rate for the optimizer
    int64_t num_epochs; // number of epochs for training
    int64_t random_seed; // random seed for initialization
    std::string optimizer_type; // type of optimizer (adam, sgd, etc.)
    std::string loss_type; // type of loss function (cross_entropy, mse, etc.)
    std::string selected_method; // selected method for the loaded model 
    std::vector<std::string> available_methods; // available methods for the loaded model
    std::vector<std::string> available_attributes; // available attributes for the loaded model
    std::vector<float> loss_history; // history of loss values

    std::string init_method;  // init method
    contorchionist::core::dp_sequential::InitOptions init_opts; // init options

    bool training_mode; // flag to check if the module is in training mode
    bool target; // flag to check if the dataset has a target (if true, the dataset is used for supervised learning, if false, the dataset is used for unsupervised learning)
    bool loaded_model; // flag to check if the model is loaded
    bool verbose; // flag to check if verbose logging is enabled
    bool output_loss; // flag to return the loss after training

    std::vector<std::vector<float>> data; // dataset loaded from a file

    std::vector<float> flat_data; // flattened dataset to create a tensor
    std::vector<float> flat_target; // flattened target to create a tensor
    std::vector<int64_t> full_shape;
    std::vector<int64_t> input_shape; // shape of the input tensor for the sequential module
    std::vector<int64_t> input_shape_model; // shape of the input tensor for the loaded model
    std::vector<int64_t> output_shape_model; // shape of the output tensor for the loaded model

    PdTorchTensors* tensors_struct; // pointer to the tensors struct

    std::shared_ptr<torch::nn::Sequential> container; // container for the sequential module
    std::unique_ptr<torch::jit::script::Module> model; 
    
    torch::optim::Optimizer* optimizer; // optimizer for the module

    torch::Device device; // CPU, CUDA ou MPS device
    at::Tensor last_input, last_output, last_target; // tensors for the last input, output and target
    
    t_symbol *name; // name of the module
    t_canvas*  m_canvas; //canvas
    
    t_atom *out; // output buffer
    t_outlet *x_out1; // outlet for processed tensor
} t_torch_sequential;



//* ------------------------ optimizer settings ------------------------ *//
static void torch_sequential_set_optimizer(t_torch_sequential *x, t_symbol *s, int argc, t_atom *argv) {

    //check if the first argument is a symbol
    if (argc < 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.sequential: Provide optimizer name (adam, sgd, rmsprop) and optionally learning rate.");
        return;
    }

    // Check if the second argument is valid (learning rate)
    if (argv[1].a_w.w_float <= 0 || argv[1].a_w.w_float > 1) { 
        pd_error(x, "torch.sequential: Learning rate must be between 0 and 1.");
        return;
    }

    //set the optimizer type and learning rate
    std::string opt_name = atom_getsymbol(argv)->s_name;
    x->learning_rate = (argc >= 2 && argv[1].a_type == A_FLOAT) ? atom_getfloat(argv + 1) : 1e-3; // if learning rate is not set, use default value

    // free the old optimizer if it exists  
    if (x->optimizer) {
        delete x->optimizer;
        x->optimizer = nullptr;
    }

    // create optimizer
    x->optimizer = contorchionist::core::dp_sequential::create_optimizer(
        opt_name,
        *(x->container->get()),
        x->learning_rate,
        x->optimizer_type
    );

    if (x->optimizer && x->verbose) {
        post("torch.sequential: Optimizer set to '%s' (learning rate = %g)", x->optimizer_type.c_str(), x->learning_rate);
    }
}

//* ------------------------ set loss function ------------------------ *//
static void torch_sequential_set_loss(t_torch_sequential *x, t_symbol *s, int argc, t_atom *argv) {

    //check if the first argument is a symbol
    if (argc != 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.sequential: Provide a loss function name");
        return;
    }

    //set the loss type
    std::string loss_name = atom_getsymbol(argv)->s_name;
    x->loss_type = loss_name;

    // creates a loss function
    x->tensors_struct->loss_function = contorchionist::core::dp_sequential::create_loss_function(loss_name);

    if (x->verbose) {
        post("torch.sequential: Loss function set to '%s'", loss_name.c_str());
    }
}


//* ------------------------ set dataset target true or false ------------------------ *//
static void torch_sequential_target(t_torch_sequential *x, t_floatarg t) {
    if(t < 0 || t > 1) {
        pd_error(x, "torch.sequential: target must be 0 or 1.");
        return;
    }
    x->target = (t != 0);
    if (x->verbose) {
        post("torch.sequential: target set to %s", x->target ? "true" : "false");
    }
}

//* ------------------------ initialize weights ------------------------ *//
static void torch_sequential_init_weights(t_torch_sequential *x, t_symbol *s, int argc, t_atom *argv) {

    // check with the container is not empty
    if (!x->container || x->container->get()->is_empty()) {
        pd_error(x, "torch.sequential: No layers added to the container.");
        return;
    }

    if (argc < 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.sequential: init requires a method (kaiming_uniform, kaiming_normal, xavier_uniform, xavier_normal, uniform, normal, constant).");
        return;
    }
    //set training mode
    x->training_mode = true;
    x->loss_history.clear();

    // set random seed for reproducibility
    contorchionist::core::dp_sequential::set_random_seed(x->random_seed, false);

    std::string method_str = atom_getsymbol(argv)->s_name;
    bool ok = false;
    auto method = contorchionist::core::dp_sequential::parse_init_method(method_str, &ok);
    if (!ok) {
        post("torch.sequential: Unknown init method '%s', using '%s'",
             method_str.c_str(),
             contorchionist::core::dp_sequential::init_method_to_string(method).c_str());
    }

    // set default options
    contorchionist::core::dp_sequential::InitOptions opts;

    // parse for extra parameters:
    switch (method) {
        case contorchionist::core::dp_sequential::InitMethod::KaimingUniform:
        case contorchionist::core::dp_sequential::InitMethod::KaimingNormal: {
            // init kaiming_* [relu|leaky_relu] [a]
            if (argc >= 2 && argv[1].a_type == A_SYMBOL) {
                opts.nonlinearity = atom_getsymbol(argv+1)->s_name;
            }
            if (argc >= 3 && argv[2].a_type == A_FLOAT) {
                opts.a = atom_getfloat(argv+2);
            }
            break;
        }
        case contorchionist::core::dp_sequential::InitMethod::XavierUniform:
        case contorchionist::core::dp_sequential::InitMethod::XavierNormal: {
            // init xavier_* [gain]
            if (argc >= 2 && argv[1].a_type == A_FLOAT) {
                opts.gain = atom_getfloat(argv+1);
            }
            break;
        }
        case contorchionist::core::dp_sequential::InitMethod::Uniform: {
            // init uniform [low] [high]
            if (argc >= 2 && argv[1].a_type == A_FLOAT){
                opts.low  = atom_getfloat(argv+1);
            }
            if (argc >= 3 && argv[2].a_type == A_FLOAT){
                opts.high = atom_getfloat(argv+2);
            }
            break;
        }
        case contorchionist::core::dp_sequential::InitMethod::Normal: {
            // init normal [mean] [std]
            if (argc >= 2 && argv[1].a_type == A_FLOAT) {
                opts.mean = atom_getfloat(argv+1);
            }
            if (argc >= 3 && argv[2].a_type == A_FLOAT) {
                opts.std  = atom_getfloat(argv+2);
            }
            break;
        }
        case contorchionist::core::dp_sequential::InitMethod::Constant: {
            // init constant [value]
            if (argc >= 2 && argv[1].a_type == A_FLOAT) opts.constant = atom_getfloat(argv+1);
            break;
        }
    }

    contorchionist::core::dp_sequential::initialize_sequential_weights(x->container, method, opts);
    if (x->verbose) {
        std::string seed_info = (x->random_seed >= 0) ? 
                " (seed=" + std::to_string(x->random_seed) + ")" : "";
        post("torch.sequential: Initialized weights with '%s'%s", method_str.c_str(), seed_info.c_str());
    }
}


//* ----------------------- sets the device (cpu or cuda) --------------------- *//
static void torch_sequential_device(t_torch_sequential *x, t_symbol *s, int argc, t_atom *argv) {
    // Check if the first argument is a symbol
    if (argc != 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "pdtorch.linear: Please provide a device cpu, cuda, or mps.");
        return;
    }
    // get the device name received
    t_symbol *device_name = atom_getsymbol(&argv[0]);
    std::string device_arg_str = device_name->s_name;

    // set the device
    // x->device = PdTorchUtils::select_device(device_arg_str, (t_pd*)x);
    bool device_flag_present = true;
    pd_parse_and_set_torch_device(
        (t_object*)x,
        x->device,
        device_arg_str,
        x->verbose,
        "torch.sequential",
        device_flag_present
    );
    if (x->verbose) {
        post("torch.sequential: Device set to %s", device_arg_str.c_str());
    }
}


//* -------------------- creates a new module with a name passed as message ---------------------- //
static void module_name(t_torch_sequential *x, t_symbol *s, int argc, t_atom *argv) {
    
    // Check if the first argument is a symbol
    if (argc != 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.sequential: Please provide a valid module name.");
        return;
    }
    // reset the training mode
    x->training_mode = true;
    x->target = false; //set target to false
    x->loss_history.clear();

    // reset parameters
    contorchionist::core::dp_sequential::reset_sequential_parameters(x->container);

    // notify all instances added to this module that it was removed when the module is destroyed
    PDGlobalState::notify_and_clear_registry(PDGlobalState::mha_registry, x->name->s_name);
    PDGlobalState::notify_and_clear_registry(PDGlobalState::activation_registry, x->name->s_name);
    PDGlobalState::notify_and_clear_registry(PDGlobalState::linear_registry, x->name->s_name);
    PDGlobalState::notify_and_clear_registry(PDGlobalState::reshape_registry, x->name->s_name);


    // Save old name
    t_symbol *old_name = x->name;

    // Get the new name
    t_symbol *new_name = atom_getsymbol(argv);

    // Remove old module from registry if it exists and unbind it
    if (old_name && PDGlobalState::module_registry.count(old_name->s_name) > 0) {
        PDGlobalState::module_registry.erase(old_name->s_name);
        pd_unbind((t_pd *)x, old_name);
        post("torch.sequential: Removed previous module '%s'", old_name->s_name);
    }

    // Update to new name
    x->name = new_name;
    x->container = std::make_shared<torch::nn::Sequential>(); // Recreate container

    
    // Check if the new name already exists (rare but safe to check)
    if (PDGlobalState::module_registry.count(x->name->s_name) > 0) {
        pd_error(x, "torch.sequential: Module '%s' already exists!", x->name->s_name);
        return;
    }

    // new bind to the new name 
    pd_bind((t_pd *)x, x->name);

    // Register new module
    PDGlobalState::module_registry[x->name->s_name] = x->container;
    if (x->verbose) {
        post("torch.sequential: Created module '%s'", x->name->s_name);
    }
}



//* ------------------ reset the current module and recreate a new with the same name (it can be used to clear the module without deleting the object) ------------------------- //
static void torch_sequential_clear(t_torch_sequential *x) {

    if (!x->name) {
        pd_error(x, "torch.sequential: No module to clear.");
        return;
    }

    // reset the parameters of all layers in the container
    contorchionist::core::dp_sequential::reset_sequential_parameters(x->container);

    // reset the training mode
    x->training_mode = true;
    x->target = false; //set target to false
    x->loss_history.clear();

    // notify all the instances added to this module that it was removed when the module is cleared
    PDGlobalState::notify_and_clear_registry(PDGlobalState::mha_registry, x->name->s_name);
    PDGlobalState::notify_and_clear_registry(PDGlobalState::activation_registry, x->name->s_name);
    PDGlobalState::notify_and_clear_registry(PDGlobalState::linear_registry, x->name->s_name);
    PDGlobalState::notify_and_clear_registry(PDGlobalState::reshape_registry, x->name->s_name);


    // remove old container from registry
    if (PDGlobalState::module_registry.count(x->name->s_name) > 0) { //if the module exists
        PDGlobalState::module_registry.erase(x->name->s_name); // remove it from the registry
    }

    // recreate an empty container
    x->container = std::make_shared<torch::nn::Sequential>(); // Recreate the container

    // re-register the empty container with the same name
    PDGlobalState::module_registry[x->name->s_name] = x->container;

    if (x->verbose) {
        post("torch.sequential: Cleared module '%s'", x->name->s_name);
    }
}


//* ---------------- print information about the added modules ------------------ //
static void torch_sequential_print(t_torch_sequential *x) {

    auto it = PDGlobalState::module_registry.find(x->name->s_name); //look for the module in the registry
    if (it != PDGlobalState::module_registry.end()) { //if the module exists update the container
        x->container = it->second; //update the container
    }
    // List all registered modules
    if (PDGlobalState::module_registry.empty()) {
        post("torch.sequential: No modules currently registered.");
    } else {
        post("torch.sequential: All registered modules:");
        for (const auto& pair : PDGlobalState::module_registry) {
            post("- %s", pair.first.c_str());
        }
    }
    // Then, info about the current module
    if (!x->name) {
        post("torch.sequential: No current module loaded for detailed info.");
        return;
    }
    post(" ");
    post("torch.sequential: Module '%s':", x->name->s_name);

    // print the layers added
    auto summary = contorchionist::core::dp_sequential::summarize_sequential(
        x->container, x->input_shape
    );

    if (summary.empty()) {
        post("  - (Empty container)");
        return;
    }

    for (size_t i = 0; i < summary.size(); ++i) {
        const auto& li = summary[i];
        if (li.friendly_name == "Linear") {
            long long in_f  = (!li.input_shape.empty() ? li.input_shape.back() : -1);
            long long out_f = (!li.output_shape.empty() ? li.output_shape.back() : -1);
            post("  - Layer %zu: Linear %lld %lld", i, in_f, out_f);
        } else if (li.friendly_name == "Mha") {
            post("  - Layer %zu: Mha %s",
                 i,
                 contorchionist::core::dp_sequential::shape_to_string(li.input_shape).c_str());
        } else {
            post("  - Layer %zu: %s %s",
                 i,
                 li.friendly_name.c_str(),
                 contorchionist::core::dp_sequential::shape_to_string(li.input_shape).c_str());
        }
    }
}


//* --------------------------- receive the input shape of the module ---------------------------------- //
static void torch_sequential_set_input_shape(t_torch_sequential *x, t_symbol *s, int argc, t_atom *argv) {
    //check if the list is empty
    if (argc < 1) {
        pd_error(x, "torch.sequential: input_shape requires at least one value.");
        return;
    }
    //convert input atoms to std::vector<int64_t> (tensor shape must be int64_t)
    std::vector<int64_t> input_shape;
    for (int i = 0; i < argc; ++i) {
        if (argv[i].a_type != A_FLOAT) {
            pd_error(x, "torch.sequential: Only float values are accepted in input_shape.");
            return;
        }
        // store the shape values in shape vector
        input_shape.push_back(static_cast<int64_t>(atom_getfloat(argv + i)));
    }

    // store the input shape received just if the module has only one layer
    if (x->container && x->container->get()->size() == 1) {
        x->input_shape = input_shape;
        // print the input shape
        std::string shape_str;
        for (auto v : input_shape) shape_str += std::to_string(v) + " ";
        if (x->verbose) {
            post("torch.sequential: Input shape set to [%s]", shape_str.c_str());
        }
    }
}



//* ------------------ load the dataset from a file ------------------ */
void load_dataset(t_torch_sequential *x, t_symbol *filename, t_symbol *format) {
    std::vector<std::vector<float>> data; // 2D vector to store the dataset

    if (!filename) {
        pd_error(x, "torch.sequential: No filename provided to load_dataset.");
        return;
    }
    char dirresult[MAXPDSTRING]; // buffer to store the directory address
    char *nameresult; // pointer to store the name of the file
    int fd; // file descriptor 
    std::string fname(filename->s_name); // convert the filename to std::string
    const char *canvas_dir = canvas_getdir(x->m_canvas)->s_name; // get the cirrent directory of the Pure Data canvas

    fd = open_via_path(canvas_dir, fname.c_str(), "", dirresult, &nameresult, MAXPDSTRING, 1); // look for the file in the current directory of the canvas

    if (fd >= 0) { // if the file is found
        sys_close(fd); // close the file descriptor (just need the address of the file)
        char fullpath[MAXPDSTRING]; // buffer to store the full path of the file
        snprintf(fullpath, MAXPDSTRING, "%s/%s", dirresult, nameresult); // create the full path of the file
        char normalized[MAXPDSTRING]; // buffer to store the normalized path of the file (without special characters)
        sys_unbashfilename(fullpath, normalized); // normalize the path (remove special characters)

        std::ifstream infile(normalized); // open the file for reading
        if (!infile.is_open()) { // sends an error if the file cannot be opened and return
            pd_error(x, "Failed to open the file '%s' for reading.", normalized);
            return;
        }

        // read the file line by line and store the data in a 2D vector
        
        std::string line; // string to store each line read from the file
        while (std::getline(infile, line)) { // read each line from the file
            // removes the trailing semicolon if it exists
            if (!line.empty() && line.back() == ';') //removes the last character if it is a semicolon
                line.pop_back();

            std::istringstream iss(line); // create a string stream to parse the line
            std::vector<float> row; // vector to store the values of the line
            std::string token; // string to store each token read from the line
            while (iss >> token) { // read each token from the line
                try {
                    row.push_back(std::stof(token)); // convert the token to float and store it in the row vector
                } catch (...) {
                    // ignores invalid tokens
                }
            } // if the row is not empty, store it in the data vector
            if (!row.empty()) data.push_back(row);
        }
        infile.close(); // close the file
        //sends an error if the dataset is empty and return
        if (data.empty()) {
            pd_error(x, "Failed to load dataset '%s' (empty file or invalid format).", fname.c_str());
            return;
        }
        x->data = data; // store the dataset in  x->data
        if (x->verbose) {
            post("torch.sequential: Dataset loaded successfully '%s' (%zu lines)", fname.c_str(), data.size());
        }
    } else {
        pd_error(x, "File not found '%s'", fname.c_str());
    }

    // if the input shape is set and the dataset file is loaded, create a tensor with the input shape and fill it with the dataset loaded
    if (!x->input_shape.empty() && x->data.size() > 0) { 
        int64_t expected_size = 1; // expected size of the dataset
        for (auto v : x->input_shape) expected_size *= v; // calculate the expected size of the dataset (multiply all the input shape dimensions)

        // ---- deals with inputs and targets ---- //
        std::vector<std::vector<float>> inputs, targets; // vectors to store the inputs and targets
        // iterate through the dataset and separate the inputs and targets if traget=true
        for (const auto& row : data) {
            // if the dataset has a target
            if (x->target) { 
                if ((int64_t)row.size() < expected_size) { // compare the size of each line with the expected size
                    pd_error(x, "Line has less columns than input_shape.");
                    return;
                }
                // extras columns are considered targets
                std::vector<float> input_row(row.begin(), row.begin() + expected_size);
                std::vector<float> target_row(row.begin() + expected_size, row.end());
                inputs.push_back(input_row);
                targets.push_back(target_row);
            // if the dataset has no target
            } else if (!x->target) { 
                // all coluns are considered inputs
                if ((int64_t)row.size() != expected_size) { // compare the size of each line with the expected size, if doesn't match, send an error
                    pd_error(x, "Line has %zu elements, expected %lld (input_shape).", row.size(), expected_size);
                    return;
                }
                inputs.push_back(row);
            }
        }
        int64_t num_samples = inputs.size(); // number of the dataset training examples (number of lines)
        x->full_shape.clear(); // clear the full shape vector
        x->full_shape.push_back(num_samples); // set the first dimension of the full shape to the number of training examples
        x->full_shape.insert(x->full_shape.end(), x->input_shape.begin(), x->input_shape.end()); // append the input shape to the full shape
        x->flat_data.clear(); // clear the flat_data vector and fill it with the dataset loaded
        x->flat_target.clear(); // clear the flat_target vector and fill it with the targets loaded

        //flatten the dataset
        for (const auto& row : inputs)  // iterate through the dataset
            x->flat_data.insert(x->flat_data.end(), row.begin(), row.end()); //flatten the whole dataset
        //flatten the targets
        for (const auto& row : targets) // iterate through the targets
            x->flat_target.insert(x->flat_target.end(), row.begin(), row.end()); //flatten the targets

       
        int64_t total_shape = 1; // variable to store the total shape of the dataset
        for (auto v : x->full_shape) { // calculate the total shape of the dataset
            if (v <= 0) {
                pd_error(x, "torch.sequential: Shape not valid (%lld). No shape can be <= 0.", v);
                return;
            }
            total_shape *= v;
        }
        // check if the flat_data size is equal to the total shape or if the flat_data is empty
        if (x->flat_data.size() != total_shape ||x->flat_data.empty()) {
            pd_error(x, "torch.sequential: flat_data size (%zu) != total shape (%lld or is empty", x->flat_data.size(), total_shape);
            return;
        }
        // check if the flat_target is empty
        if (x->target && x->flat_target.empty()) {
            pd_error(x, "torch.sequential: Flat_target is empty.");
            return;
        }
        //check if the flat_data has valid values (not NaN or Inf)
        for (size_t i = 0; i < x->flat_data.size(); ++i) {
            if (std::isnan(x->flat_data[i]) || std::isinf(x->flat_data[i])) {
                pd_error(x, "torch.sequential: Value not valid (NaN or Inf) in flat_data position %zu", i);
                return;
            }
        }
        //check if the flat_target has valid values (not NaN or Inf)
        for (size_t i = 0; i < x->flat_target.size(); ++i) {
            if (std::isnan(x->flat_target[i]) || std::isinf(x->flat_target[i])) {
                pd_error(x, "torch.sequential: Value not valid (NaN or Inf) in flat_target position %zu", i);
                return;
            }
        }
        // create a tensor with the full shape and fill it with the dataset loaded
        try {
            x->tensors_struct->dataset_tensor = torch::from_blob(
                x->flat_data.data(),
                torch::IntArrayRef(x->full_shape),
                torch::TensorOptions().dtype(torch::kFloat)
            ).clone(); // clone the tensor to avoid using the same memory
        } catch (const c10::Error& e) {
            pd_error(x, "torch.sequential: Failed to create dataset tensor: %s", e.what());
        }
        // check if the input tensor was created successfully
        if (!x->tensors_struct->dataset_tensor.defined()) {
                pd_error(x, "torch.sequential: Failed to create dataset tensor.");
                return;
            }
        // create a target tensor if the target is true
        if (x->target && !targets.empty()){
            int64_t target_dim = targets[0].size(); // get the target dimension
            std::vector<int64_t> target_shape = {num_samples, target_dim}; // create the target shape
            try{
                x->tensors_struct->target_tensor = torch::from_blob( // create the target tensor 
                    x->flat_target.data(), // fill it with the targets loaded
                    torch::IntArrayRef(target_shape), 
                    torch::TensorOptions().dtype(torch::kFloat)
                ).clone();
            } catch (const c10::Error& e) {
                pd_error(x, "torch.sequential: Failed to create dataset tensor: %s", e.what());
            }
        }
        // check the compatibility of the loss function with the target tensor
        if (x->target && x->tensors_struct->target_tensor.defined() && !x->loss_type.empty()) {
            auto validation_result = contorchionist::core::dp_sequential::check_loss_target_compatibility(
                x->loss_type,
                x->tensors_struct->target_tensor
            );
            if (!validation_result.is_valid) {
                pd_error(x, "torch.sequential: %s", validation_result.error_message.c_str());
                return;
            }
        } else if (!x->target) {
            x->tensors_struct->target_tensor = at::Tensor(); // empty tensor target for unsupervised learning
            if (x->verbose) {
                post("torch.sequential: target tensor not created (unsupervised learning).");
            }
        }
        // check the created input tensor shape
        if (x->tensors_struct->dataset_tensor.defined()) {
            std::ostringstream oss;
            oss << "torch.sequential: Dataset tensor created with shape [";
            for (auto s : x->tensors_struct->dataset_tensor.sizes()) oss << s << " ";
            oss << "]";
            post("%s", oss.str().c_str());
        } else {
            x->tensors_struct->dataset_tensor = at::Tensor();
            pd_error(x, "torch.sequential: input shape not set. Dataset tensor not created.");
        }
        // check the created target tensor shape
        if (x->target && x->tensors_struct->target_tensor.defined()) {
            std::ostringstream oss;
            oss << "torch.sequential: Target tensor created with shape [";
            for (auto s : x->tensors_struct->target_tensor.sizes()) oss << s << " ";
            oss << "]";
            post("%s", oss.str().c_str());
        } else if (x->target && !x->tensors_struct->target_tensor.defined()) {
            x->tensors_struct->target_tensor = at::Tensor();
            pd_error(x, "torch.sequential: target tensor not created.");
        }
    }
}



// * ------------------ forward the input tensor through the module (model trained ) ------------------ //
static void torch_sequential_forward(t_torch_sequential *x, t_symbol *s, int argc, t_atom *argv) {
    // Check if the module is in training mode
    if(x->training_mode) {
        pd_error(x, "torch.sequential: Module is in training mode. Use 'train' to run the training.");
        return;
    }
    
    // Check if the input is a list of floats
    if (argc == 0) {
        pd_error(x, "torch.sequential: No input list received.");
        return;
    }
   
   // if the container has layers
    if (x->container && x->container->get()->size() > 0) { // 

        // Check if the module is registered
        if (!x->container || x->container->get()->size() == 0) {
            pd_error(x, "torch.sequential: No layers in module '%s'.", x->name->s_name);
            return;
        }
        // Check if the input shape is set
        if (x->input_shape.size() == 0) {
            pd_error(x, "torch.sequential: input_shape not set or incomplete.");
            return;
        }
        // calculate the expected size of the input tensor
        int64_t expected_size = 1;
        for (auto v : x->input_shape){ 
            expected_size *= v; // multiply all the input shape dimensions to get the expected size
        }

        // if the input size is less than the expected size, send an error
        if (argc < expected_size) {
            pd_error(x, "torch.sequential: Input has %d elements, expected at least %lld (input_shape).", argc, expected_size);
            return;
        }

        // convert input atoms to std::vector<float>
        std::vector<float> values;
        for (int i = 0; i < expected_size; ++i) {
            if (argv[i].a_type != A_FLOAT) {
                pd_error(x, "torch.sequential: Only float values are accepted in list.");
                return;
            }
            //copy the input received to the vector (ignoring targets with it exists)
            values.push_back(atom_getfloat(argv + i));
        }

        // create a tensor with the input shape and fill it with the input received
        at::Tensor input = torch::from_blob(
            values.data(),
            torch::IntArrayRef(x->input_shape),
            torch::TensorOptions().dtype(torch::kFloat)
        ).clone();

        // forward the input tensor through the sequential container
        at::Tensor output;
        try {
            output = contorchionist::core::dp_sequential::SequentialProcessor(
                input,           // input tensor
                x->container,    // sequential container  
                x->device       // target device
            );
        } catch (const std::runtime_error& e) {
            pd_error(x, "torch.sequential: %s", e.what());
            return;
        } catch (const c10::Error& e) {
            pd_error(x, "torch.sequential: PyTorch error: %s", e.what());
            return;
        }

        try{
            // Flatten and move the output tensor to CPU
            output = output.view({-1}).to(torch::kCPU);
            int64_t out_size = output.size(0);

            // convert output tensor to atom array
            t_atom *out_atoms = (t_atom *)getbytes(sizeof(t_atom) * out_size);
            for (int64_t i = 0; i < out_size; ++i) {
                SETFLOAT(&out_atoms[i], output[i].item<float>());
            }
            if (x->output_loss) {
                // send the forward output
                outlet_anything(x->x_out1, gensym("forward"), out_size, out_atoms);
                freebytes(out_atoms, sizeof(t_atom) * out_size);
            } else {
                outlet_anything(x->x_out1, &s_list, out_size, out_atoms);
                   freebytes(out_atoms, sizeof(t_atom) * out_size);
            }
        } catch (const c10::Error& e) {
            pd_error(x, "torch.sequential: Error converting output: %s", e.what());
            return;
        }
    } else {
        pd_error(x, "torch.sequential: No layers in module '%s'.", x->name->s_name);
        return;
    }
}


//* ------------------ set the tensor to use ------------------ */
static void torch_sequential_settensor(t_torch_sequential *x, t_symbol *tensor_name) {
    if (!tensor_name || !tensor_name->s_name) {
        pd_error(x, "torch.sequential: Please provide a valid tensor name.");
        return;
    }

    if (x->tensors_struct->dataset_tensor.defined()) {
        post("torch.sequential: Replacing existing dataset tensor.");
    }

    auto it = PDGlobalState::ls2tensor_registry.find(tensor_name->s_name);
    if (it == PDGlobalState::ls2tensor_registry.end()) {
        pd_error(x, "torch.sequential: Tensor object '%s' not found.", tensor_name->s_name);
        return;
    }

    t_torch_ls2tensor* source_tensor_obj = it->second;
    if (!source_tensor_obj->tensors_struct->dataset_tensor.defined()) {
        pd_error(x, "torch.sequential: Source tensor in '%s' is not defined. Fill it with data first.", tensor_name->s_name);
        return;
    }

    if (!source_tensor_obj) {
        pd_error(x, "torch.sequential: Tensor object '%s' is null.", tensor_name->s_name);
        return;
    }

    if (!source_tensor_obj->tensors_struct) {
        pd_error(x, "torch.sequential: Source tensor '%s' has no tensor structure.", tensor_name->s_name);
        return;
    }

    try{
        // Copy tensors from source object
        x->tensors_struct->dataset_tensor = source_tensor_obj->tensors_struct->dataset_tensor.clone();
        x->target = source_tensor_obj->has_target;

        if (x->target) {
            if (!source_tensor_obj->tensors_struct->target_tensor.defined()) {
                pd_error(x, "torch.sequential: Source '%s' is set to has_target, but its target tensor is not defined.", tensor_name->s_name);
                return;
            }
            x->tensors_struct->target_tensor = source_tensor_obj->tensors_struct->target_tensor.clone();
        } else {
            x->tensors_struct->target_tensor = at::Tensor();
        }
    } catch (const c10::Error& e) {
        pd_error(x, "torch.sequential: Failed to clone tensors from '%s': %s", tensor_name->s_name, e.what());
        return;
    }

    try{
        // Update shape information from the new tensor
        x->full_shape.assign(
            x->tensors_struct->dataset_tensor.sizes().begin(),
            x->tensors_struct->dataset_tensor.sizes().end()
        );
        if (x->full_shape.size() > 1) {
            x->input_shape.assign(x->full_shape.begin() + 1, x->full_shape.end());
        } else {
            x->input_shape.clear();
        } 
    } catch (const std::exception& e) {
        pd_error(x, "torch.sequential: Failed to update shapes from '%s': %s", tensor_name->s_name, e.what());
        return;
    }
    
    // check compatibility of the loss function with the target tensor
    if (x->target && x->tensors_struct->target_tensor.defined() && !x->loss_type.empty()) {
        try {
            auto validation_result = contorchionist::core::dp_sequential::check_loss_target_compatibility(
                x->loss_type,
                x->tensors_struct->target_tensor
            );
            if (!validation_result.is_valid) {
                pd_error(x, "torch.sequential: %s", validation_result.error_message.c_str());
                return;
            }
        } catch (const std::exception& e) {
            pd_error(x, "torch.sequential: Failed to check loss-target compatibility: %s", e.what());
            return;
        }
    }
    if (x->verbose) {
    post("torch.sequential: Dataset set from tensor object '%s'. Shape: [%s], Target: %s", 
         tensor_name->s_name,
         contorchionist::core::dp_sequential::shape_to_string(x->full_shape).c_str(),
         x->target ? "yes" : "no");
    }
}

// * ------------------ train the model ------------------ */
static void torch_sequential_train(t_torch_sequential *x, t_symbol *s, int argc, t_atom *argv) {

    // check if the second argument is a number
    if (argc >= 1 && argv[0].a_type == A_FLOAT) {
        int64_t arg_epochs = static_cast<int64_t>(atom_getfloat(argv));
        if (arg_epochs > 0) {
            x->num_epochs = arg_epochs;
        } else {
            pd_error(x, "torch.sequential: Number of epochs must be a positive integer");
            return;
        }
    } else {
        pd_error(x, "torch.sequential: Number of epochs must be a positive integer. Set to default (100).");
        x->num_epochs = 100; // default value
    }

    //check if the module is registered
    if (!x->container || x->container->get()->size() == 0) {
        pd_error(x, "torch.sequential: No layers in module '%s'.", x->name->s_name);
        return;
    }

    // create optimizer
    if (!x->optimizer) {
        try {
            x->optimizer = contorchionist::core::dp_sequential::create_optimizer(
                x->optimizer_type,
                *(x->container->get()),
                x->learning_rate,
                x->optimizer_type
            );
        } catch (const std::exception& e) {
            pd_error(x, "torch.sequential: Failed to create optimizer: %s", e.what());
            return;
        }
        
        if (!x->optimizer) {
            pd_error(x, "torch.sequential: Optimizer creation failed");
            return;
        }
    } else if (x->verbose) {
        post("torch.sequential: Using default optimizer '%s'", x->optimizer_type.c_str());
    }

    // initialize weights
    if (!x->init_method.empty()) {

        // set random seed for reproducibility
        contorchionist::core::dp_sequential::set_random_seed(x->random_seed, false);

        bool ok = false;
        auto method_enum = contorchionist::core::dp_sequential::parse_init_method(x->init_method, &ok);
        if (!ok) {
            if (x->verbose) {
                    post("torch.sequential: Unknown init method '%s', using default", x->init_method.c_str());
                }
            //fallback
            method_enum = contorchionist::core::dp_sequential::InitMethod::Uniform;
        }
        try {
            contorchionist::core::dp_sequential::initialize_sequential_weights(
                x->container, 
                method_enum, 
                x->init_opts
            );
            if (x->verbose) {
                std::string seed_info = (x->random_seed >= 0) ? 
                " (seed=" + std::to_string(x->random_seed) + ")" : "";
                post("torch.sequential: Weights initialized with '%s'%s", x->init_method.c_str(), seed_info.c_str());
            }
        } catch (const std::exception& e) {
            pd_error(x, "torch.sequential: Weight initialization failed: %s", e.what());
            return;
        }
    } else if (x->verbose) {
        post("torch.sequential: No weight initialization method configured, using default method");
    }

    // check if the input shape is set
    if (x->input_shape.size() == 0) {
        pd_error(x, "torch.sequential: input_shape not set or incomplete.");
        return;
    }
    // check if the dataset tensor is defined
    if (!x->tensors_struct->dataset_tensor.defined()) {
        pd_error(x, "torch.sequential: No dataset tensor loaded. Use [load( to load a dataset.");
        return;
    }

     // check if the module is already trained
     if (!x->training_mode) {
        pd_error(x, "torch.sequential: Module is already trained.");
        return;
    }

    // move the container, dataset and target tensors to the selected device
    x->container->get()->to(x->device);
    x->tensors_struct->dataset_tensor = x->tensors_struct->dataset_tensor.to(x->device);
    if (x->target && x->tensors_struct->target_tensor.defined()) {
        x->tensors_struct->target_tensor = x->tensors_struct->target_tensor.to(x->device);
    }

    //calculate the expected size of the input tensor
    int64_t expected_size = 1;
    for (auto v : x->input_shape){ 
        expected_size *= v;
    }

    // start the training time
    auto start = std::chrono::high_resolution_clock::now();

    // clean loss history
    if (x->output_loss) {
        x->loss_history.clear();
    }

    // Progress callback para Pure Data
    auto progress_callback = [x](int64_t epoch, float loss) {
        if (x->verbose) {
            post("torch.sequential: Epoch %lld, Loss: %f", epoch, loss);
        }
        if (x->output_loss) {
            // accumulate loss values
            x->loss_history.push_back(loss);
        }
    };

    // training loop
    contorchionist::core::dp_sequential::TrainingResult result;
    try {
        result = contorchionist::core::dp_sequential::SequentialTrainer(
            x->container,                           // container 
            x->tensors_struct->dataset_tensor,      // dataset 
            x->tensors_struct->target_tensor,       // target 
            x->tensors_struct->loss_function,       // loss function
            x->optimizer,                           // optimizer
            x->num_epochs,                          // epochs
            x->device,                              // device
            x->target,                              // use_target flag
            progress_callback                       // progress callback para PD
        );
    } catch (const std::exception& e) {
        pd_error(x, "torch.sequential: Training failed: %s", e.what());
        return;
    }

    if (!result.success) {
        pd_error(x, "torch.sequential: Training failed: %s", result.error_message.c_str());
        return;
    }

    // end the training time measurement
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();

    if (x->output_loss && !x->loss_history.empty()) {
        size_t actual_epochs = x->loss_history.size();
        // send the loss history to the outlet
        t_atom *loss_atoms = (t_atom *)getbytes(sizeof(t_atom) * actual_epochs);
        // fill the loss atoms
        for (size_t i = 0; i < actual_epochs; ++i) {
            SETFLOAT(&loss_atoms[i], x->loss_history[i]);
        }
        outlet_anything(x->x_out1, gensym("loss"), actual_epochs, loss_atoms);
        freebytes(loss_atoms, sizeof(t_atom) * actual_epochs);
    }

    // sets the training mode to false after training (ready to receive input to inference)
    x->training_mode = false;
    x->loaded_model = false; // model is not loaded
    post("torch.sequential: Training done. Training time: %.3f ms, Epochs: %lld, Final Loss: %f", elapsed_sec*1000, x->num_epochs, result.final_loss);
}


//* -------------------------- constructor -------------------------- //
void *torch_sequential_new(t_symbol *s, int argc, t_atom *argv) {
    t_torch_sequential *x = (t_torch_sequential *)pd_new(torch_sequential_class);

    // first argument must be a symbol (module name)
    if (argc < 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.sequential: Provide a module name as first argument.");
        return NULL;
    }
    // set the name of the module
    x->name = atom_getsymbol(argv);
    
    // Check if a module with the same name already exists
    //Access module_registry always using PDGlobalState::module_registry (module_registry is inside the PDGlobalState namespace)
    if (PDGlobalState::module_registry.count(x->name->s_name) > 0) { 
        pd_error(x, "torch.sequential: Module '%s' already exists!", x->name->s_name);
        return NULL;
    }

    // default parameters
    x->optimizer = nullptr; // optimizer for the module
    x->training_mode = true; // training mode disabled by default
    x->loaded_model = false; // model not loaded by default

    // Initialize the container
    x->container = std::make_shared<torch::nn::Sequential>(); 

    // create a new module torch::nn::Sequential and register it in a global map
    PDGlobalState::module_registry[x->name->s_name] = x->container; 

    // symbolically binds the module name (x->name) to this instance of torch.sequential.
    pd_bind((t_pd *)x, x->name);

    // create the tensors struct
    x->tensors_struct = new PdTorchTensors(); 
    // get the current canvas
    x->m_canvas = canvas_getcurrent(); 

    // ------------- flags parsing ------------- //
    pd_utils::ArgParser parser(argc-1, argv+1, (t_object*)x);
    // verbose flag
    x->verbose = parser.has_flag("verbose v");
    //optimizer
    x->optimizer_type = parser.get_string("optimizer opt", "sgd");
    // learning rate
    x->learning_rate = parser.get_float("lr learning_rate", 1e-3);
    // loss function
    x->loss_type = parser.get_string("loss", "mse");
    // target
    x->target = parser.get_bool("target tar", false);
    // random state. (-1 means random)
    x->random_seed = parser.get_float("seed randomstate rs", -1);
    // output loss flag
    x->output_loss = parser.get_bool("ol outloss", false);

    // weight initialization
    std::string init_method;
    contorchionist::core::dp_sequential::InitOptions init_opts;

    if (parser.has_flag("init winit weight_init")) {
        init_method = parser.get_string("init winit weight_init", "uniform");
        // specific options for each weight initialization
        if (init_method == "hu" || init_method == "hn" || init_method == "heuniform" || init_method == "henormal") {
            init_opts.nonlinearity = parser.get_string("nonlinearity nl", "relu");
            init_opts.a = parser.get_float("negative_slope ns", 0.01);

        } else if (init_method == "xu" || init_method == "xn" || init_method == "xavieruniform" || init_method == "xaviernormal") {
            init_opts.gain = parser.get_float("gain g", 1.0);

        } else if (init_method == "uniform" || init_method == "u") {
            init_opts.low = parser.get_float("low l", -0.1);
            init_opts.high = parser.get_float("high h", 0.1);

        } else if (init_method == "normal" || init_method == "n") {
            init_opts.mean = parser.get_float("mean m", 0.0);
            init_opts.std = parser.get_float("std s", 0.01);

        } else if (init_method == "constant" || init_method == "const") {
            init_opts.constant = parser.get_float("constant c", 0.0);
        }
        
        if (x->verbose && !init_method.empty()) {
            post("torch.sequential: Weight initialization configured: '%s'", init_method.c_str());
        }
    }

    // device 
    bool device_flag_present = parser.has_flag("device d");
    std::string device_arg_str = parser.get_string("device d", "cpu");
    // Get device from string
    auto device_result = get_device_from_string(device_arg_str);
    // x->device = device_result.first;
    bool device_parse_success = device_result.second;
    // set the device
    pd_parse_and_set_torch_device(
        (t_object*)x,
        x->device,
        device_arg_str,
        x->verbose,
        "torch.sequential",
        device_flag_present
    );
    
    // set the loss function
    x->tensors_struct->loss_function = contorchionist::core::dp_sequential::create_loss_function(x->loss_type);

    // set and save the method and options for initialization before the training
    x->init_method = init_method;
    x->init_opts = init_opts;

    // outlet
    x->x_out1 = outlet_new(&x->x_obj, &s_anything);

    if (x->verbose) {
        std::string seed_info = (x->random_seed >= 0) ? 
            ", seed=" + std::to_string(x->random_seed) : ", random seed";
        std::string init_info = !init_method.empty() ? 
            ", weight init=" + init_method : "";
        post("torch.sequential: Module '%s' created with optimizer '%s', learning rate=%f, loss function '%s'%s%s", 
             x->name->s_name, 
             x->optimizer_type.c_str(), 
             x->learning_rate, 
             x->loss_type.c_str(),
             seed_info.c_str(),
             init_info.c_str());
     }

    return (void *)x;
}

//* ------------------------- destructor ------------------------- //
void torch_sequential_free(t_torch_sequential *x) {
    
    //notify all instances added to this module that it was removed when the object is destroyed
    PDGlobalState::notify_and_clear_registry(PDGlobalState::mha_registry, x->name->s_name);
    PDGlobalState::notify_and_clear_registry(PDGlobalState::activation_registry, x->name->s_name);
    PDGlobalState::notify_and_clear_registry(PDGlobalState::linear_registry, x->name->s_name);

    
    // Removes the symbolic association between the module name and this instance of torch.sequential created by pd_bind
    pd_unbind((t_pd *)x, x->name);


    // erase the module from the global map
    PDGlobalState::module_registry.erase(x->name->s_name);

    // free the optimizer if it exists 
    if (x->optimizer) {
        delete x->optimizer;
        x->optimizer = nullptr;
    }

    x->loss_history.clear();

    delete x->tensors_struct; // free the tensors struct
    x->tensors_struct = nullptr;
    // free the output buffer and outlet
    freebytes(x->out, 1 * sizeof(t_atom));
    outlet_free(x->x_out1);
}

//* -------------------------- setup function -------------------------- //
extern "C" void setup_torch0x2esequential(void) {
    torch_sequential_class = class_new(
        gensym("torch.sequential"),
        (t_newmethod)torch_sequential_new,
        (t_method)torch_sequential_free,
        sizeof(t_torch_sequential),
        CLASS_DEFAULT, 
        A_GIMME, //Argument type: Optional Symbol (defaults to &s_)
        0);

        // needed to other objects find the class and send messages to torch.sequential instances using pd_findbyclass and pd_typedmess
        PDGlobalState::torch_sequential_class = torch_sequential_class; 

    class_addlist(torch_sequential_class, (t_method)torch_sequential_forward); //receive a list of input values
    class_addmethod(torch_sequential_class, (t_method)torch_sequential_device, gensym("device"), A_GIMME, 0); //set the device (cpu, cuda or mps)
    class_addmethod(torch_sequential_class, (t_method)torch_sequential_train, gensym("train"), A_GIMME); // train the sequential module 
    class_addmethod(torch_sequential_class, (t_method)module_name, gensym("set"), A_GIMME, 0); //set and create a module name
    class_addmethod(torch_sequential_class, (t_method)torch_sequential_print, gensym("info"), A_GIMME, 0); //list all activated modules
    class_addmethod(torch_sequential_class, (t_method)torch_sequential_clear, gensym("clear"), A_GIMME, 0); //clear the current module
    class_addmethod(torch_sequential_class, (t_method)torch_sequential_set_input_shape, gensym("input_shape"), A_GIMME, 0); //set the input shape
    class_addmethod(torch_sequential_class, (t_method)torch_sequential_set_optimizer, gensym("optimizer"), A_GIMME, 0); //set the optimizer parameters
    class_addmethod(torch_sequential_class, (t_method)load_dataset, gensym("load"), A_SYMBOL, 0); //load a dataset from a file
    class_addmethod(torch_sequential_class, (t_method)torch_sequential_target, gensym("target"), A_FLOAT, 0); //set the target to true or false
    class_addmethod(torch_sequential_class, (t_method)torch_sequential_set_loss, gensym("loss"), A_GIMME, 0); //set the loss function
    class_addmethod(torch_sequential_class, (t_method)torch_sequential_init_weights, gensym("init"), A_GIMME, 0); // initialize weights
    class_addmethod(torch_sequential_class, (t_method)torch_sequential_settensor, gensym("dataset"), A_SYMBOL, 0); // set the dataset tensor to use
}

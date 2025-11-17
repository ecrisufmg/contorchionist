#include "m_pd.h"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <functional>
#include "../../../core/include/core_dp_mha.h"
#include "../utils/include/pd_torch_device_adapter.h"
#include "../utils/include/pd_torch_types.h" // Include the object struct definition
#include "../utils/include/pd_torch_utils.h" // Utility functions
#include "../utils/include/pd_global_state.h" // shared library to manage global state for pd modules and layers
#include "../utils/include/pd_arg_parser.h"




static t_class *torch_mha_class;


//* ----------------------------- manages the modification of MHA module parameters ----------------------------- *//
static void torch_mha_rebuild(t_torch_mha *x) {

    // check if embed_dim and num_heads are valid (embed must be divisible by heads)
    if (x->embed_dim % x->num_heads != 0) {
        pd_error(x, "torch.mha: embed_dim (%d) must be divisible by heads (%d).", x->embed_dim, x->num_heads);
        return;
    }
    // free module
    x->mha = nullptr; // <--- important!

    x->mha_wrapper = nullptr; // <--- important!

    // create MultiheadAttention and SelfAttentionWrapper
    auto r = contorchionist::core::dp_mha::create_mha_layer(x->embed_dim, x->num_heads, x->bias, x->add_zero_attn, x->dropout, x->device);
    if (!r.success) {
        pd_error(x, "torch.mha: %s", r.error_message.c_str());
        return;
    }

    // assign modules
    x->mha = r.mha;
    x->mha_wrapper = r.wrapper;

    if(x->verbose){

        post("torch.mha: batch_size=%d, embed_dim=%d, heads=%d, bias=%s, add_zero_attn=%s, dropout=%f, device=%s",
            x->batch_size,
            x->embed_dim,
            x->num_heads,
            x->bias ? "true" : "false",
            x->add_zero_attn ? "true" : "false",
            x->dropout,
            x->device.is_cuda() ? "cuda" : "cpu");
    }
}



//* -------------------------- number of heads -------------------------- *//
static void torch_mha_heads(t_torch_mha *x, t_floatarg h) {
    int heads = (int)h;
    if (heads <= 0) {
        pd_error(x, "torch.mha: Number of heads must be positive integer.");
        return;
    }
    x->num_heads = heads;
    torch_mha_rebuild(x);

    if (x->verbose) {
        post("torch.mha: Heads = %d", x->num_heads);
    }
}


//* ------------------------- embed_dim ------------------------- *//
static void torch_mha_embed_dim(t_torch_mha *x, t_floatarg e) {
    int embed = (int)e;
    if (embed <= 0) {
        pd_error(x, "torch.mha: Embed dim must be positive integer.");
        return;
    }
    x->embed_dim = embed;
    torch_mha_rebuild(x);

    if (x->verbose) {
        post("torch.mha: Embed dim = %d", x->embed_dim);
    }
}

//* ------------------------- seq_length ------------------------- *//
static void torch_mha_seqlength(t_torch_mha *x, t_floatarg seq) {
    int seql = (int)seq;
    if (seql <= 0) {
        pd_error(x, "torch.mha: Seq_length must be positive integer.");
        return;
    }
    x->seq_len = seql;
    if (x->verbose) {
        post("torch.mha: seq_length = %d", x->seq_len);
    }
}


//* --------------------- enables/disables bias --------------------- *//
static void torch_mha_bias(t_torch_mha *x, t_floatarg b) {
    if(b < 0 || b > 1) {
        pd_error(x, "torch.mha: bias must be 0 or 1.");
        return;
    }
    x->bias = (b != 0);
    torch_mha_rebuild(x);
    if (x->verbose) {
        post("torch.mha: bias = %s", x->bias ? "true" : "false");
    }
}


//* -------------------- enables/disables add_zero_attn --------------------- *//
static void torch_mha_add_zero_attn(t_torch_mha *x, t_floatarg a) {
    x->add_zero_attn = (a != 0);
    torch_mha_rebuild(x);
    if (x->verbose) {
        post("torch.mha: add_zero_attn = %s", x->add_zero_attn ? "true" : "false");
    }
}


//* -------------------- batch_size --------------------- *//
static void torch_mha_batch_size(t_torch_mha *x, t_floatarg b) {
    int batch = (int)b;
    if (batch <= 0) {
        pd_error(x, "torch.mha: batch_size must be positive integer.");
        return;
    }
    x->batch_size = batch;
    if (x->verbose) {
        post("torch.mha: batch_size = %d", x->batch_size);
    }
}



//* ----------------------- sets the device (cpu or cuda) --------------------- *//
static void torch_mha_device(t_torch_mha *x, t_symbol *s, int argc, t_atom *argv) {

    // Check if the first argument is a symbol
    if (argc != 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.mha: Please provide a device (cpu, cuda or mps).");
        return;
    }

    std::string dev = atom_getsymbol(argv)->s_name;
    pd_parse_and_set_torch_device(
        (t_object*)x,
        x->device,
        dev,
        x->verbose,
        "torch.mha",
        true
    );
    torch_mha_rebuild(x); // rebuild the module with the new device
}


//* ------------------------- sends the input shape to an instance of torch.sequential ----------------- *//
void torch_mha_send_input_shape(t_torch_mha *x, t_symbol *module_name) {
    //create a t_atom array to store the input shape
    t_atom shape[3];
    // set the input tensor shape (seq_len: -1)
    SETFLOAT(&shape[0], x->seq_len); // seq_len
    SETFLOAT(&shape[1], x->batch_size); // batch_size
    SETFLOAT(&shape[2], x->embed_dim);     // embed_dim

    /*
    Find a torch.sequential module instance by the module name associated with it and global class pointer (PDGlobalState::pdtorch_sequential_class). 
    This allows torch.mha to send messages directly to a specific instance of torch.sequential. pdtorch_sequential_class need to be defined in the header file
    */
    t_pd *target = (t_pd *)pd_findbyclass(module_name, PDGlobalState::torch_sequential_class);

    // look for the correct instance of torch.sequential and send the input shape
    if (target) {
        pd_typedmess(target, gensym("input_shape"), 3, shape);

        post("torch.mha: Input shape [%d %d %d] sent to module '%s'",
             (int)atom_getfloat(&shape[0]),
             (int)atom_getfloat(&shape[1]),
             (int)atom_getfloat(&shape[2]),
             module_name->s_name);
    } else {
        pd_error(x, "torch.mha: Could not find module '%s' to send input shape.", module_name->s_name);
    }
}


//* ----------------------- multi-head attention forward on the input tensor sent by the inlet --------------------- *//
static void torch_mha_forward(t_torch_mha *x, t_symbol *s, int argc, t_atom *argv) {
    if (argc == 0) {
        pd_error(x, "torch.mha: No input list received.");
        return;
    }

    // convert input atoms to std::vector<float>
    std::vector<float> values;
    for (int i = 0; i < argc; ++i) {
        if (argv[i].a_type != A_FLOAT) {
            pd_error(x, "torch.mha: Only float values are accepted in list.");
            return;
        }
        values.push_back(atom_getfloat(argv + i));
    }

    int64_t total_floats = values.size();
    int64_t expected_div = x->batch_size * x->embed_dim;

    if (expected_div == 0) {
        pd_error(x, "torch.mha: Invalid configuration: batch_size * embed_dim == 0.");
        return;
    }

    if (total_floats % expected_div != 0) {
        pd_error(x, "torch.mha: Input size (%lld) must be divisible by batch_size (%d) * embed_dim (%d).", total_floats, x->batch_size, x->embed_dim);
        return;
    }

    int64_t sequence_length = total_floats / expected_div;
    post("torch.mha: sequence_length = %lld", sequence_length);
    
    // Create MHAParams with the current configuration
    contorchionist::core::dp_mha::MHAParams p;
    p.seq_len = x->seq_len;  // if seq_len == 0, infer from input shape
    p.batch_size = x->batch_size;
    p.embed_dim = x->embed_dim;
    p.flatten_output = true;  
    p.return_weights = x->need_weights; // if attention weights are needed

    contorchionist::core::dp_mha::MHAResult res;
    if (x->use_wrapper) {
        // wrapper: self-attention (Q=K=V=query)
        res = contorchionist::core::dp_mha::MHAProcessor(
            values, p, x->device, nullptr,
            [x](const at::Tensor& q){ return x->mha_wrapper->forward(q); }
        );
    } else {
        // multi-head attention
        res = contorchionist::core::dp_mha::MHAProcessor(
            values, p, x->device, &x->mha, {}
        );
    }

    if (!res.success) {
        pd_error(x, "torch.mha: %s", res.error_message.c_str());
        return;
    }

    
    // if attention weights are needed
    if (x->need_weights && res.attn_weights.defined()) {

        // output attention
        at::Tensor output = res.output.cpu().contiguous().view({-1});
        int64_t attn_output_size = output.size(0);
        t_atom *attn_output_atoms = (t_atom *)getbytes(sizeof(t_atom) * attn_output_size);

        for (int64_t i = 0; i < attn_output_size; ++i) {
            SETFLOAT(&attn_output_atoms[i], output[i].item<float>());
        }
        outlet_anything(x->x_out1, gensym("attn_output"), attn_output_size, attn_output_atoms);
        freebytes(attn_output_atoms, sizeof(t_atom) * attn_output_size);

        // attention weights
        at::Tensor attn_output_weights = res.attn_weights.cpu().contiguous().view({-1});
        int64_t att_weights_size = attn_output_weights.size(0);
        t_atom *att_weights_atoms = (t_atom*)getbytes(sizeof(t_atom)*att_weights_size);

        for (int64_t i=0;i<att_weights_size;++i){
            SETFLOAT(&att_weights_atoms[i], attn_output_weights[i].item<float>());
        }
        outlet_anything(x->x_out1, gensym("attn_output_weights"), att_weights_size, att_weights_atoms);
        freebytes(att_weights_atoms, sizeof(t_atom)*att_weights_size);

    } else{ // no attention weights
        at::Tensor output = res.output;

        // Flatten output
        output = output.view({-1});
        output = output.cpu();

        int64_t out_size = output.size(0);
        t_atom *out_atoms = (t_atom *)getbytes(sizeof(t_atom) * out_size);

        for (int64_t i = 0; i < out_size; ++i) {
            SETFLOAT(&out_atoms[i], output[i].item<float>());
        }
        // Send the output tensor
        outlet_anything(x->x_out1, &s_list, out_size, out_atoms);
        freebytes(out_atoms, sizeof(t_atom) * out_size);
    }
}



//* ------------------------ adds the MHA layer to a specific container (module) created by pdtoch.sequential ------------------------ *//
static void torch_mha_add_to_module(t_torch_mha *x, t_symbol *s, int argc, t_atom *argv) {
    if (x->added_to_module) {
        pd_error(x, "torch.mha: This instance has already been added to a module.");
        return;
    }
    if (argc != 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.mha: Please provide a module name to add to.");
        return;
    }
    t_symbol *module_name = atom_getsymbol(argv);

    bool added = PDGlobalState::add_layer_to_module(
        x, "mha", x->mha_wrapper, module_name,
        PDGlobalState::mha_registry,
        x->added_layer_index, x->added_layer_name,
        x->added_to_module, x->added_to_module_name
    );
    if (!added) return;

    x->use_wrapper = true; // set the use_wrapper flag to true

    // if is the first layer sends the input shape to a instance of torch.sequential that matches the module name only if this is the first layer
    auto it = PDGlobalState::module_registry.find(module_name->s_name);
    if (it != PDGlobalState::module_registry.end()) {
        std::shared_ptr<torch::nn::Sequential> container = it->second;
        if (container && container->get()->size() == 1) {
            torch_mha_send_input_shape(x, module_name);
        }
    }
}



//* ---------------- calls remove_layer function ------------------- *//
static void torch_mha_remove(t_torch_mha *x, t_symbol *s, int argc, t_atom *argv) {
    if (x->added_to_module) {
        PDGlobalState::remove_layer(x, "mha", x->verbose);
        x->use_wrapper = false;
    } else {
        pd_error(x, "torch.mha: This instance was not added to a module.");
    }
}



//* ------------------------ constructor ------------------------ *//
void *torch_mha_new(t_symbol *s, int argc, t_atom *argv) {
    t_torch_mha *x = (t_torch_mha *)pd_new(torch_mha_class);

    post("torch.mha: libtorch version: %s", TORCH_VERSION);

    // Default parameters 
    x->batch_size = 1; // default = 1
    x->embed_dim = 8;
    x->num_heads = 2;
    x->dropout = 0.0; // default dropout
    x->seq_len = 0; // if seq_len == 0, infer from input shape
    x->bias = true;
    x->add_zero_attn = false;
    x->added_to_module = false; // added to a module
    x->use_wrapper = false;
    x->need_weights = false;
    x->device = torch::kCPU;
    std::string device_str = "cpu"; 

    pd_utils::ArgParser parser(argc, argv, (t_object*)x);

    x->verbose = parser.has_flag("verbose v");

    x->embed_dim = static_cast<int>(parser.get_float("embed e", 8));
    x->num_heads = static_cast<int>(parser.get_float("heads h", 2));
    x->batch_size = static_cast<int>(parser.get_float("batch btz", 1));
    x->seq_len = static_cast<int>(parser.get_float("seqlength seqlen seq", 1));
    x->dropout = parser.get_float("dropout drop", 0.0);

    if (parser.has_flag("bias b")) {
        x->bias = true;
    }
    if (parser.has_flag("addzero addz")) {
        x->add_zero_attn = true;
    }
    if (parser.has_flag("weights w")) {
        x->need_weights = true;
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
        "torch.mha",
        device_flag_present
    );

    // create multi-head attention layer
    auto r = contorchionist::core::dp_mha::create_mha_layer(
        x->embed_dim, x->num_heads, x->bias, x->add_zero_attn, x->dropout, x->device
    );
    if (!r.success) {
        pd_error(x, "torch.mha: %s", r.error_message.c_str());
    } else {
        x->mha = r.mha;
        x->mha_wrapper = r.wrapper;
    }

    if (x->verbose) {
        post("torch.mha: Created with seq_length=%d, batch_size=%d, embed_dim=%d, heads=%d, bias=%s, dropout=%f, attention_weights=%s, add_zero_attn=%s",
            x->seq_len,
            x->batch_size, x->embed_dim,
            x->num_heads,
            x->bias ? "true" : "false",
            x->dropout,
            x->need_weights ? "true" : "false",
            x->add_zero_attn ? "true" : "false");
    }

    x->x_out1 = outlet_new(&x->x_obj, &s_anything);

    return (void *)x;
}


//* ------------------ destructor ------------------ *//
void torch_mha_destroy(t_torch_mha *x) {

    PDGlobalState::remove_layer(x, "mha", x->verbose); // remove the layer from the module
    freebytes(x->out, 1 * sizeof(t_atom));
    outlet_free(x->x_out1);
}

//* ------------------ setup function ------------------ *//
extern "C" void setup_torch0x2emha(void) {
    torch_mha_class = class_new(
        gensym("torch.mha"),
        (t_newmethod)torch_mha_new, 
        (t_method)torch_mha_destroy, 
        sizeof(t_torch_mha), 
        CLASS_DEFAULT, 
        A_GIMME, //Argument type: Optional Symbol (defaults to &s_)
        0);

    class_addlist(torch_mha_class, (t_method)torch_mha_forward); //receive a list of values
    class_addmethod(torch_mha_class, (t_method)torch_mha_heads, gensym("heads"), A_FLOAT, 0); // number of heads
    class_addmethod(torch_mha_class, (t_method)torch_mha_batch_size, gensym("batchsize"), A_FLOAT, 0); // batch size
    class_addmethod(torch_mha_class, (t_method)torch_mha_seqlength, gensym("seqlength"), A_FLOAT, 0); // seq_length
    class_addmethod(torch_mha_class, (t_method)torch_mha_embed_dim, gensym("embed"), A_FLOAT, 0); // embed_dim
    class_addmethod(torch_mha_class, (t_method)torch_mha_bias, gensym("bias"), A_FLOAT, 0); // bias
    class_addmethod(torch_mha_class, (t_method)torch_mha_add_zero_attn, gensym("add_zero_attn"), A_FLOAT, 0); // add_zero_attn
    class_addmethod(torch_mha_class, (t_method)torch_mha_device, gensym("device"), A_GIMME, 0); // device
    class_addmethod(torch_mha_class, (t_method)torch_mha_add_to_module, gensym("add"), A_GIMME, 0); // add to module
    class_addmethod(torch_mha_class, (t_method)torch_mha_remove, gensym("remove"), A_GIMME, 0); // remove from module
}
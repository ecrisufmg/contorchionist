#include <m_pd.h>        // Pure Data header
#include <torch/torch.h> // LibTorch header
#include <cstring>       // For std::memcpy and std::memset
#include <string>        // For std::string
#include <vector>        // For intermediate storage if needed
#include <stdexcept>     // For standard exceptions (used in catch blocks)
#include "../utils/include/pd_arg_parser.h" // Include the new argument parser
#include "../utils/include/pd_torch_types.h" // Include the object struct definition
#include "../utils/include/pd_torch_device_adapter.h"
#include "../../../core/include/core_ap_linear.h"


static t_class * t_torch_linear_tilde_class;




//* ------------------ perform routine ------------------ */
static t_int *linear_tilde_perform(t_int *w){

    // Get arguments passed by dsp_add
    t_torch_linear_tilde *x = ( t_torch_linear_tilde *)(w[1]); //pointer to the object
    t_sample *in_buf = (t_sample *)(w[2]); // input signal buffer
    t_sample *out_buf = (t_sample *)(w[3]); // output signal buffer
    int n = (int)(w[4]); // size of the block signal 


    // check if linear layer exists
    if (!x->linear_tilde) {
        pd_error(x, "torch.linear~: Linear layer not initialized. Output will be zero.");
        std::memset(out_buf, 0, sizeof(t_sample) * n);
        return (w + 5);
    }

    // check if the block size has the correct size for the linear layer
    if (n != x->in_features) {
        pd_error(x, "torch.linear~: Block size (%d) != in_features (%d). Output will be zero.", n, x->in_features);
        std::memset(out_buf, 0, sizeof(t_sample) * n);
        return (w + 5);
    }

    // process the input signal block
    try {
        // pass the input signal block to the linear layer
        at::Tensor output = contorchionist::core::ap_linear::LinearAProcessorSignal(
            *(x->linear_tilde), 
            x->device, in_buf, 
            x->in_features
        );

        // move the output tensor to CPU and copy to the output buffer
        output = output.to(torch::kCPU);

        // get output data pointer
        auto out_data = output.data_ptr<float>();

        // copy to the output buffer (untill out_features or n, the smallest)
        int out_n = std::min((int)x->out_features, n);

        // if out_n is greater than or equal to n, copy the output data
        if (out_n >= n){
            std::memcpy(out_buf, out_data, sizeof(t_sample) * n);
        }
        // if out_n is less than n, copy the output data and fill the rest with zeros
        else {
            std::memcpy(out_buf, out_data, sizeof(t_sample) * out_n);
            std::memset(out_buf + out_n, 0, sizeof(t_sample) * (n - out_n));
        }
    } catch (const c10::Error& e) {
        pd_error(x, "torch.linear~: LibTorch error: %s", e.what());
        std::memset(out_buf, 0, sizeof(t_sample) * n);
    } catch (const std::runtime_error& e) {
        pd_error(x, "torch.linear~: Runtime error: %s", e.what());
        std::memset(out_buf, 0, sizeof(t_sample) * n);
    } catch (const std::exception& e) {
        pd_error(x, "torch.linear~: Error: %s", e.what());
        std::memset(out_buf, 0, sizeof(t_sample) * n);
    }
    return (w + 5);
}


//* ------------------ add perform routine to the DSP-tree ------------------ */
static void linear_AddDsp( t_torch_linear_tilde *x, t_signal **sp){
     if (x->block_size != sp[0]->s_n) {
        x->block_size = sp[0]->s_n;
        x->in_features = sp[0]->s_n; // update in_features to match the block size
        // recreate the linear layer with the new block size
        x->linear_tilde = std::make_shared<torch::nn::Linear>(
            torch::nn::LinearOptions(x->in_features, x->out_features).bias(x->bias)
        );
        // x->linear_tilde->get()->to(x->device); // move the linear layer to the device specified (CPU, CUDA or MPS)
        post("torch.linear~: Block size/in_features set to %d", x->in_features);
    }
    
    /*
    add the perform routine to the DSP-tree 
    (1st arg: function that will process the signal blocks, 2nd arg: number of pointers to the array of samples, 
    3rd arg: pointers to the array of samples, 4th arg: size of the array of samples)
    The function linear_tilde_perform will be called for each signal vector: 
    sp[0]->s_vec is the input signal vector
    sp[1]->s_vec is the output signal vector
    sp[0]->s_n is the size of the signal vector (it is sufficient to get the length of one of these vectors, since all are the same length.)    
    */ 
    dsp_add(linear_tilde_perform, 4, x, sp[0]->s_vec, sp[1]->s_vec, (t_int)sp[0]->s_n);
}

//* ------------------ set the device ------------------ */
static void torch_linear_tilde_device( t_torch_linear_tilde *x, t_symbol *s, int argc, t_atom *argv) {
    // Check if the first argument is a symbol
    if (argc != 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.linear~: Please provide a device name.");
        return;
    }
    // get the device name received
    t_symbol *device_name = atom_getsymbol(&argv[0]);
    std::string dev = device_name->s_name;

    // set the device
    // x->device = PdTorchUtils::select_device(dev, (t_pd*)x);

    // move the linear layer to the new device
    // if (x->linear_tilde) {
    //    x->linear_tilde->get()->to(x->device);
    // }
}

//* ------------------ enable/disable bias ------------------ */
static void torch_linear_tilde_bias( t_torch_linear_tilde *x, t_floatarg b) {
    if(b < 0 || b > 1) {
        pd_error(x, "torch.linear~: bias must be 0 or 1.");
        return;
    }
    x->bias = (b != 0);

    //recreate the linear layer with new parameters
    x->linear_tilde = std::make_shared<torch::nn::Linear>(
        torch::nn::LinearOptions(x->in_features, x->out_features).bias(x->bias)
    );
    // x->linear_tilde->get()->to(x->device); // move the linear layer to the device specified (CPU, CUDA or MPS)
    post("torch.linear~: bias set to %d", x->bias);
}



//* ------------------ set out_features ------------------ *//
static void torch_linear_tilde_outsize( t_torch_linear_tilde *x, t_symbol *s, int argc, t_atom *argv) {
    //check if value received is valid
    if (argc < 1 || argv[0].a_type != A_FLOAT) {
        pd_error(x, "pdtorch.linea~: out_features must be less than or equal to block size.");
        return;
    }

    int new_out = (int)atom_getfloat(argv);
    if (new_out <= 0 || new_out > x->in_features) {
        pd_error(x, "torch.linear~: out_features must be > 0 and <= block size.");
        return;
    }
    //set the in_features and out_features values
    x->out_features = new_out;
    // rebuild the linear layer with new parameters
    x->linear_tilde = std::make_shared<torch::nn::Linear>(
        torch::nn::LinearOptions(x->in_features, x->out_features).bias(x->bias)
    );
    //move to device        
    // x->linear_tilde->get()->to(x->device); // move the linear layer to the device specified (CPU or CUDA)
    post("torch.linear~: out_features set to %d", x->out_features);
}


//* ------------------ constructor ------------------ */
static void *linear_tilde_New(t_symbol *s, int argc, t_atom *argv){
     t_torch_linear_tilde *x = ( t_torch_linear_tilde *)pd_new( t_torch_linear_tilde_class);

    post("torch.linear~: libtorch version: %s", TORCH_VERSION);

    // Check if the object was created successfully
    if (!x) {
        pd_error(nullptr, "torch.linear~: Failed to allocate memory for object.");
        return NULL;
    }

    //default parameters
    x->in_features = 64; // number of input features
    x->out_features = 64; // number of output features (neurons)
    x->bias = true; // if bias is used
   
    
    // read the in_features and out_features from the arguments
    int found_floats = 0;
    for (int i = 0; i < argc; ++i) {
        if (argv[i].a_type == A_FLOAT) {
            if (found_floats == 0) {
                x->in_features = (int)atom_getfloat(argv + i);
                found_floats++;
            } else if (found_floats == 1) {
                x->out_features = (int)atom_getfloat(argv + i);
                found_floats++;
                break;
            }
        }
    }
    //flags
    pd_utils::ArgParser parser(argc, argv, &x->x_obj);

    bool verbose_arg = false;
    if (parser.has_flag("verbose") || parser.has_flag("v")) {
        verbose_arg = true;
    }

    torch::Device parsed_device_obj = torch::kCPU; 
    std::string device_arg_str = "cpu"; 
     bool device_flag_present = false;
    if (parser.has_flag("device")) {
        device_arg_str = parser.get_string("device", "cpu");
        device_flag_present = true;
    } else if (parser.has_flag("d")) {
        device_arg_str = parser.get_string("d", "cpu");
        device_flag_present = true;
    } else if (parser.has_flag("cuda")) {
        device_arg_str = "cuda"; 
        device_flag_present = true;
    } else if (parser.has_flag("mps")) {
        device_arg_str = "mps"; 
        device_flag_present = true;
    }

    

    pd_parse_and_set_torch_device(&x->x_obj, parsed_device_obj, device_arg_str, verbose_arg, "torch.linear~", device_flag_present);
  
    //creates the linear layer
    x->linear_tilde = std::make_shared<torch::nn::Linear>(torch::nn::LinearOptions(x->in_features, x->out_features).bias(x->bias));
    //move to device
    x->linear_tilde->get()->to(x->device); // move the linear layer to the device specified (CPU, CUDA or MPS)
    
    post("torch.linear~: Created with in_features=%d, out_features=%d, bias=%d, device=%s", x->in_features, x->out_features, x->bias, pd_torch_device_friendly_name(x->device).c_str());

    //initialize the block size
    x->block_size = 0;
    
    // create signal outlet
    x->out = outlet_new(&x->x_obj, &s_signal);

    return (void *)x;
    
}



//* ------------------ destructor ------------------ */
static void linear_tilde_Free( t_torch_linear_tilde *x) {
    x->linear_tilde.reset();
}


extern "C" {
    void setup_torch0x2elinear_tilde(void)
    {
         t_torch_linear_tilde_class = class_new(
            gensym("torch.linear~"),
            (t_newmethod)linear_tilde_New,             // Constructor
            (t_method)linear_tilde_Free,               // Destructor
            sizeof( t_torch_linear_tilde),
            CLASS_DEFAULT,
            A_GIMME,                         // Use A_GIMME for flexible argument parsing
            0);                              // Argument list terminator

        if (! t_torch_linear_tilde_class) {
             pd_error(nullptr, "torch.linear~: Failed to create class.");
             return;
        }

        CLASS_MAINSIGNALIN( t_torch_linear_tilde_class,  t_torch_linear_tilde, x_f); //enable signal at the first inlet (1st argument: pointer to the class, 2nd argument: pointer to the object, 3rd argument: dummy variable) 
        class_addmethod( t_torch_linear_tilde_class, (t_method)linear_AddDsp, gensym("dsp"), A_CANT, 0); // send a message with the selector "dsp" when pd's audioengine is started
        class_addmethod( t_torch_linear_tilde_class, (t_method)torch_linear_tilde_device, gensym("device"), A_GIMME, 0); // set the device
        class_addmethod( t_torch_linear_tilde_class, (t_method)torch_linear_tilde_outsize, gensym("out"), A_GIMME, 0); //receive in_features and out_features
        class_addmethod( t_torch_linear_tilde_class, (t_method)torch_linear_tilde_bias, gensym("bias"), A_FLOAT, 0); // bias
    }
} // extern "C"
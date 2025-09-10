#include <m_pd.h>        // Pure Data header

//check if the version is 0.54 or higher for multi-channel support
#ifndef CLASS_MULTICHANNEL
#if PD_MINOR_VERSION >= 0.54 // check if the version is 0.54 or higher for multi-channel support
#define PD_HAVE_MULTICHANNEL // define the macro for multi-channel support
#else
#pragma message("Pd version bellow 0.54: building without multi-channel support") // send a message if the version is below 0.54
#define CLASS_MULTICHANNEL 0 // define the macro for multi-channel support as zero for no multi channel support
#endif
#else
#define PD_HAVE_MULTICHANNEL CLASS_MULTICHANNEL
#endif


#include <cstring>       // For std::memcpy and std::memset
#include <string>        // For std::string
#include <vector>        // For intermediate storage if needed
#include <stdexcept>     // For standard exceptions (used in catch blocks)
#include <torch/torch.h> // LibTorch header
#include <torch/script.h> // LibTorch header for TorchScript
#include <memory>   // For smart pointers
#include <thread> // For multithreading support
#include <atomic> // For atomic operations
#include <mutex> // For mutex support
#include <cmath>       // For std::abs
#include <fstream> // For file operations
#include <sstream> // For string stream operations
#include <chrono> // For timing operations
#include <cstdint> // For int64_t
#include <numeric> // For std::accumulate
#include <algorithm> // For std::min_element, std::max_element
#include "../utils/include/pd_torch_types.h" // Include the object struct definition
#include "../utils/include/pd_torch_utils.h" // Utility functions
#include "../../../core/include/core_ap_torchts.h" // include TorchTSProcessorSignal 
#include "../../../core/include/core_util_circbuffer.h" // Include the circular buffer implementation
#include "../../../core/include/core_ap_monitor.h" // Include the performance monitor
#include "../../../core/include/core_nn_ts_manager.h" // Include the header for managing models (load, select methods, extract shape, etc.)
#include "../utils/include/pd_torch_device_adapter.h"
#include "../utils/include/pd_arg_parser.h"

// for multi-channel support
#ifdef _WIN32 // for windows
  #define NOMINMAX
  #include <windows.h> // windows API
  #include <shlwapi.h> // windows utility functions to manipulate file paths

  std::string get_executable_path(){ // function to return the path of the .dll 
    HMODULE hModule = GetModuleHandle("torch.ts~.dll"); // get the handle of the dll module torch.ts~ 
    if (hModule) { // check if the handle is valid
      char path[MAX_PATH]; // buffer to store the path
      GetModuleFileName(hModule, path, sizeof(path)); // get the full path of the module
      PathRemoveFileSpec(path); // Remove filename, keeping just the path
      SetDllDirectory(path);    // Add the directory to the DLL search path
      std::string path_std = path; // convert the char array to string
      return path_std; // return the path
    }
  }
// for linux and macOS
#else
  #include <dlfcn.h> // for dynamic loading of shared libraries
#endif

// for mps support
#if defined(__APPLE__) && defined(__arm64__)
#define HAVE_MPS
#else
#pragma message("No MPS suport: building without MPS support") // send a message if the architecture is not arm64
#endif

// for multi-channel support
using t_signal_setmultiout = void (*)(t_signal **, int); // define the function pointer type for signal_setmultiout
static t_signal_setmultiout g_signal_setmultiout; // global variable to store the function pointer for the signal_setmultiout function

static t_class *torch_ts_tilde_class;



//* ----------------------- sets the device (cpu or cuda) --------------------- *//
static void torch_ts_tilde_device(t_torch_ts_tilde *x, t_symbol *s, int argc, t_atom *argv) {
    // Check if the first argument is a symbol
    if (argc != 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.ts~: Please provide a device (cpu, cuda or mps).");
        return;
    }
    // get the device name received
    t_symbol *device_name = atom_getsymbol(&argv[0]);
    std::string dev = device_name->s_name;

    // set the device
    pd_parse_and_set_torch_device(
        (t_object*)x,
        x->device,
        dev,
        x->verbose,
        "torch.ts~",
        true
    );
    // move the model to the new device
    if (x->model) {
        x->model->to(x->device);
    }
}

//* ----------------------- sets the circular buffers size --------------------- *//
static void torch_ts_tilde_buffer_size(t_torch_ts_tilde *x, t_floatarg size) {
    int requested_size = (int)size;
    int corrected_size;
    std::string error_message;
    std::string warning_message;

    // if async mode is disabled: buffer size = block size
    if (!x->async_mode) {
        x->buffer_size = x->block_size;
        if (x->verbose) {
            post("torch.ts~: Async mode disabled. Buffer size set to block size (%d).", x->block_size);
        }
    } else {
        // async mode enabled: validate the requested buffer size
        bool buffer_valid = contorchionist::core::ap_torchts::validate_buffer_size(
            requested_size, // requested buffer size
            x->block_size, // default block size
            x->model_buffer_size, // model internal buffer size
            x->max_buffer_size, // max buffer size
            true, // power of two verification = true
            true, // auto correct = true
            &corrected_size, // corrected buffer size
            &error_message, // validation error message
            &warning_message // validation warning message
        );
        //if buffer size is invalid
        if (!buffer_valid) {
            x->buffer_size = corrected_size;
            if (x->verbose) {
                post("torch.ts~: %s Buffer size set to %d.", error_message.c_str(), corrected_size);
            }
        // if buffer size is valid
        } else {
            x->buffer_size = requested_size;
            if (!warning_message.empty()) {
                post("torch.ts~: %s", warning_message.c_str());
            }
        } 
    }
    // Resize the input and output buffers to the new buffer size
    contorchionist::core::ap_torchts::create_circular_buffer(x->in_buffers, x->out_buffers, x->in_ch, x->out_ch, x->buffer_size);

    // Resize the input blocks to the new buffer size
    for (auto& block : x->in_blocks) {
        block.resize(x->buffer_size);
    }

    if (x->verbose) {
        post("torch.ts~: Buffer size set to %d samples", x->buffer_size);
    }
}


// * ------------------ pass the input signal through the loaded model (perform routine) ------------------ */
static t_int *torch_ts_tilde_perform(t_int *w) {
    t_torch_ts_tilde *x = (t_torch_ts_tilde *)(w[1]); // get the object pointer from the perform routine arguments

    // If the model is not loaded, send zero to output channels
    if (!x->loaded_model) {
        for (int ch = 0; ch < x->out_ch; ++ch) {
            t_sample *out = x->out_sig_vec[ch];
            std::memset(out, 0, x->block_size * sizeof(t_sample));
        }
        return (w + 2);
    } 

    // 1 - copies each input channel to its corresponding circular buffer
    for (int ch = 0; ch < x->in_ch; ++ch) {
        x->in_buffers[ch]->write_overwrite(x->in_sig_vec[ch], x->block_size); //(input channel, number of samples to write) 
    }

    // 2 - check if all the input circular buffers have enough data to trigger the processing (full)
    bool can_process = true;
    for (int ch = 0; ch < x->in_ch; ++ch) {
        if (x->in_buffers[ch]->getSamplesAvailable() < x->buffer_size) {
            can_process = false;
            break;
        }
    }
    // if all input buffers are full and ready 
    if (can_process) {
        bool should_run_model = false;
        // case 1: if async mode is enabled: check if the thread is already running and set should_run_model boolean accordingly
        if (x->async_mode) {
            if (!x->thread_running.load()) {
                should_run_model = true;
                x->thread_running.store(true);
            }
        // case 2: if synchronous mode is enabled set should_run_model boolean accordingly
        } else {
            should_run_model = true;
        }
        // 3 - if the model is ready: prepare input buffers and run the model
        if (should_run_model) {
            // Peek data from input buffers
            for (int ch = 0; ch < x->in_ch; ++ch) {
                // read the current audio window without consuming the data
                x->in_buffers[ch]->peek(x->in_blocks[ch].data(), x->buffer_size);
                x->in_ptrs[ch] = x->in_blocks[ch].data();
                // discard the oldest block of signal to slide the window
                x->in_buffers[ch]->discard(x->block_size);
            }
            // case 1: - ASYNC PROCESSING
            if (x->async_mode) {
                // start a new thread
                x->model_thread = std::make_unique<std::thread>([x, in_blocks = x->in_blocks]() mutable {
                    // create a vector of pointers to the thread-local input blocks
                    std::vector<const float*> in_ptrs;
                    in_ptrs.reserve(in_blocks.size());
                    for (const auto& block : in_blocks) {
                        in_ptrs.push_back(block.data());
                    }
                    // process the model
                    try {
                        bool batch = contorchionist::core::ap_torchts::TorchTSProcessorSignal(
                            *x->model, 
                            x->selected_method, 
                            in_ptrs, 
                            x->in_ch, 
                            x->out_ch, 
                            x->buffer_size,
                            x->model_buffer_size,
                            x->device, 
                            x->model_mutex, 
                            x->out_buffers
                        );
                    } catch (const std::exception& e) {
                        pd_error(x, "torch.ts~: Exception in model thread: %s", e.what());
                    }
                    // terminate the thread
                    x->thread_running.store(false);
                });
                // detach the thread
                x->model_thread->detach();
            // case 2: - SYNC PROCESSING
            } else {
                bool batch = contorchionist::core::ap_torchts::TorchTSProcessorSignal(
                    *x->model, 
                    x->selected_method, 
                    x->in_ptrs, 
                    x->in_ch, 
                    x->out_ch, 
                    x->buffer_size,
                    x->model_buffer_size,
                    x->device, 
                    x->model_mutex, 
                    x->out_buffers
                );    
            }
        }
    }
    // 4 - OUTPUT PROCESSING
    // for each dsp cycle, get a block of samples from the output circular buffers and send to the output signal vectors
    for (int ch = 0; ch < x->out_ch; ++ch) {
        // read the output circular buffer by blocks equal to the block size 
        size_t read = x->out_buffers[ch]->try_read(x->out_sig_vec[ch], x->block_size);
        // if the read samples are less than the block size, fill the rest with zeros
        if (read < x->block_size) {
            // if there are not enough samples in the output buffer, fill the rest with zeros
            std::memset(x->out_sig_vec[ch] + read, 0, (x->block_size - read) * sizeof(t_sample));
        }
    }

    return (w + 2);
}


//* ------------------ add perform routine to the DSP-tree ------------------ */
static void torch_ts_tilde_AddDsp(t_torch_ts_tilde *x, t_signal **sp) {

    // get the block size from the signal vector
    x->block_size = sp[0]->s_n;
    x->sample_rate = sp[0]->s_sr; // get the sample rate from the signal vector
    // clear the input and output vectors
    x->in_sig_vec.clear(); // clear the input vector
    x->out_sig_vec.clear(); // clear the output vector

    // set the buffer size to the block size if async mode is disabled and the buffer size is not equal to the block size
    if (!x->async_mode && x->block_size != x->buffer_size) {
        x->buffer_size = x->block_size;
        // create input and output circular buffers
        contorchionist::core::ap_torchts::create_circular_buffer(x->in_buffers, x->out_buffers, x->in_ch, x->out_ch, x->buffer_size);
        // resize the in_blocks and in_ptrs vectors to hold the input circular buffers
        x->in_blocks.resize(x->in_ch);
        x->in_ptrs.resize(x->in_ch);
        // print message
        if (x->verbose) {
            post("torch.ts~: Buffer size set to %d samples", x->buffer_size);
        }
    }

    // multi-channel mode
    if (x->multi_ch){
#if PD_HAVE_MULTICHANNEL
    int in_channels = sp[0]->s_nchans; // get the number of input channels
#else
    int in_channels = 1; // set the number of input channels to 1
#endif
        //map the input channels to the input vector
        for (int i = 0; i < x->in_ch; ++i){
             x->in_sig_vec.push_back(sp[0]->s_vec + x->block_size * (i % in_channels));
        }
        // set the number of output channels equal to x->out_ch
        g_signal_setmultiout(&sp[1], x->out_ch);
        //map the output channels to the output vector
        for (int i = 0; i < x->out_ch; ++i){
            x->out_sig_vec.push_back(sp[1]->s_vec + x->block_size * i);
        }
    }
    // no multi channel mode
    else {
        // fill the input vectors with the input vector pointers
        for (int i = 0; i < x->in_ch; ++i){
            x->in_sig_vec.push_back(sp[i]->s_vec);
        }
        // fill the output vectors with the output vector pointers
        for (int i = x->in_ch; i < x->in_ch + x->out_ch; i++){
            if (g_signal_setmultiout){
                g_signal_setmultiout(&sp[i], 1); // ensure single output for each channel
            }
            x->out_sig_vec.push_back(sp[i]->s_vec);
        }
    }
    // add the perform routine to the DSP-tree
    dsp_add(torch_ts_tilde_perform, 1, x);
    // print some info about latency
    if (x->verbose && x->sample_rate > 0) {
        float latency_ms = (static_cast<float>(x->buffer_size) / x->sample_rate) * 1000.0f;
        post("torch.ts~: Latency = %d samples (%.2f ms)", x->buffer_size, latency_ms);
    }
}



//* -------------------------- constructor -------------------------- //
void *torch_ts_tilde_New(t_symbol *s, int argc, t_atom *argv) {
    t_torch_ts_tilde*x = (t_torch_ts_tilde*)pd_new(torch_ts_tilde_class);

    // post("torch.ts~: libtorch version: %s", TORCH_VERSION);

    // Check if the object was created successfully
    if (!x) {
        pd_error(nullptr, "torch.ts~: Failed to allocate memory for object.");
        return NULL;
    }

    // initialize default parameters
    x->loaded_model = false; // model not loaded by default
    x->multi_ch = false; // multi-channel support
    x->m_canvas = canvas_getcurrent(); // get the current canvas
    x->name = gensym("torch.ts~"); // set the name of the object
    x->in_ch = 1; // number of input channels
    x->out_ch = 1; // number of output channels
    x->block_size = 64; // block size
    x->sample_rate = 0; // sample rate
    x->model_buffer_size = 0;
    x->max_buffer_size = 4096;
    x->buffer_size = x->block_size; // default buffer size
    x->last_block_size = 0; // last block size
    x->tensors_struct = new PdTorchTensors(); // create the tensors struct
    x->selected_method = "forward"; // set the default method to be used on the loaded model
    new (&x->model_mutex) std::mutex(); // initialize the mutex for the model thread
    x->model_thread = nullptr;
    x->thread_running = false; // model thread is not running by default
    x->async_mode = false; // asynchronous mode is not enabled by default
    x->last_async_mode = false; // last async mode is not enabled by default
    std::string device_str = "cpu"; //device

    // criar logger para o performance monitor
    auto pd_log_function = [](const std::string& msg) {
        post("%s", msg.c_str());  // post() é específico do PD
    };

    // if no arguments are passed or the first argument is not a symbol
    if (argc < 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.ts~: WARNING: No model passed. Please provide a model name.");
        return (void *)x;
    }

    t_symbol *model_name_symbol = argv[0].a_w.w_symbol;
    const char *model_name = model_name_symbol->s_name;

    // if the model name doesn't end with .ts
    size_t name_len = strlen(model_name);
    bool has_ts_extension = (name_len > 3 && strcmp(model_name + name_len - 3, ".ts") == 0);
    if (!has_ts_extension) {
        pd_error(x, "torch.ts~: WARNING: Model name '%s' must have '.ts' extension. Make sure it's a TorchScript file.", model_name);
    }
    // Check if the model name is a flag
    if (model_name[0] == '-') {
        pd_error(x, "torch.ts~: First argument '%s' appears to be a flag. Model name must come first.", model_name);
        return (void *)x;
    }

    //read the model path (for now, the model must be at the same directory as the patch)
    x->model_path = atom_gensym(argv);
    
    //------- parse arguments -----//
    pd_utils::ArgParser parser(argc-1, argv+1, (t_object*)x);
    // verbose
    x->verbose = parser.has_flag("verbose v");
    // model method
    x->selected_method = parser.get_string("method m", "forward");
    // device
    bool device_flag_present = parser.has_flag("device d");
    std::string device_arg_str = parser.get_string("device d", "cpu");
    auto device_result = get_device_from_string(device_arg_str);
    // x->device = device_result.first;
    // bool device_parse_success = device_result.second;
    pd_parse_and_set_torch_device(
        (t_object*)x,
        x->device,
        device_arg_str,
        x->verbose,
        "torch.ts~",
        device_flag_present
    );

    // multi-channel
    bool mc_flag_present = parser.has_flag("multichannel mc");
    if (mc_flag_present) { // check if multi-channel flag is present
        if (g_signal_setmultiout) { // check if the multi-channel support is available
            x->multi_ch = true; // set the multi-channel as true
            // post("torch.ts~: multi-channel support enabled");
        // if multi-channel support is not available, send a warning
        } else {
            int max = 0, min = 0, bug = 0;
            sys_getversion(&max, &min, &bug); // get the Pd version
            pd_error(x, "torch.ts~: WARNING: No multichannel support in Pd %i.%i-%i", max, min, bug);
        }
    }

    // ------- setting parameters and arguments ------ //

    // load the model
    const char *canvas_dir = canvas_getdir(x->m_canvas)->s_name;
    char dirname[MAXPDSTRING], *dummy;
    char normalized[MAXPDSTRING];
    int fd = open_via_path(canvas_dir, x->model_path->s_name, "", dirname, &dummy, MAXPDSTRING, 1);
    if (fd >= 0) {
        sys_close(fd);
        char fullpath[MAXPDSTRING];
        snprintf(fullpath, MAXPDSTRING, "%s/%s", dirname, x->model_path->s_name);
        sys_unbashfilename(fullpath, normalized);
    } else {
        pd_error(x, "torch.ts~: File not found: %s", x->model_path->s_name);
        return (void *)x;
    }
    //store the error message if the model fails to load
    std::string load_error_message; 
    // load the model from the path normalized and send it to device
    bool loaded = contorchionist::core::nn_ts_manager::load_torchscript_model(normalized, x->model, x->device, &load_error_message);

    // check if the model was loaded successfully 
    if (!loaded) {
         x->loaded_model = false;
         pd_error(x, "torch.ts~: %s", load_error_message.c_str());
         return (void *)x;
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

    // get the number of in/out channels, model's input size and max buffer size from the model selected method
    std::string error_message;
    std::vector<std::string> log_messages;
    bool channels_extracted = contorchionist::core::nn_ts_manager::extract_channels_method(
        x->model.get(), // model
        x->selected_method, // method
        x->in_ch, // input channels
        x->out_ch, // output channels
        x->model_buffer_size, // model internal buffer size
        x->max_buffer_size, // max buffer size that the model can handle
        &error_message, // error message
        &log_messages // log messages
    );
    // error handling for channel extraction
    if (!channels_extracted) {
        pd_error(x, "torch.ts~: %s", error_message.c_str());
    }

    // log messages from the channel extraction
    if (x->verbose && !log_messages.empty()) {
        for (const auto& msg : log_messages) {
            post("torch.ts~: %s", msg.c_str());
        }
    }

    //async mode
    bool has_async_flag = parser.has_flag("async");
    int default_buffer_size = (x->model_buffer_size > 0) ? x->model_buffer_size : 64;
    int requested_size;
    if (has_async_flag) {
        x->async_mode = true;
        x->last_async_mode = true;
        requested_size = parser.get_float("async", default_buffer_size);
        //log
        if (x->verbose) {
            if (x->model_buffer_size > 0) {
                post("torch.ts~: Async mode enabled. Using model buffer size %d as default.", x->model_buffer_size);
            } else {
                post("torch.ts~: Async mode enabled. Model buffer size not available, using default 64.");
            }
        }
    }
    
    // validate and set circular buffer size
    int corrected_size;
    std::string error_messages;
    std::string warning_message;

    // if async mode is disabled: buffer size = block size
    if (!x->async_mode) {
        x->buffer_size = x->block_size;
        if (x->verbose) {
            post("torch.ts~: Async mode disabled. Buffer size set to block size (%d).", x->block_size);
        }
    } else {
        // async mode enabled: validate the requested buffer size
        bool buffer_valid = contorchionist::core::ap_torchts::validate_buffer_size(
            requested_size, // requested buffer size
            x->block_size, // default block size
            x->model_buffer_size, // model internal buffer size
            x->max_buffer_size, // max buffer size
            true, // power of two verification = true
            true, // auto correct = true
            &corrected_size, // corrected buffer size
            &error_messages, // validation error message
            &warning_message // validation warning message
        );
        //if buffer size is invalid
        if (!buffer_valid) {
            x->buffer_size = corrected_size;
            if (x->verbose) {
                post("torch.ts~: %s Buffer size set to %d.", error_messages.c_str(), corrected_size);
            }
        // if buffer size is valid
        } else {
            x->buffer_size = requested_size;
            if (!warning_message.empty()) {
                post("torch.ts~: %s", warning_message.c_str());
            }
        } 
    }
    // create input and output circular buffers
    contorchionist::core::ap_torchts::create_circular_buffer(x->in_buffers, x->out_buffers, x->in_ch, x->out_ch, x->buffer_size);

    // resize the in_blocks and in_ptrs vectors to hold the input circular buffers
    x->in_blocks.resize(x->in_ch);
    x->in_ptrs.resize(x->in_ch);

    // Resize the input blocks to the new buffer size
    for (auto& block : x->in_blocks) {
        block.resize(x->buffer_size);
    }

    // if no multi channel support available, create the inlets and outlets manually
    if (!x->multi_ch){
        // create the inlet signals
        if (x->in_ch > 0){
            for (int i = 0; i < x->in_ch; ++i) {
                if (i < x->in_ch - 1) { // first signal inlet is created for default
                    inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
                }
            }
        }
        // create outlet signal 
        if(x->out_ch > 0) { 
            for (int i = 0; i < x->out_ch; ++i) {
                outlet_new(&x->x_obj, &s_signal);
            }
        } else {
            outlet_new(&x->x_obj, &s_signal); // create a default outlet
        }
    }
    // if multi channel support available, create a signal outlet (a signal inlet is automatically created by default)
    else {
        outlet_new(&x->x_obj, &s_signal); // create a default outlet
    }
    
    // print log message 
    if (x->verbose) {
        post("torch.ts~: Configuration summary:");
        post("  - Model: %s", normalized);
        post("  - Method: %s", x->selected_method.c_str());
        post("  - Input channels: %d", x->in_ch);
        post("  - Output channels: %d", x->out_ch);
        post("  - Async mode: %s", x->async_mode ? "enabled" : "disabled");
        post("  - Buffer size: %d samples", x->buffer_size);
        post("  - Multi-channel mode: %s", x->multi_ch ? "enabled" : "disabled");
        
        for (int i = 0; i < x->in_ch; ++i) {
            post("  - Input buffer %d capacity: %zu", i, x->in_buffers[i]->getCapacity());
        }
        for (int i = 0; i < x->out_ch; ++i) {
            post("  - Output buffer %d capacity: %zu", i, x->out_buffers[i]->getCapacity());
        }
    }

    return (void *)x;
}


//* ------------------------- destructor ------------------------- //
void torch_ts_tilde_Free(t_torch_ts_tilde *x) {

    // wait for the model thread to finish if it is running
    while (x->thread_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    delete x->tensors_struct; // free the tensors struct
    x->tensors_struct = nullptr;
}


extern "C" {
    void setup_torch0x2ets_tilde(void){

// for multi-channel support
#ifdef PD_HAVE_MULTICHANNEL // check if multi-channel support is available
#ifdef _WIN32 // for windows
    HMODULE module;
    if (GetModuleHandleEx( // get the module handle where the function pd_typedmess is located
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            (LPCSTR)&pd_typedmess, &module)) {
        g_signal_setmultiout = (t_signal_setmultiout)(void *)GetProcAddress( // get the address of the function signal_setmultiout stored in g_signal_setmultiout
            module, "signal_setmultiout");
    }
// for linux and macOS
#else
    g_signal_setmultiout = (t_signal_setmultiout)dlsym( // use dlsym to get the address of symbol signal_setmultiout and store it in g_signal_setmultiout
        dlopen(nullptr, RTLD_NOW), "signal_setmultiout"); // ensure that the search starts from the main executable
#endif
#endif // PD_HAVE_MULTICHANNEL

        torch_ts_tilde_class = class_new(
            gensym("torch.ts~"),
            (t_newmethod)torch_ts_tilde_New,             // Constructor
            (t_method)torch_ts_tilde_Free,               // Destructor
            sizeof(t_torch_ts_tilde),
            CLASS_MULTICHANNEL, // for multi-channel support
            A_GIMME,                         // Use A_GIMME for flexible argument parsing
            0);                              // Argument list terminator

        if (!torch_ts_tilde_class) {
             pd_error(nullptr, "torch.ts~: Failed to create class.");
             return;
        }

        CLASS_MAINSIGNALIN(torch_ts_tilde_class, t_torch_ts_tilde, x_f);
        class_addmethod(torch_ts_tilde_class, (t_method)torch_ts_tilde_AddDsp, gensym("dsp"), A_CANT, 0);
        class_addmethod(torch_ts_tilde_class, (t_method)torch_ts_tilde_device, gensym("device"), A_GIMME, 0);
        class_addmethod(torch_ts_tilde_class, (t_method)torch_ts_tilde_buffer_size, gensym("buffersize"), A_FLOAT, 0);         
    }
} // extern "C"
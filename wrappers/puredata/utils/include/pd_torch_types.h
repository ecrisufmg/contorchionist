#include "m_pd.h"
#include <torch/torch.h>
#include <string>
#include <mutex>    
#include <atomic>   
#include <thread>   
#include <memory>  
#include "../../../core/include/core_util_circbuffer.h"
#include "../../../core/include/core_ap_torchts.h"
#include "../../../core/include/core_dp_mha.h"
#include "../../../core/include/core_ap_monitor.h"
#pragma once //avoid multiple inclusions





/*
pd_torch_types.h file defines all the data structures (structs) used by the Pure Data externals in the contorchionist project, 
serving as a central header for representing the internal state, parameters, and buffers of each neural or tensor processing object. 
This header enables seamless ensures consistency and interoperability among different externals, 
and simplifies the development and maintenance of new objects.
*/


// ------------------- Struct pdtorch.mha ------------------- //
typedef struct _torch_mha {
    t_object x_obj;
    torch::nn::MultiheadAttention mha{nullptr}; // multi-head attention module 
    int batch_size;      // batch size
    int embed_dim;       // input dimension
    int num_heads;       // number of heads
    int seq_len;     // sequence length
    size_t added_layer_index; // index of the layer in the module
    bool bias;           // if bias is used 
    bool add_zero_attn;  // if zero attention is added
    bool added_to_module = false; // boolean to check if it was added to a module
    bool use_wrapper; // true: wrapper for use inside a module, false: muti-head attention for isolated use 
    bool verbose; // verbose flag for logging
    bool need_weights; // if attention weights are needed
    float dropout; // dropout probability
    contorchionist::core::dp_mha::MHAWrapper mha_wrapper; // Wrapper customizado
    t_symbol* added_to_module_name = nullptr; // stores the name of the module wich was added to (created by pdtorch.sequential)
    t_symbol* added_layer_name = nullptr;
    torch::Device device; // CPU or CUDA
    t_atom  *out; //output buffer
    t_outlet *x_out1;    // Outlet para saída
} t_torch_mha;

// ------------------- Struct pdtorch.conv ------------------- //
typedef struct _torch_conv {
    t_object x_obj;

    int in_channels; // number of input channels
    int out_channels; // number of output channels
    int kernel_size; // kernel size (single int for 1D conv)
    int stride; // stride size
    int padding; // padding size
    int dilation; // dilation size
    int groups; // number of groups for grouped convolution
    bool bias; // whether to use bias in the convolution
    bool verbose; // verbose flag for logging

    std::string padding_mode; // padding mode (e.g., "zeros", "reflect", "replicate", "circular")
    int batch_size; // batch size for input tensors
    torch::Device device; // device
    
    std::shared_ptr<torch::nn::Conv1d> conv; // shared pointer to the Conv1d layer
    bool added_to_module; // flag to check if the layer was added to a module
    size_t added_layer_index; // index of the layer in the module
    t_symbol* added_layer_name; // stores the name of the layer in the module
    t_symbol* added_to_module_name; // stores the name of the module which was added to (created by pdtorch.sequential)
    t_outlet* x_out1; // outlet for processed tensor
} t_torch_conv;




// ------------------- Struct pdtorch.activation ------------------- //
typedef struct _torch_activation {
    t_object x_obj;
    float alpha;  //leakyrelu alpha
    float lambda; //softshrink and hardshrink lambda
    bool added_to_module = false; // boolean to check if it was added to a module
    bool verbose; // verbose flag for logging
    size_t added_layer_index; // index of the layer in the module
    int64_t dim; //dim for softmax, logsoftmax and softmin
    std::vector<int64_t> shape; // shape of the tensor
    t_symbol *activation; // activation function
    t_symbol* added_to_module_name = nullptr; // stores the name of the module wich was added to (created by pdtorch.sequential)
    t_symbol* added_layer_name = nullptr;
    torch::Device device; // Stores the chosen compute device (CPU or CUDA)
    t_atom *out;
    t_outlet *x_out1; // outlet for processed tensor
}t_torch_activation;


// -----------------------// Struct pdtorch.linear ------------------- //
typedef struct _torch_linear {
    t_object x_obj;

    int64_t in_features; // number of input features
    int64_t out_features; // number of output features (number of neurons)
    int64_t batch_size; // batch size
    bool bias; // if bias is used
    bool added_to_module;
    bool verbose; // verbose flag for logging

    std::shared_ptr<torch::nn::Linear> linear; // linear layer

    t_symbol *added_to_module_name; // stores the name of the module wich was added to (created by pdtorch.sequential)
    t_symbol *added_layer_name; // stores the name of the layer in the module
    
    size_t added_layer_index; // index of the layer in the module

    torch::Device device;
    t_atom *out; // output buffer
    t_outlet *x_out1;
} t_torch_linear;


// -----------------------// struct for complex C++ members ------------------- //
/* This struct serves as a container for complex C++ members that are not directly supported by Pure Data.
Encapsulate the dataset tensor and other complex members in this struct to avoid issues with Pure Data's memory management.
*/
struct PdTorchTensors {
    at::Tensor dataset_tensor; // tensor for the dataset loaded from a file by pdtorch.sequential
    at::Tensor target_tensor; // tensor for the target loaded from a file by pdtorch.sequential
    at::Tensor input_tensor_model; // tensor for the model loaded from a file by pdtorch.load
    at::Tensor input_tensor_model_tilde; // tensor for the model loaded from a file by pdtorch.load~
    at::Tensor mel_fb; // mel filterbank tensor 
    std::function<at::Tensor(const at::Tensor&, const at::Tensor&)> loss_function; //stores the loss function by pdtorch.sequential (it receives two tensors and returns one tensor with a loos function)
};


// -----------------------// Struct pdtorch.reshape ------------------- //
typedef struct _torch_reshape {
    t_object x_obj;
    std::string mode; // "view", "reshape", "flatten", "squeeze", "unsqueeze", "permute", "transpose"
    std::vector<int64_t> shape; // para view/reshape
    int64_t dim1, dim2; // para unsqueeze/squeeze/transpose
    std::vector<int64_t> dims; // para permute

    std::vector<int64_t> input_shape; // stores the input shape
    int64_t out_tensor_size; // stores the size of the output tensor
    int64_t out_shape_size; // stores the size of the output shape

    bool added_to_module;
    bool verbose; // verbose flag for logging
    bool recursive; // if true, sends the tensor recursively (nested lists), if false, sends a flat list
    size_t added_layer_index; // index of the layer in the module
    t_symbol *added_to_module_name; // stores the name of the module wich was added to (created by pdtorch.sequential)
    t_symbol *added_layer_name; // stores the name of the layer in the module
    t_symbol *reshape_method; // stores the reshape method

    torch::Device device;
    t_atom *out_tensor; // output buffer
    t_atom *out_shape; // output shape buffer
    t_outlet *x_out1;
} t_torch_reshape;



// -----------------------// Struct pdtorch.linear_tilde ------------------- //
typedef struct _torch_linear_tilde
{
    t_object x_obj; // Pd object management header
    t_sample x_f;   // Dummy for CLASS_MAINSIGNALIN

    int in_features; // number of input features (block size)
    int out_features; // number of output features (number of neurons)
    int64_t batch_size; // batch size
    int block_size;       // Pd's signal block size (vectorsize)
    bool bias; // if bias is used
    bool verbose; // verbose flag for logging

    torch::Device device; // Stores the chosen compute device (CPU or CUDA)

    std::shared_ptr<torch::nn::Linear> linear_tilde; // linear layer
   

    t_outlet *out; // Outlet for the signal output

} t_torch_linear_tilde;



// -----------------------// Struct pdtorch.load ------------------- //
typedef struct _torch_ts{
    t_object x_obj;

    bool loaded_model; // flag to check if the model is loaded
    bool verbose; // verbose flag for logging

    std::vector<int64_t> input_shape_model; // shape of the input tensor for the loaded model
    std::vector<int64_t> output_shape_model; // shape of the output tensor for the loaded model

    std::string selected_method; // selected method for the loaded model 
    std::vector<std::string> available_methods; // available methods for thel loaded model
    std::vector<std::string> available_attributes; // available attributes for the loaded model

    torch::Device device; // CPU, CUDA ou MPS device

    PdTorchTensors* tensors_struct; // pointer to the tensors struct

    std::unique_ptr<torch::jit::script::Module> model; // pointer to the loaded model

    t_symbol *name; // name of the object

    t_canvas*  m_canvas; //canvas
    t_atom *out; // output buffer
    t_outlet *x_out1; // outlet for processed tensor

}t_torch_ts;


// -----------------------// Struct pdtorch.load~ ------------------- //
typedef struct _torch_ts_tilde{
    t_object x_obj;
    t_sample x_f;   // Dummy for CLASS_MAINSIGNALIN

    int block_size;       // Pd's signal block size (vectorsize)
    int model_buffer_size; // size of the internal buffer used by the model (circular buffer size must be less than this value)
    int max_buffer_size; // max buffer size allowed by the model
    float sample_rate; // sample rate

    int  in_ch; // number of input channels
    int  out_ch; // number of output channels
    int  buffer_size; // size of the input and output buffers
    int last_block_size;

    bool can_batch;
    bool last_async_mode;
    bool loaded_model; // flag to check if the model is loaded
    bool multi_ch; // flag to check multi-channel support
    bool async_mode; // flag to set the asynchronous processing mode
    bool verbose; // verbose flag for logging

    std::vector<int64_t> input_shape_model; // shape of the input tensor for the loaded model
    std::vector<int64_t> output_shape_model; // shape of the output tensor for the loaded model

    std::string selected_method; // selected method for the loaded model 
    std::vector<std::string> available_methods; // available methods for thel loaded model
    std::vector<std::string> available_attributes; // available attributes for the loaded model

    std::vector<std::unique_ptr<contorchionist::core::util_circbuffer::CircularBuffer<float>>> in_buffers;  // circular buffers for input (one per channel)
    std::vector<std::unique_ptr<contorchionist::core::util_circbuffer::CircularBuffer<float>>> out_buffers; // circular buffers for output (one per channel)

    std::vector<std::vector<float>> in_blocks; // intermediate buffer to hold the input blocks for each channel
    std::vector<const float*> in_ptrs; // pointers to the input blocks for each channel

    // management of the model thread
    std::unique_ptr<std::thread> model_thread;  // model's thread for processing the model asynchronously.
    std::atomic<bool> thread_running; // flag to check if the model thread is running
    std::atomic<float> last_processing_time_ms; // For benchmark logging
    std::mutex model_mutex;    // Protects the access to the model thread and its data.

    // not used yet
    // std::unique_ptr<contorchionist::core::ap_monitor::PdMonitor> perf_monitor;
    
    std::vector<t_sample*> in_sig_vec; // input signal vector
    std::vector<t_sample*> out_sig_vec; // output signal vector
 
    torch::Device device; // CPU, CUDA ou MPS device

    PdTorchTensors* tensors_struct; // pointer to the tensors struct

    // not used yet
    // contorchionist::core::ap_torchts::ProcessingMode processing_mode; // processing mode
    // c10::IValue hidden_state; // hidden state for recurrent models

    std::unique_ptr<torch::jit::script::Module> model; // pointer to the loaded model

    t_symbol *name; // name of the object
    t_symbol *model_path; // path to the model

    t_canvas*  m_canvas; //canvas
    t_atom *out; // output buffer
    t_outlet *x_out1; // outlet for processed tensor

}t_torch_ts_tilde;


// ------------------- Struct torch.tensor ------------------- //
typedef struct _torch_ls2tensor {
    t_object x_obj;
    t_symbol *name; // unique name for this tensor instance

    PdTorchTensors* tensors_struct; // Holds the actual dataset and target tensors

    std::vector<int64_t> tensor_shape; // Shape of the full dataset tensor
    std::vector<int64_t> sample_shape; // Shape of a single sample/row
    bool has_target; // Flag to indicate if data includes a target for supervised learning
    int64_t target_size; // Number of elements in the target part of a row

    std::vector<float> data_buffer; // Internal buffer to accumulate incoming float data
    size_t buffer_write_pos; // Current position in the data_buffer
    t_canvas*  m_canvas; //canvas

    bool recursive; // Flag for recursive output
    bool verbose; // verbose flag for logging
    bool shuffle; // Flag for shuffling the dataset
    torch::Device device; // Not strictly necessary if it only holds data, but good for consistency

    t_atom *out_tensor; // output buffer
    t_atom *out_shape; // output shape buffer
    
    t_outlet *x_out1; // Outlet to send a bang when the tensor is full
    t_outlet *x_out2; // Outlet to send the bang
} t_torch_ls2tensor;




// add other structs here
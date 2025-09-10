#include "pd_torch_utils.h"
#include "pd_torch_types.h"
#include "../../../core/include/core_util_circbuffer.h"   // Include the circular buffer implementation
#include <torch/script.h>
#include <cmath>       // For M_PI, cos
#include "m_pd.h"      // For post() and pd_error()

// Define M_PI if not already defined (might be needed on some platforms/compilers)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//PD global variable to check if DSP is on
extern "C" {
    extern int canvas_dspstate;
}

namespace PdTorchUtils {

// ----------------- select the device (CPU or CUDA) ----------------- //
torch::Device select_device(const std::string& device_str, t_pd* pd_obj) {
    if (device_str == "cuda") {
        if (torch::cuda::is_available()) {
            if (pd_obj) post("torch: Using CUDA device.");
            return torch::Device(torch::kCUDA);
        } else {
            if (pd_obj) pd_error(pd_obj, "torch: CUDA requested but not available. Using CPU.");
            return torch::Device(torch::kCPU);
        }

    } else if (device_str == "mps" || device_str == "metal") {
#if defined(__APPLE__) && defined(__arm64__)
        if (torch::hasMPS() && torch::mps::is_available()) {
            if (pd_obj) post("torch: Using Metal (MPS) device.");
            return torch::Device(torch::kMPS);
        } else {
            if (pd_obj) pd_error(pd_obj, "torch: MPS/Metal requested but not available. Using CPU.");
            return torch::Device(torch::kCPU);
        }
#else
        if (pd_obj) pd_error(pd_obj, "torch: MPS/Metal requested but not supported on this platform. Using CPU.");
        return torch::Device(torch::kCPU);
#endif
    
    } else if (device_str == "cpu") {
        if (pd_obj) post("torch: Using CPU device.");
        return torch::Device(torch::kCPU);
    } else {
        if (pd_obj) pd_error(pd_obj, "torch: Unknown device '%s'. Using CPU.", device_str.c_str());
        return torch::Device(torch::kCPU);
    }
}
 

// ----------------- check if DSP is on ----------------- //
bool is_dsp_on() {
    return canvas_dspstate != 0;
}

// ----------------- create a circular buffer ----------------- //
// void create_circular_buffer(std::vector<std::unique_ptr<contorchionist::core::util_circbuffer::CircularBuffer<float>>>& in_buffers, std::vector<std::unique_ptr<contorchionist::core::util_circbuffer::CircularBuffer<float>>>& out_buffers, int in_channel, int out_channel, int buffer_size){

//     // resize the buffers
//     in_buffers.clear();
//     out_buffers.clear();

//     // Reserve capacity for efficiency
//     in_buffers.reserve(in_channel);
//     out_buffers.reserve(out_channel);
    
//     // initialize the input circular buffers (one for each input channel)
//     for (int i = 0; i < in_channel; ++i) {
//         in_buffers.emplace_back(std::make_unique<contorchionist::core::util_circbuffer::CircularBuffer<float>>(buffer_size));
//     }

//     // initialize the output circular buffers (one for each output channel)
//     for (int i = 0; i < out_channel; ++i) {
//         out_buffers.emplace_back(std::make_unique<contorchionist::core::util_circbuffer::CircularBuffer<float>>(buffer_size));
//     }
// }


// ---------------------- print reshape configuration -------------------------------------
void print_reshape_config(t_torch_reshape *x, const std::string& method, bool verbose) {
    if (!verbose && !x->verbose) return;

    std::stringstream ss;
    ss << "torch.reshape: method=" << method;

    if (method == "reshape" || method == "view") {
        ss << " shape=[";
        for (size_t i = 0; i < x->shape.size(); ++i) {
            ss << x->shape[i];
            if (i + 1 < x->shape.size()) ss << ", ";
        }
        ss << "]";
        
    } else if (method == "flatten") {
        ss << " start_dim=" << x->dim1 << " end_dim=" << x->dim2;
        
    } else if (method == "squeeze" || method == "unsqueeze") {
        ss << " dim=" << x->dim1;
        
    } else if (method == "permute") {
        ss << " dims=[";
        for (size_t i = 0; i < x->dims.size(); ++i) {
            ss << x->dims[i];
            if (i + 1 < x->dims.size()) ss << ", ";
        }
        ss << "]";
        
    } else if (method == "transpose") {
        ss << " dim1=" << x->dim1 << " dim2=" << x->dim2;
    }

    post("%s", ss.str().c_str());
}

// send the tensor shape to the outlet
void send_tensor_shape(const at::Tensor& tensor, t_outlet* outlet) {
    int64_t ndim = tensor.dim();
    t_atom *shape_atoms = (t_atom *)getbytes(sizeof(t_atom) * ndim);
    
    for (int64_t i = 0; i < ndim; ++i) {
        SETFLOAT(&shape_atoms[i], tensor.size(i));
    }
    
    outlet_anything(outlet, gensym("shape"), ndim, shape_atoms);
    freebytes(shape_atoms, sizeof(t_atom) * ndim);
}

// send tensor as a list
void send_tensor_1d(const at::Tensor& tensor, t_outlet* outlet) {
    int64_t size = tensor.size(0);
    t_atom *data = (t_atom *)getbytes(sizeof(t_atom) * size);
    
    for (int64_t i = 0; i < size; ++i) {
        SETFLOAT(&data[i], tensor[i].item<float>());
    }
    
    outlet_anything(outlet, gensym("tensor"), size, data);
    freebytes(data, sizeof(t_atom) * size);
}

// send tensor as a 2D array (row-major)
void send_tensor_2d_by_rows(const at::Tensor& tensor, t_outlet* outlet) {
    int64_t rows = tensor.size(0);
    int64_t cols = tensor.size(1);
    
    for (int64_t row = 0; row < rows; ++row) {
        t_atom *row_data = (t_atom *)getbytes(sizeof(t_atom) * cols);
        
        for (int64_t col = 0; col < cols; ++col) {
            SETFLOAT(&row_data[col], tensor[row][col].item<float>());
        }
        
        outlet_anything(outlet, gensym("tensor"), cols, row_data);
        freebytes(row_data, sizeof(t_atom) * cols);
    }
}

// send tensor recursively
void send_tensor_recursive(const at::Tensor& tensor, t_outlet* outlet, int level) {
    if (tensor.dim() == 0) {
        // Escalar
        t_atom scalar_atom;
        SETFLOAT(&scalar_atom, tensor.item<float>());
        outlet_anything(outlet, gensym("tensor"), 1, &scalar_atom);
        
    } else if (tensor.dim() == 1) {
        // 1D: enviar como lista
        send_tensor_1d(tensor, outlet);
        
    } else if (tensor.dim() == 2) {
        // 2D: enviar linha por linha
        send_tensor_2d_by_rows(tensor, outlet);
        
    } else {
        // 3D+: processar recursivamente
        int64_t first_dim_size = tensor.size(0);
    
        for (int64_t dim_idx = 0; dim_idx < first_dim_size; ++dim_idx) {
            at::Tensor slice = tensor[dim_idx];
            
            if (slice.dim() == 2) {
                // Chegamos em 2D: enviar linha por linha
                send_tensor_2d_by_rows(slice, outlet);
            } else {
                // Continuar recursão
                send_tensor_recursive(slice, outlet, level + 1);
            }
        }
    }
}

// main function to send structured tensor data
void send_tensor_structured(const at::Tensor& tensor, t_outlet* outlet) {

    send_tensor_shape(tensor, outlet);
    send_tensor_recursive(tensor, outlet, 0);
}

 
} // namespace PdTorchUtils
#ifndef PD_TORCH_UTILS_H
#define PD_TORCH_UTILS_H

#include <torch/torch.h>
#include <string>
#include <vector> // Include if needed for other utils later
#include <map>
#include "pd_torch_types.h"
#include "../../../core/include/core_util_circbuffer.h"
#pragma once // Avoid multiple inclusions


// Use a namespace to avoid polluting the global scope
namespace PdTorchUtils {

// Function to select a device based on a string input
torch::Device select_device(const std::string& device_str, t_pd* pd_obj = nullptr);


// check if DSP is on
bool is_dsp_on();

// void create_circular_buffer(
//     std::vector<std::unique_ptr<contorchionist::core::util_circbuffer::CircularBuffer<float>>>& in_buffers,
//     std::vector<std::unique_ptr<contorchionist::core::util_circbuffer::CircularBuffer<float>>>& out_buffers,
//     int in_channel,
//     int out_channel,
//     int buffer_size
// );

void print_reshape_config(t_torch_reshape *x, const std::string& method, bool verbose);

void send_tensor_shape(const at::Tensor& tensor, t_outlet* outlet);

void send_tensor_structured(const at::Tensor& tensor, t_outlet* outlet);

void send_tensor_1d(const at::Tensor& tensor, t_outlet* outlet);

void send_tensor_2d_by_rows(const at::Tensor& tensor, t_outlet* outlet);

void send_tensor_recursive(const at::Tensor& tensor, t_outlet* outlet, int level = 0);

} // namespace PdTorchUtils

#endif // PD_TORCH_UTILS_H
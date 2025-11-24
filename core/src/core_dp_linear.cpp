#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include "core_dp_linear.h"



namespace contorchionist {
    namespace core {
        namespace dp_linear {


// create a fully connected linear layer
std::shared_ptr<torch::nn::Linear> create_linear_layer(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    const torch::Device& device
) {
    auto layer = std::make_shared<torch::nn::Linear>(
        torch::nn::LinearOptions(in_features, out_features).bias(bias)
    );
    layer->ptr()->to(device);
    return layer;
}


// for processing linear layers
LinearResult LinearProcessor(
    const at::Tensor& input, // input tensor
    const std::shared_ptr<torch::nn::Linear>& layer) {

    LinearResult r;
    if (!layer) {
        r.error_message = "LinearProcessor: layer nullptr";
        return r;
    }
    if (!input.defined()) {
        r.error_message = "LinearProcessor: input tensor undefined";
        return r;
    }

    try {
        auto target_device = (*layer)->weight.device();
        at::Tensor in = input.device() == target_device ? input : input.to(target_device);
        // layer->forward returns Tensor
        r.output = (*layer)->forward(in);
        r.success = true;
    } catch (const c10::Error& e) {
        r.error_message = std::string("LinearProcessor: c10 error: ") + e.what();
    } catch (const std::exception& e) {
        r.error_message = std::string("LinearProcessor: std::exception: ") + e.what();
    }
    return r;
}


        } // namespace dp_linear
    } // namespace core
} // namespace contorchionist
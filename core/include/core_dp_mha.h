#ifndef CONTORCHIONIST_CORE_DP_MHA_H
#define CONTORCHIONIST_CORE_DP_MHA_H

#include <torch/torch.h>
#include <string>
#include <vector>
#include <cstdint>
#include <utility>
#include <functional>
#include "contorchionist_core/contorchionist_core_export.h"

namespace contorchionist {
    namespace core {
        namespace dp_mha {



// wrapper for self-attention (Q=K=V=x) for torch.sequential compatibility (sequential container doesn't acecepts multi-head-attention, only self-attention)
struct MHAWrapperImpl : torch::nn::Module {
    torch::nn::MultiheadAttention mha{nullptr};

    MHAWrapperImpl(int embed_dim, int num_heads, bool bias, bool add_zero_attn, float dropout)
        : mha(torch::nn::MultiheadAttentionOptions(embed_dim, num_heads)
                  .bias(bias)
                  .add_zero_attn(add_zero_attn)
                  .dropout(dropout)) {
        register_module("mha", mha);
    }

    // self-attention forward pass (discards attention weights)
    torch::Tensor forward(const torch::Tensor& x) {
        auto out = mha(x, x, x);
        return std::get<0>(out);
    }
};
TORCH_MODULE(MHAWrapper);


// result of multi-head attention
struct MHAResult {
    bool success = false;
    at::Tensor output;
    at::Tensor attn_weights; // empty if not requested
    std::string error_message;
};

// multi-Head Attention Layer 
struct MHALayer {
    bool success = false;
    std::string error_message;
    torch::nn::MultiheadAttention mha{nullptr};
    MHAWrapper wrapper{nullptr};
};

// parameters for multi-head attention  
struct MHAParams {
    int64_t seq_len = 0;      // if 0, will be inferred from values.size()
    int64_t batch_size = 1; // batch size
    int64_t embed_dim = 1; // embed_dim
    bool flatten_output = true;   // flatten output to 1D
    bool return_weights = false;  // return attn_weights
    float dropout = 0.0; // dropout probability
};




// create multi-head attention layer
CONTORCHIONIST_CORE_EXPORT
MHALayer create_mha_layer(int64_t embed_dim,
                                        int64_t num_heads,
                                        bool bias,
                                        bool add_zero_attn,
                                        float dropout,
                                        const torch::Device& device);




// Build a 3D tensor [seq_len, batch_size, embed_dim] for use in multi-head attention (query)
CONTORCHIONIST_CORE_EXPORT
MHAResult build_query_tensor(const std::vector<float>& values,
                             const MHAParams& params,
                             const torch::Device& device);

// Executes self-attention (Q=K=V=query) using a MultiheadAttention module
CONTORCHIONIST_CORE_EXPORT
MHAResult forward_self_attention(torch::nn::MultiheadAttention& mha,
                                 const at::Tensor& query,
                                 bool return_weights);

// Convenient version that takes values and calls the two stages above.
// If SelfAttention is provided, it is used (self-attention), otherwise the mha module is used.
CONTORCHIONIST_CORE_EXPORT
MHAResult MHAProcessor(const std::vector<float>& values,
                                             const MHAParams& params,
                                             const torch::Device& device,
                                             torch::nn::MultiheadAttention* mha,
                                             std::function<at::Tensor(const at::Tensor&)> SelfAttention = {});



        } // namespace dp_mha
    } // namespace core
} // namespace contorchionist

#endif // CONTORCHIONIST_CORE_DP_MHA_H
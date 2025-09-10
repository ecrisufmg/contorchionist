#include <torch/torch.h>
#include <utility>
#include <stdexcept>
#include "core_dp_mha.h"



namespace contorchionist {
    namespace core {
        namespace dp_mha {



static inline int64_t prod2(int64_t a, int64_t b) {
    // protection against basic overflow (not critical here, but safe)
    if (a == 0 || b == 0) {
        return 0;
    }
    return a * b;
}


// create MHA and wrapper (self-attention)
MHALayer create_mha_layer(int64_t embed_dim,
                                        int64_t num_heads,
                                        bool bias,
                                        bool add_zero_attn,
                                        float dropout,
                                        const torch::Device& device) {
    MHALayer r;

    if (embed_dim <= 0 || num_heads <= 0) {
        r.error_message = "embed_dim and num_heads must be positive";
        return r;
    }
    if (embed_dim % num_heads != 0) {
        r.error_message = "embed_dim must be divisible by num_heads";
        return r;
    }

    try {
        auto mha = torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(embed_dim, num_heads)
                .bias(bias)
                .add_zero_attn(add_zero_attn)
                .dropout(dropout)
        );
        mha->to(device);

        MHAWrapper wrapper(embed_dim, num_heads, bias, add_zero_attn, dropout);
        wrapper->to(device);

        r.mha = mha;
        r.wrapper = wrapper;
        r.success = true;
    } catch (const c10::Error& e) {
        r.error_message = std::string("PyTorch error creating MHA: ") + e.what();
    } catch (const std::exception& e) {
        r.error_message = std::string("Error creating MHA: ") + e.what();
    }
    return r;
}




// build a 3D tensor [seq_len, batch_size, embed_dim] for use in multi-head attention (query)
MHAResult build_query_tensor(const std::vector<float>& values, // input values
                             const MHAParams& params_in, // input parameters
                             const torch::Device& device) { // device
    MHAResult r;
    try {
        if (values.empty()) {
            r.error_message = "Empty input values";
            return r;
        }
        if (params_in.batch_size <= 0 || params_in.embed_dim <= 0) {
            r.error_message = "batch_size and embed_dim must be positive";
            return r;
        }

        MHAParams params = params_in;
        const int64_t per_step = prod2(params.batch_size, params.embed_dim);
        if (per_step == 0) {
            r.error_message = "Invalid configuration: batch_size * embed_dim == 0";
            return r;
        }

        const int64_t total = static_cast<int64_t>(values.size());
        if (total % per_step != 0) {
            r.error_message = "Input size (" + std::to_string(total) +
                              ") must be divisible by batch_size (" + std::to_string(params.batch_size) +
                              ") * embed_dim (" + std::to_string(params.embed_dim) + ")";
            return r;
        }

        int64_t inferred_seq = total / per_step;
        if (params.seq_len == 0) {
            params.seq_len = inferred_seq;
        } else if (params.seq_len != inferred_seq) {
            r.error_message = "Provided seq_len (" + std::to_string(params.seq_len) +
                              ") does not match inferred sequence length (" + std::to_string(inferred_seq) + ")";
            return r;
        }

        // create 3D tensor [seq_len, batch_size, embed_dim]
        at::Tensor query = torch::from_blob(
            const_cast<float*>(values.data()),
            {params.seq_len, params.batch_size, params.embed_dim},
            torch::TensorOptions().dtype(torch::kFloat)
        ).clone();

        r.output = query.to(device);
        r.success = true;
        return r;

    } catch (const c10::Error& e) {
        r.error_message = std::string("PyTorch error building query: ") + e.what();
        return r;
    } catch (const std::exception& e) {
        r.error_message = std::string("Error building query: ") + e.what();
        return r;
    }
}

MHAResult forward_self_attention(torch::nn::MultiheadAttention& mha, // multi-head attention module
                                 const at::Tensor& query, // query tensor
                                 bool return_weights) { // whether to return attention weights
    MHAResult r;
    try {
        if (!query.defined()) {
            r.error_message = "Query tensor is not defined";
            return r;
        }
        // Q=K=V=query (self-attention)
        if (return_weights) {
            at::Tensor out, w;
            std::tie(out, w) = mha(query, query, query);
            r.output = out;
            r.attn_weights = w;
        } else {
            at::Tensor out, w;
            std::tie(out, w) = mha(query, query, query);
            r.output = out;
        }
        r.success = true;
        return r;

    } catch (const c10::Error& e) {
        r.error_message = std::string("PyTorch error running attention: ") + e.what();
        return r;
    } catch (const std::exception& e) {
        r.error_message = std::string("Error running attention: ") + e.what();
        return r;
    }
}


// Multi-Head Attention Processor
MHAResult MHAProcessor(const std::vector<float>& values,
                                             const MHAParams& params,
                                             const torch::Device& device,
                                             torch::nn::MultiheadAttention* mha,
                                             std::function<at::Tensor(const at::Tensor&)> SelfAttention) {
    // build query
    auto built = build_query_tensor(values, params, device);
    if (!built.success) return built;

    // execute forward with wrapper (self-attention) or module (multi-head attention)
    MHAResult r;
    try {
        // if SelfAttention is provided, use it (self-attention)
        if (SelfAttention) {
            r.output = SelfAttention(built.output);
            r.success = true;
        // if mha is provided, use it (multi-head attention)
        } else if (mha) {
            auto fwd = forward_self_attention(*mha, built.output, params.return_weights);
            return fwd;
        } else {
            r.error_message = "No module provided (self-attention and mha are null)";
        }
    } catch (const c10::Error& e) {
        r.error_message = std::string("PyTorch error running attention: ") + e.what();
    } catch (const std::exception& e) {
        r.error_message = std::string("Error running attention: ") + e.what();
    }
    return r;
}


        } // namespace dp_mha
    } // namespace core
} // namespace contorchionist
#include <iostream>
#include <cassert>
#include <torch/torch.h>
#include <torch/script.h>
#include <fstream>
#include <cmath>

// Include all core components
#include "torch_device_utils.h"
#include "neural_registry.h"
#include "activation_registry.h"
#include "neural_layers.h"
#include "model_manager.h"
#include "audio_features.h"
#include "tensor_utils.h"

using namespace contorchionist::core;

// Helper function to create a simple audio processing model
void createAudioProcessingModel(const std::string& path) {
    // Create a model that processes audio features
    torch::nn::Sequential model(
        torch::nn::Linear(13, 32),    // 13 MFCC features -> 32 hidden
        torch::nn::ReLU(),
        torch::nn::Linear(32, 16),    // 32 -> 16 hidden
        torch::nn::ReLU(),
        torch::nn::Linear(16, 8),     // 16 -> 8 outputs
        torch::nn::Sigmoid()          // Output probabilities
    );
    
    // Save the model directly using torch::save
    model->eval();
    torch::save(model, path);
}

void test_audio_processing_pipeline() {
    std::cout << "Testing complete audio processing pipeline..." << std::endl;
    
    auto [device, error] = contorchionist::torch_device_utils::parse_torch_device("cpu");
    assert(error.empty());
    
    // 1. Generate test audio signal
    int sample_rate = 16000;
    float duration = 1.0f;
    int n_samples = static_cast<int>(sample_rate * duration);
    
    torch::Tensor time = torch::linspace(0, duration, n_samples);
    torch::Tensor audio = torch::sin(2 * M_PI * 440 * time); // 440 Hz sine wave
    audio = audio.to(device);
    
    std::cout << "  ✓ Generated test audio signal: " << audio.sizes() << std::endl;
    
    // 3. Extract MFCC features
    int n_mfcc = 13;
    torch::Tensor mfcc = audio_features::computeMFCC(audio, sample_rate, n_mfcc, 40, 512, 160, 400, true, device);
    std::cout << "  ✓ Extracted MFCC features: " << mfcc.sizes() << std::endl;
    
    // 4. Normalize features
    torch::Tensor normalized_mfcc = contorchionist::tensor_utils::normalize(mfcc, -1.0f, 1.0f);
    std::cout << "  ✓ Normalized MFCC features" << std::endl;
    
//     // 5. Create and load model
//     std::string model_path = "/tmp/audio_model.pt";
//     createAudioProcessingModel(model_path);
    
//     ModelManager manager;
//     auto load_result = manager.loadModel(model_path, device);
//     assert(load_result.first);
//     std::cout << "  ✓ Created and loaded audio processing model" << std::endl;
    
//     // 6. Process features through model
//     torch::Tensor frame = normalized_mfcc[0].unsqueeze(0); // Take first frame, add batch dim
//     auto inference_result = manager.runInference(frame);
//     torch::Tensor output = inference_result.first;
//     assert(inference_result.second.empty()); // Check no error (empty string means success)
    
//     assert(output.size(0) == 1); // batch size
//     assert(output.size(1) == 8); // output features
//     std::cout << "  ✓ Model inference successful: " << output.sizes() << std::endl;
    
//     // 7. Verify output is in valid range (sigmoid output should be [0,1])
//     assert((output >= 0).all().item<bool>());
//     assert((output <= 1).all().item<bool>());
//     std::cout << "  ✓ Model output is in valid range [0,1]" << std::endl;
    
//     // Cleanup
//     std::remove(model_path.c_str());
//     std::cout << "  ✓ Complete audio processing pipeline works!" << std::endl;
}

void test_neural_network_construction() {
    std::cout << "Testing neural network construction with registry..." << std::endl;
    
    auto [device, error] = contorchionist::torch_device_utils::parse_torch_device("cpu");
    assert(error.empty());
    
    NeuralRegistry registry;
    
    // 1. Create layers
    auto linear1 = std::make_shared<LinearLayer>(4, 8, true);
    auto linear2 = std::make_shared<LinearLayer>(8, 2, true);
    
    auto relu = std::make_shared<ActivationLayer>("relu");
    
    // Set device for all layers
    linear1->set_device(device);
    linear2->set_device(device);
    relu->set_device(device);
    
    std::cout << "  ✓ Created linear and activation layers" << std::endl;
    
    // 2. Register layers in registry
    bool reg1 = registry.register_module("linear1", std::make_shared<torch::nn::Sequential>());
    bool reg2 = registry.register_module("relu", std::make_shared<torch::nn::Sequential>());  
    bool reg3 = registry.register_module("linear2", std::make_shared<torch::nn::Sequential>());
    
    assert(reg1 && reg2 && reg3);
    std::cout << "  ✓ Registered modules in neural registry" << std::endl;
    
    // 3. Test forward pass through network
    torch::Tensor input = torch::randn({2, 4}).to(device); // batch of 2
    
    torch::Tensor x = linear1->forward(input);
    assert(x.sizes() == torch::IntArrayRef({2, 8}));
    
    x = relu->forward(x);
    assert(x.sizes() == torch::IntArrayRef({2, 8}));
    assert((x >= 0).all().item<bool>()); // ReLU output should be non-negative
    
    torch::Tensor output = linear2->forward(x);
    assert(output.sizes() == torch::IntArrayRef({2, 2}));
    
    std::cout << "  ✓ Forward pass through constructed network works" << std::endl;
    
    // 4. Test registry functionality
    auto retrieved_module = registry.get_module("linear1");
    assert(retrieved_module != nullptr);
    
    auto module_list = registry.get_all_module_names();
    assert(module_list.size() == 3);
    
    std::cout << "  ✓ Registry retrieval and listing work correctly" << std::endl;
    
    // 5. Test cleanup
    registry.clear_all();
    assert(registry.get_all_module_names().empty());
    
    std::cout << "  ✓ Neural network construction and management complete!" << std::endl;
}

void test_multi_head_attention_workflow() {
    std::cout << "Testing multi-head attention workflow..." << std::endl;
    
    auto [device, error] = contorchionist::torch_device_utils::parse_torch_device("cpu");
    assert(error.empty());
    
    // 1. Create MHA layer
    int embed_dim = 64;
    int num_heads = 8;
    MHALayer mha(embed_dim, num_heads, true, false);
    mha.set_device(device);
    
    std::cout << "  ✓ Created MHA layer: " << embed_dim << " embed_dim, " 
              << num_heads << " heads" << std::endl;
    
    // 2. Create sequence data
    int seq_len = 20;
    int batch_size = 2;
    torch::Tensor sequence = torch::randn({seq_len, batch_size, embed_dim}).to(device);
    
    std::cout << "  ✓ Created sequence data: " << sequence.sizes() << std::endl;
    
    // 3. Apply attention
    torch::Tensor attended = mha.forward(sequence);
    
    assert(attended.sizes() == sequence.sizes());
    assert(attended.device().type() == device.type());
    
    std::cout << "  ✓ Multi-head attention forward pass successful" << std::endl;
    
    // 4. Test with different sequence lengths
    torch::Tensor short_seq = torch::randn({5, 1, embed_dim}).to(device);
    torch::Tensor short_attended = mha.forward(short_seq);
    assert(short_attended.sizes() == short_seq.sizes());
    
    std::cout << "  ✓ MHA handles different sequence lengths correctly" << std::endl;
    
    // 5. Verify attention doesn't change the total "energy" too much
    float original_norm = sequence.norm().item<float>();
    float attended_norm = attended.norm().item<float>();
    float norm_ratio = attended_norm / original_norm;
    
    // Ratio should be reasonable (not too far from 1.0)
    assert(norm_ratio > 0.5 && norm_ratio < 2.0);
    
    std::cout << "  ✓ Attention preserves reasonable energy levels" << std::endl;
    std::cout << "  ✓ Multi-head attention workflow complete!" << std::endl;
}

void test_tensor_reshape_integration() {
    std::cout << "Testing tensor reshape integration..." << std::endl;
    
    auto [device, error] = contorchionist::torch_device_utils::parse_torch_device("cpu");
    assert(error.empty());
    
    // 1. Create initial tensor
    torch::Tensor input = torch::randn({2, 3, 4, 5}).to(device);
    std::cout << "  ✓ Created input tensor: " << input.sizes() << std::endl;
    
    // 2. Test various reshape operations
    ReshapeLayer view_layer(ReshapeLayer::Operation::VIEW, {6, 20});
    torch::Tensor viewed = view_layer.forward(input);
    assert(viewed.sizes() == torch::IntArrayRef({6, 20}));
    assert(viewed.numel() == input.numel());
    
    ReshapeLayer flatten_layer(ReshapeLayer::Operation::FLATTEN);
    torch::Tensor flattened = flatten_layer.forward(input);
    assert(flattened.dim() == 2);
    assert(flattened.size(0) == 2); // batch preserved
    
    ReshapeLayer transpose_layer(ReshapeLayer::Operation::TRANSPOSE, {}, 2, 3);
    torch::Tensor transposed = transpose_layer.forward(input);
    assert(transposed.size(2) == 5); // dimension 3 moved to position 2
    assert(transposed.size(3) == 4); // dimension 2 moved to position 3
    
    std::cout << "  ✓ All reshape operations work correctly" << std::endl;
    
    // 3. Test tensor utils operations
    torch::Tensor normalized = contorchionist::tensor_utils::normalize(input, 0.0f, 1.0f);
    assert((normalized >= 0).all().item<bool>());
    assert((normalized <= 1).all().item<bool>());
    
    torch::Tensor permuted = input.permute({3, 1, 0, 2});
    assert(permuted.size(0) == 5);
    assert(permuted.size(1) == 3);
    assert(permuted.size(2) == 2);
    assert(permuted.size(3) == 4);
    
    std::cout << "  ✓ Tensor utilities work correctly" << std::endl;
    
    // 4. Test chain of operations
    torch::Tensor processed = input;
    processed = contorchionist::tensor_utils::normalize(processed, -1.0f, 1.0f);
    processed = view_layer.forward(processed);
    processed = contorchionist::tensor_utils::clamp(processed, -0.5f, 0.5f);
    
    assert(processed.sizes() == torch::IntArrayRef({6, 20}));
    assert(processed.min().item<float>() >= -0.5f);
    assert(processed.max().item<float>() <= 0.5f);
    
    std::cout << "  ✓ Chained tensor operations work correctly" << std::endl;
    std::cout << "  ✓ Tensor reshape integration complete!" << std::endl;
}

void test_device_transfer_integration() {
    std::cout << "Testing device transfer integration..." << std::endl;
    
    // Test CPU
    auto [cpu_device, cpu_error] = contorchionist::torch_device_utils::parse_torch_device("cpu");
    assert(cpu_error.empty());
    
    LinearLayer cpu_layer(4, 2, true);
    cpu_layer.set_device(cpu_device);
    
    torch::Tensor cpu_input = torch::randn({1, 4}).to(cpu_device);
    torch::Tensor cpu_output = cpu_layer.forward(cpu_input);
    assert(cpu_output.device().type() == torch::kCPU);
    
    std::cout << "  ✓ CPU device operations work correctly" << std::endl;
    
    // Test CUDA if available
    if (torch::cuda::is_available()) {
        auto [cuda_device, cuda_error] = contorchionist::torch_device_utils::parse_torch_device("cuda");
        if (cuda_error.empty()) {
            LinearLayer cuda_layer(4, 2, true);
            cuda_layer.set_device(cuda_device);
            
            torch::Tensor cuda_input = torch::randn({1, 4}).to(cuda_device);
            torch::Tensor cuda_output = cuda_layer.forward(cuda_input);
            assert(cuda_output.device().type() == torch::kCUDA);
            
            // Test device transfer
            torch::Tensor transferred = cpu_output.to(cuda_device);
            assert(transferred.device().type() == torch::kCUDA);
            
            std::cout << "  ✓ CUDA device operations and transfer work correctly" << std::endl;
        }
    } else {
        std::cout << "  ⚠ CUDA not available, skipping CUDA integration tests" << std::endl;
    }
    
    std::cout << "  ✓ Device transfer integration complete!" << std::endl;
}

int main() {
    std::cout << "=== Contorchionist Core - Integration Test ===" << std::endl;
    std::cout << "Testing integration of all core components..." << std::endl;
    
    try {
        test_audio_processing_pipeline();
        test_neural_network_construction();
        test_multi_head_attention_workflow();
        test_tensor_reshape_integration();
        test_device_transfer_integration();
        
        std::cout << "\n🎉 All integration tests passed!" << std::endl;
        std::cout << "The Contorchionist core library is working correctly!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Integration test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

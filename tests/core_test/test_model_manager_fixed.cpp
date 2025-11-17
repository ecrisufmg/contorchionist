#include <iostream>
#include <cassert>
#include <torch/torch.h>
#include <fstream>
#include "model_manager.h"

using namespace contorchionist;

// Helper function to create a test model file
void createTestModel(const std::string& path) {
    // Create a simple linear model
    torch::nn::Sequential model(
        torch::nn::Linear(10, 5),
        torch::nn::ReLU(),
        torch::nn::Linear(5, 2)
    );
    
    // Convert to TorchScript
    torch::jit::script::Module script_module = torch::jit::script::compile(model);
    
    // Save the model
    script_module.save(path);
}

void test_model_loading() {
    std::cout << "Testing model loading..." << std::endl;
    
    ModelManager manager;
    std::string model_path = "/tmp/test_model.pt";
    torch::Device device = torch::kCPU;
    
    // Create test model
    createTestModel(model_path);
    
    // Test successful loading
    auto [loaded, error] = manager.loadModel(model_path, device);
    assert(loaded);
    assert(error.empty());
    assert(manager.isModelLoaded());
    std::cout << "  ✓ Model loading succeeded" << std::endl;
    
    // Test loading non-existent model
    ModelManager failed_manager;
    auto [failed_load, failed_error] = failed_manager.loadModel("/tmp/nonexistent_model.pt", device);
    assert(!failed_load);
    assert(!failed_error.empty());
    assert(!failed_manager.isModelLoaded());
    std::cout << "  ✓ Non-existent model loading correctly failed" << std::endl;
    
    // Clean up
    std::remove(model_path.c_str());
}

void test_model_inference() {
    std::cout << "Testing model inference..." << std::endl;
    
    ModelManager manager;
    std::string model_path = "/tmp/test_inference_model.pt";
    torch::Device device = torch::kCPU;
    
    // Create and load test model
    createTestModel(model_path);
    auto [loaded, error] = manager.loadModel(model_path, device);
    assert(loaded);
    
    // Test inference
    torch::Tensor input = torch::randn({1, 10});
    auto [output, inference_error] = manager.runInference(input);
    assert(inference_error.empty());
    assert(output.defined());
    assert(output.size(0) == 1);
    assert(output.size(1) == 2);
    std::cout << "  ✓ Model inference works correctly" << std::endl;
    
    // Test inference with unloaded model
    ModelManager empty_manager;
    auto [empty_output, empty_error] = empty_manager.runInference(input);
    assert(!empty_error.empty());
    assert(!empty_output.defined());
    std::cout << "  ✓ Inference with unloaded model correctly fails" << std::endl;
    
    // Clean up
    std::remove(model_path.c_str());
}

void test_model_introspection() {
    std::cout << "Testing model introspection..." << std::endl;
    
    ModelManager manager;
    std::string model_path = "/tmp/test_introspection_model.pt";
    torch::Device device = torch::kCPU;
    
    // Create and load test model
    createTestModel(model_path);
    auto [loaded, error] = manager.loadModel(model_path, device);
    assert(loaded);
    
    // Test device query
    assert(manager.getDevice() == device);
    std::cout << "  ✓ Device query works correctly" << std::endl;
    
    // Test path query
    assert(manager.getModelPath() == model_path);
    std::cout << "  ✓ Model path query works correctly" << std::endl;
    
    // Test method listing (should at least have 'forward')
    auto methods = manager.getAvailableMethods();
    bool has_forward = false;
    for (const auto& method : methods) {
        if (method == "forward") {
            has_forward = true;
            break;
        }
    }
    assert(has_forward);
    std::cout << "  ✓ Method introspection works correctly" << std::endl;
    
    // Clean up
    std::remove(model_path.c_str());
}

void test_model_shape_analysis() {
    std::cout << "Testing model shape analysis..." << std::endl;
    
    ModelManager manager;
    std::string model_path = "/tmp/test_shape_model.pt";
    torch::Device device = torch::kCPU;
    
    // Create and load test model
    createTestModel(model_path);
    auto [loaded, error] = manager.loadModel(model_path, device);
    assert(loaded);
    
    // Test with single input
    torch::Tensor test_input = torch::randn({1, 10});
    auto [output, inference_error] = manager.runInference(test_input);
    assert(inference_error.empty());
    assert(output.size(0) == 1);
    assert(output.size(1) == 2);
    std::cout << "  ✓ Single input shape handling works correctly" << std::endl;
    
    // Test with batch input
    torch::Tensor batch_input = torch::randn({4, 10});
    auto [batch_output, batch_error] = manager.runInference(batch_input);
    assert(batch_error.empty());
    assert(batch_output.size(0) == 4);
    assert(batch_output.size(1) == 2);
    std::cout << "  ✓ Batch input shape handling works correctly" << std::endl;
    
    // Clean up
    std::remove(model_path.c_str());
}

void test_model_device_transfer() {
    std::cout << "Testing model device transfer..." << std::endl;
    
    ModelManager manager;
    std::string model_path = "/tmp/test_device_model.pt";
    torch::Device cpu_device = torch::kCPU;
    
    // Create and load test model on CPU
    createTestModel(model_path);
    auto [loaded, error] = manager.loadModel(model_path, cpu_device);
    assert(loaded);
    assert(manager.getDevice() == cpu_device);
    
    // Test inference on CPU
    torch::Tensor cpu_input = torch::randn({2, 10}).to(cpu_device);
    auto [cpu_output, cpu_error] = manager.runInference(cpu_input);
    assert(cpu_error.empty());
    std::cout << "  ✓ CPU device works correctly" << std::endl;
    
    // Test CUDA if available
    if (torch::cuda::is_available()) {
        try {
            torch::Device cuda_device = torch::kCUDA;
            auto [cuda_loaded, cuda_error] = manager.loadModel(model_path, cuda_device);
            assert(cuda_loaded);
            assert(manager.getDevice() == cuda_device);
            
            torch::Tensor cuda_input = torch::randn({2, 10}).to(cuda_device);
            auto [cuda_output, cuda_inference_error] = manager.runInference(cuda_input);
            assert(cuda_inference_error.empty());
            std::cout << "  ✓ CUDA device works correctly" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  ! CUDA test skipped: " << e.what() << std::endl;
        }
    } else {
        std::cout << "  ! CUDA not available, skipping CUDA tests" << std::endl;
    }
    
    // Clean up
    std::remove(model_path.c_str());
}

void test_multiple_models() {
    std::cout << "Testing multiple model managers..." << std::endl;
    
    std::string model1_path = "/tmp/test_model1.pt";
    std::string model2_path = "/tmp/test_model2.pt";
    torch::Device device = torch::kCPU;
    
    // Create test models
    createTestModel(model1_path);
    createTestModel(model2_path);
    
    // Test multiple managers
    ModelManager manager1, manager2;
    auto [loaded1, error1] = manager1.loadModel(model1_path, device);
    auto [loaded2, error2] = manager2.loadModel(model2_path, device);
    
    assert(loaded1 && loaded2);
    assert(manager1.isModelLoaded() && manager2.isModelLoaded());
    assert(manager1.getModelPath() == model1_path);
    assert(manager2.getModelPath() == model2_path);
    std::cout << "  ✓ Multiple model managers work independently" << std::endl;
    
    // Test inference with both
    torch::Tensor input = torch::randn({1, 10});
    auto [output1, inference_error1] = manager1.runInference(input);
    auto [output2, inference_error2] = manager2.runInference(input);
    
    assert(inference_error1.empty() && inference_error2.empty());
    assert(output1.defined() && output2.defined());
    std::cout << "  ✓ Independent inference works correctly" << std::endl;
    
    // Clean up
    std::remove(model1_path.c_str());
    std::remove(model2_path.c_str());
}

int main() {
    std::cout << "=== Contorchionist Core - Model Manager Test ===" << std::endl;
    
    try {
        test_model_loading();
        test_model_inference();
        test_model_introspection();
        test_model_shape_analysis();
        test_model_device_transfer();
        test_multiple_models();
        
        std::cout << "\n✅ All model manager tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

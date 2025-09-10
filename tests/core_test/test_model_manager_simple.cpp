#include <iostream>
#include <cassert>
#include <fstream>
#include <torch/torch.h>
#include <torch/script.h>
#include "model_manager.h"
#include "torch_device_utils.h"

using namespace contorchionist;

// Helper function to copy the pre-created test model
void createTestModel(const std::string& path) {
    std::cout << "Creating test TorchScript model..." << std::endl;
    
    // Copy the pre-created model file
    std::string source_model = "simple_test_model.pt";
    std::ifstream src(source_model, std::ios::binary);
    std::ofstream dst(path, std::ios::binary);
    
    if (!src.is_open()) {
        throw std::runtime_error("Could not find test model file: " + source_model);
    }
    
    dst << src.rdbuf();
    src.close();
    dst.close();
    
    std::cout << "  ✓ Test model copied to: " << path << std::endl;
}

void test_model_loading() {
    std::cout << "Testing model loading..." << std::endl;
    
    // Create a test model file
    std::string model_path = "/tmp/test_model.pt";
    createTestModel(model_path);
    
    auto [device, error] = contorchionist::torch_device_utils::parse_torch_device("cpu");
    assert(error.empty());
    
    ModelManager manager;
    
    // Test successful loading
    auto [loaded, load_error] = manager.loadModel(model_path, device);
    assert(loaded);
    assert(load_error.empty());
    assert(manager.isModelLoaded());
    
    std::cout << "  ✓ Model loaded successfully" << std::endl;
    
    // Test loading non-existent file
    auto [failed_load, fail_error] = manager.loadModel("/tmp/nonexistent_model.pt", device);
    assert(!failed_load);
    assert(!fail_error.empty());
    
    std::cout << "  ✓ Non-existent model loading fails correctly" << std::endl;
    
    // Cleanup
    std::remove(model_path.c_str());
}

void test_model_inference() {
    std::cout << "Testing model inference..." << std::endl;
    
    // Create and load test model
    std::string model_path = "/tmp/test_inference_model.pt";
    createTestModel(model_path);
    
    auto [device, error] = contorchionist::torch_device_utils::parse_torch_device("cpu");
    assert(error.empty());
    
    ModelManager manager;
    auto [loaded, load_error] = manager.loadModel(model_path, device);
    assert(loaded);
    assert(load_error.empty());
    
    // Test inference
    torch::Tensor input = torch::randn({2, 4}); // batch size 2, 4 features
    auto [output, inference_error] = manager.runInference(input);
    
    assert(inference_error.empty());
    assert(output.size(0) == 2); // batch size preserved
    assert(output.size(1) == 2); // output features
    assert(output.device().type() == device.type()); // correct device
    
    std::cout << "  ✓ Model inference works correctly" << std::endl;
    
    // Test inference without loaded model
    ModelManager empty_manager;
    auto [empty_output, empty_error] = empty_manager.runInference(input);
    assert(!empty_error.empty()); // should return error
    
    std::cout << "  ✓ Inference without model returns error correctly" << std::endl;
    
    // Cleanup
    std::remove(model_path.c_str());
}

int main() {
    std::cout << "=== Contorchionist Core - Model Manager Test ===" << std::endl;
    
    try {
        test_model_loading();
        test_model_inference();
        
        std::cout << "\n✅ All model manager tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

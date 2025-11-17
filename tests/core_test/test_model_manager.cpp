#include <iostream>
#include <cassert>
#include <fstream>
#include <torch/torch.h>
#include <torch/script.h>
#include "../../core/include/model_manager.h" 
#include "../../core/include/torch_device_utils.h" 

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
    
    auto device_result = torch_device_utils::parse_torch_device("cpu");
    assert(device_result.first.is_cpu()); // Verifica se o device é CPU
    auto device = device_result.first; // O device está no .first do pair
    
    // Crie uma instância do MODELManager
    std::shared_ptr<MODELManager> manager = std::make_shared<MODELManager>();
    
    // Teste de carregamento bem-sucedido
    bool load_success = manager->loadModel(model_path, device);
    assert(load_success);
    assert(manager->isModelLoaded());
    
    std::cout << "  ✓ Model loaded successfully" << std::endl;
    
    // Teste de carregamento de arquivo inexistente
    bool failed_load = manager->loadModel("/tmp/nonexistent_model.pt", device);
    assert(!failed_load); // Deve falhar com arquivo inexistente
    
    std::cout << "  ✓ Non-existent model loading fails correctly" << std::endl;
    
    // Cleanup
    std::remove(model_path.c_str());
}

void test_model_inference() {
    std::cout << "Testing model inference..." << std::endl;
    
    // Create and load test model
    std::string model_path = "/tmp/test_inference_model.pt";
    createTestModel(model_path);
    
    auto device_result = torch_device_utils::parse_torch_device("cpu");
    auto device = device_result.first; // O device está no .first do pair
    
    std::shared_ptr<MODELManager> manager = std::make_shared<MODELManager>();
    bool load_success = manager->loadModel(model_path, device);
    assert(load_success);
    
    // Test inference
    torch::Tensor input = torch::randn({2, 4}); // batch size 2, 4 features
    std::vector<torch::Tensor> inputs = {input};
    
    auto outputs = manager->forward(inputs);
    assert(!outputs.empty());
    
    // Verificações específicas do tensor de saída
    assert(outputs[0].size(0) == 2); // batch size preservado
    
    std::cout << "  ✓ Model inference works correctly" << std::endl;
    
    // Teste de inferência sem modelo carregado
    std::shared_ptr<MODELManager> empty_manager = std::make_shared<MODELManager>();
    try {
        auto empty_result = empty_manager->forward(inputs);
        assert(false); // Deve falhar antes daqui
    } catch (...) {
        std::cout << "  ✓ Inference without model throws exception correctly" << std::endl;
    }
    
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
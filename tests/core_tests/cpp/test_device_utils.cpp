#include <iostream>
#include <cassert>
#include <torch/torch.h>
#include "torch_device_utils.h"

using namespace contorchionist::torch_device_utils;

void test_cpu_device() {
    std::cout << "Testing CPU device selection..." << std::endl;
    
    auto [device, error] = parse_torch_device("cpu");
    
    assert(error.empty());
    assert(device.type() == torch::kCPU);
    std::cout << "  ✓ CPU device: " << device << std::endl;
}

void test_cuda_device() {
    std::cout << "Testing CUDA device selection..." << std::endl;
    
    auto [device, error] = parse_torch_device("cuda");
    
    if (torch::cuda::is_available()) {
        assert(error.empty());
        assert(device.type() == torch::kCUDA);
        std::cout << "  ✓ CUDA device: " << device << std::endl;
    } else {
        assert(!error.empty());
        assert(device.type() == torch::kCPU);
        std::cout << "  ✓ CUDA not available, fallback to CPU: " << device << std::endl;
        std::cout << "    Error: " << error << std::endl;
    }
}

void test_cuda_with_index() {
    std::cout << "Testing CUDA device with index..." << std::endl;
    
    auto [device, error] = parse_torch_device("cuda:0");
    
    if (torch::cuda::is_available()) {
        assert(error.empty());
        assert(device.type() == torch::kCUDA);
        assert(device.index() == 0);
        std::cout << "  ✓ CUDA:0 device: " << device << std::endl;
    } else {
        assert(!error.empty());
        assert(device.type() == torch::kCPU);
        std::cout << "  ✓ CUDA:0 not available, fallback to CPU: " << device << std::endl;
        std::cout << "    Error: " << error << std::endl;
    }
}

void test_invalid_device() {
    std::cout << "Testing invalid device string..." << std::endl;
    
    auto [device, error] = parse_torch_device("invalid_device");
    
    assert(!error.empty());
    assert(device.type() == torch::kCPU);
    std::cout << "  ✓ Invalid device fallback to CPU: " << device << std::endl;
    std::cout << "    Error: " << error << std::endl;
}

void test_mps_device() {
    std::cout << "Testing MPS device selection..." << std::endl;
    
    auto [device, error] = parse_torch_device("mps");
    
    if (torch::mps::is_available()) {
        assert(error.empty());
        assert(device.type() == torch::kMPS);
        std::cout << "  ✓ MPS device: " << device << std::endl;
    } else {
        assert(!error.empty());
        assert(device.type() == torch::kCPU);
        std::cout << "  ✓ MPS not available, fallback to CPU: " << device << std::endl;
        std::cout << "    Error: " << error << std::endl;
    }
}

int main() {
    std::cout << "=== Contorchionist Core - Device Utils Test ===" << std::endl;
    
    try {
        test_cpu_device();
        test_cuda_device();
        test_cuda_with_index();
        test_invalid_device();
        test_mps_device();
        
        std::cout << "\n✅ All device utils tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

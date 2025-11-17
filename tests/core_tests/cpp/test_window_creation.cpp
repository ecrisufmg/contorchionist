#include <gtest/gtest.h>
#include <torch/torch.h>
#include "torchwins.h"
#include <iostream>

class WindowCreationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Always test CPU first
        test_devices.push_back(torch::kCPU);
        std::cout << "CPU device added to test suite" << std::endl;
        
        // Auto-detect and add CUDA devices if available
        if (torch::cuda::is_available()) {
            int cuda_device_count = torch::cuda::device_count();
            std::cout << "CUDA is available with " << cuda_device_count << " device(s)" << std::endl;
            
            for (int i = 0; i < cuda_device_count; ++i) {
                torch::Device cuda_device(torch::kCUDA, i);
                test_devices.push_back(cuda_device);
                std::cout << "CUDA device " << i << " added to test suite" << std::endl;
            }
        } else {
            std::cout << "CUDA is not available - skipping CUDA tests" << std::endl;
        }
        
        // Auto-detect and add MPS device if available (Apple Silicon)
        if (torch::mps::is_available()) {
            test_devices.push_back(torch::kMPS);
            std::cout << "MPS device available and added to test suite" << std::endl;
        } else {
            std::cout << "MPS is not available - skipping MPS tests" << std::endl;
        }
        
        // Try to detect other potential devices (future-proofing)
        // Note: This is a basic approach - more sophisticated detection could be added
        
        std::cout << "Total devices available for testing: " << test_devices.size() << std::endl;
        for (size_t i = 0; i < test_devices.size(); ++i) {
            std::cout << "  Device " << i << ": " << test_devices[i] << std::endl;
        }
    }
    
    // Helper to get device-friendly name for output
    std::string device_name(const torch::Device& device) {
        switch (device.type()) {
            case torch::kCPU: 
                return "CPU";
            case torch::kCUDA: 
                return "CUDA:" + std::to_string(device.index());
            case torch::kMPS: 
                return "MPS";
            default: 
                return "Device(" + std::to_string(static_cast<int>(device.type())) + 
                       ":" + std::to_string(device.index()) + ")";
        }
    }
    
    std::vector<torch::Device> test_devices;
    
    // Test parameters
    const int window_size = 1024;
    const std::vector<std::string> window_types = {
        "hann", "hamming", "blackman", "bartlett", "rectangular",
        "cosine", "boxcar", "triang", "parzen", "bohman", 
        "nuttall", "blackmanharris", "flattop", "barthann", "lanczos"
    };
    
    // Helper function to create a window using TorchWindowing functions
    torch::Tensor create_window(const std::string& window_type, int size, const torch::Device& device, bool periodic = false) {
        auto type = TorchWindowing::string_to_torch_window_type(window_type);
        // Explicitly use float32 for compatibility across devices
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        return TorchWindowing::generate_torch_window(size, type, periodic, options);
    }
};

TEST_F(WindowCreationTest, PeriodicVsSymmetricComparison) {
    for (const auto& device : test_devices) {
        std::cout << "\n=== Testing Periodic vs Symmetric Windows on device: " << device_name(device) << " ===" << std::endl;
        std::cout << "Window Size: " << window_size << std::endl;
        
        for (const auto& window_type : window_types) {
            std::cout << "\n--- " << window_type << " Window ---" << std::endl;
            
            // Create both symmetric and periodic versions
            auto window_sym = create_window(window_type, window_size, device, false);  // periodic=false (symmetric)
            auto window_per = create_window(window_type, window_size, device, true);   // periodic=true
            
            // Basic checks for both
            ASSERT_TRUE(window_sym.defined() && window_per.defined()) 
                << "Both window modes should be defined for " << window_type << " on " << device;
            EXPECT_EQ(window_sym.size(0), window_size) << "Symmetric window size mismatch";
            EXPECT_EQ(window_per.size(0), window_size) << "Periodic window size mismatch";
            
            // Calculate and compare sums
            auto sum_sym = window_sym.sum().item<float>();
            auto sum_per = window_per.sum().item<float>();
            
            std::cout << "Symmetric (periodic=false) sum: " << std::fixed << std::setprecision(6) << sum_sym << std::endl;
            std::cout << "Periodic  (periodic=true)  sum: " << std::fixed << std::setprecision(6) << sum_per << std::endl;
            std::cout << "Difference (periodic - symmetric): " << std::fixed << std::setprecision(6) << (sum_per - sum_sym) << std::endl;
            
            // Value ranges
            auto min_sym = torch::min(window_sym).item<float>();
            auto max_sym = torch::max(window_sym).item<float>();
            auto min_per = torch::min(window_per).item<float>();
            auto max_per = torch::max(window_per).item<float>();
            
            std::cout << "Symmetric range: [" << std::fixed << std::setprecision(6) 
                      << min_sym << ", " << max_sym << "]" << std::endl;
            std::cout << "Periodic  range: [" << std::fixed << std::setprecision(6) 
                      << min_per << ", " << max_per << "]" << std::endl;
            
            // Check endpoint behavior for most windows (except rectangular/boxcar)
            if (window_type != "rectangular" && window_type != "boxcar") {
                std::cout << "Symmetric endpoints: [" << std::fixed << std::setprecision(6) 
                          << window_sym[0].item<float>() << ", " 
                          << window_sym[-1].item<float>() << "]" << std::endl;
                std::cout << "Periodic  endpoints: [" << std::fixed << std::setprecision(6) 
                          << window_per[0].item<float>() << ", " 
                          << window_per[-1].item<float>() << "]" << std::endl;
            }
            
            // First and last few values comparison
            std::cout << "Symmetric first 3: [";
            for (int i = 0; i < std::min(3, static_cast<int>(window_sym.size(0))); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(6) << window_sym[i].item<float>();
            }
            std::cout << "] last 3: [";
            for (int i = std::max(0, static_cast<int>(window_sym.size(0)) - 3); i < window_sym.size(0); ++i) {
                if (i > std::max(0, static_cast<int>(window_sym.size(0)) - 3)) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(6) << window_sym[i].item<float>();
            }
            std::cout << "]" << std::endl;
            
            std::cout << "Periodic  first 3: [";
            for (int i = 0; i < std::min(3, static_cast<int>(window_per.size(0))); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(6) << window_per[i].item<float>();
            }
            std::cout << "] last 3: [";
            for (int i = std::max(0, static_cast<int>(window_per.size(0)) - 3); i < window_per.size(0); ++i) {
                if (i > std::max(0, static_cast<int>(window_per.size(0)) - 3)) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(6) << window_per[i].item<float>();
            }
            std::cout << "]" << std::endl;
            
            // Both should be positive sums for most windows
            EXPECT_GT(sum_sym, 0.0f) << "Symmetric window sum should be positive for " << window_type;
            EXPECT_GT(sum_per, 0.0f) << "Periodic window sum should be positive for " << window_type;
            
            // For rectangular and boxcar windows, sums should be identical
            if (window_type == "rectangular" || window_type == "boxcar") {
                EXPECT_FLOAT_EQ(sum_sym, sum_per) << "Rectangular/boxcar sums should be identical";
            }
            
            std::cout << "✓ " << window_type << " periodic vs symmetric comparison completed" << std::endl;
        }
        
        std::cout << "\nPeriodic vs Symmetric comparison completed on " << device_name(device) << "!" << std::endl;
    }
}

TEST_F(WindowCreationTest, BasicWindowCreation) {
    for (const auto& device : test_devices) {
        std::cout << "\nTesting on device: " << device_name(device) << std::endl;
        std::cout << "Window Size: " << window_size << std::endl;
        std::cout << "Available Window Types: ";
        for (const auto& type : window_types) {
            std::cout << type << " ";
        }
        std::cout << "\n" << std::endl;
        
        for (const auto& window_type : window_types) {
            std::cout << "=== Testing window type: " << window_type << " ===" << std::endl;
            
            // Create window
            auto window = create_window(window_type, window_size, device, false);  // symmetric mode
            
            // Basic checks
            ASSERT_TRUE(window.defined()) << "Window should be defined for " << window_type << " on " << device;
            EXPECT_EQ(window.size(0), window_size) << "Window size mismatch for " << window_type << " on " << device;
            // Allow device index differences (e.g., mps vs mps:0)
            EXPECT_EQ(window.device().type(), device.type()) << "Device type mismatch for " << window_type << " on " << device;
            // Some advanced windows might return double, which is acceptable
            EXPECT_TRUE(window.dtype() == torch::kFloat32 || window.dtype() == torch::kFloat64) 
                << "Dtype should be float32 or float64 for " << window_type << " on " << device 
                << " (got " << window.dtype() << ")";
            
            // Calculate and print window sum
            auto window_sum = window.sum().item<float>();
            std::cout << "Window sum: " << std::fixed << std::setprecision(6) << window_sum << std::endl;
            
            // Check that window values are reasonable (between 0 and 1 for most windows)
            auto min_val = torch::min(window).item<float>();
            auto max_val = torch::max(window).item<float>();
            
            std::cout << "Value range: [" << std::fixed << std::setprecision(6) 
                      << min_val << ", " << max_val << "]" << std::endl;
            
            if (window_type == "rectangular" || window_type == "boxcar") {
                EXPECT_FLOAT_EQ(min_val, 1.0f) << "Rectangular/boxcar window should have min value 1.0";
                EXPECT_FLOAT_EQ(max_val, 1.0f) << "Rectangular/boxcar window should have max value 1.0";
            } else if (window_type == "flattop") {
                // Flattop windows can legitimately go negative
                EXPECT_LE(max_val, 1.0f) << "Window values should not exceed 1.0 for " << window_type;
                EXPECT_GT(max_val, min_val) << "Window should have variation for " << window_type;
            } else {
                // Allow small negative values due to floating point precision
                EXPECT_GE(min_val, -1e-6f) << "Window values should be approximately non-negative for " << window_type;
                EXPECT_LE(max_val, 1.0f) << "Window values should not exceed 1.0 for " << window_type;
                EXPECT_GT(max_val, min_val) << "Window should have variation for " << window_type;
            }
            
            // Check symmetry for symmetric windows
            if (window_type != "rectangular" && window_type != "boxcar" && window_size > 1) {
                auto first_half = window.slice(0, 0, window_size / 2);
                auto second_half = window.slice(0, window_size - window_size / 2, window_size).flip(0);
                int min_size = std::min(first_half.size(0), second_half.size(0));
                auto diff = torch::abs(first_half.slice(0, 0, min_size) - second_half.slice(0, 0, min_size));
                auto max_diff = torch::max(diff).item<float>();
                EXPECT_LT(max_diff, 1e-6f) << "Window should be symmetric for " << window_type;
                std::cout << "Symmetry check: max difference = " << std::scientific << max_diff << std::endl;
            }
            
            // Print first and last few values
            std::cout << "First 5 values: [";
            for (int i = 0; i < std::min(5, static_cast<int>(window.size(0))); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(6) << window[i].item<float>();
            }
            std::cout << "]" << std::endl;
            
            if (window.size(0) > 10) {
                std::cout << "Last 5 values: [";
                for (int i = window.size(0) - 5; i < window.size(0); ++i) {
                    if (i > window.size(0) - 5) std::cout << ", ";
                    std::cout << std::fixed << std::setprecision(6) << window[i].item<float>();
                }
                std::cout << "]" << std::endl;
            }
            
            std::cout << "✓ " << window_type << " window test passed" << std::endl << std::endl;
        }
    }
}

TEST_F(WindowCreationTest, WindowTypeCaseInsensitive) {
    for (const auto& device : test_devices) {
        // Test case insensitive window type names
        std::vector<std::pair<std::string, std::string>> case_variants = {
            {"hann", "HANN"},
            {"hamming", "Hamming"},
            {"blackman", "BLACKMAN"},
            {"rectangular", "Rectangular"}
        };
        
        for (const auto& variant_pair : case_variants) {
            auto window1 = create_window(variant_pair.first, window_size, device, false);
            auto window2 = create_window(variant_pair.second, window_size, device, false);
            
            ASSERT_TRUE(window1.defined() && window2.defined()) 
                << "Both case variants should be valid for " << variant_pair.first;
            
            auto diff = torch::abs(window1 - window2);
            auto max_diff = torch::max(diff).item<float>();
            EXPECT_LT(max_diff, 1e-7f) 
                << "Case variants should produce identical windows for " << variant_pair.first;
        }
    }
}

TEST_F(WindowCreationTest, InvalidWindowType) {
    for (const auto& device : test_devices) {
        // Test invalid window type
        EXPECT_THROW({
            auto window = create_window("invalid_window", window_size, device, false);
        }, std::invalid_argument) << "Invalid window type should throw exception on " << device;
    }
}

TEST_F(WindowCreationTest, ZeroSizeWindow) {
    for (const auto& device : test_devices) {
        // Test zero size window - should return empty tensor, not throw
        auto window = create_window("hann", 0, device, false);
        EXPECT_EQ(window.size(0), 0) << "Zero size window should return empty tensor on " << device;
    }
}

TEST_F(WindowCreationTest, SmallWindowSizes) {
    for (const auto& device : test_devices) {
        for (const auto& window_type : window_types) {
            // Test very small window sizes
            for (int size = 1; size <= 4; ++size) {
                auto window = create_window(window_type, size, device, false);
                
                ASSERT_TRUE(window.defined()) 
                    << "Small window should be defined for " << window_type 
                    << " size " << size << " on " << device;
                EXPECT_EQ(window.size(0), size) 
                    << "Window size should match for " << window_type 
                    << " size " << size << " on " << device;
                
                // For size 1, all windows should be [1.0]
                if (size == 1) {
                    EXPECT_FLOAT_EQ(window.item<float>(), 1.0f) 
                        << "Size-1 window should be [1.0] for " << window_type;
                }
            }
        }
    }
}

TEST_F(WindowCreationTest, LargeWindowSizes) {
    for (const auto& device : test_devices) {
        // Test larger window sizes
        std::vector<int> large_sizes = {2048, 4096, 8192};
        
        for (int size : large_sizes) {
            for (const auto& window_type : {"hann", "hamming", "rectangular"}) {
                auto window = create_window(window_type, size, device, false);
                
                ASSERT_TRUE(window.defined()) 
                    << "Large window should be defined for " << window_type 
                    << " size " << size << " on " << device;
                EXPECT_EQ(window.size(0), size) 
                    << "Window size should match for " << window_type 
                    << " size " << size << " on " << device;
            }
        }
    }
}

TEST_F(WindowCreationTest, WindowTypeComparison) {
    // Test that different window types produce different results
    for (const auto& device : test_devices) {
        auto hann_window = create_window("hann", window_size, device, false);
        auto hamming_window = create_window("hamming", window_size, device, false);
        auto blackman_window = create_window("blackman", window_size, device, false);
        auto rect_window = create_window("rectangular", window_size, device, false);
        
        // Windows should be different from each other
        auto hann_hamming_diff = torch::abs(hann_window - hamming_window);
        auto hann_blackman_diff = torch::abs(hann_window - blackman_window);
        auto hann_rect_diff = torch::abs(hann_window - rect_window);
        
        EXPECT_GT(torch::max(hann_hamming_diff).item<float>(), 0.01f) 
            << "Hann and Hamming windows should be noticeably different";
        EXPECT_GT(torch::max(hann_blackman_diff).item<float>(), 0.01f) 
            << "Hann and Blackman windows should be noticeably different";
        EXPECT_GT(torch::max(hann_rect_diff).item<float>(), 0.1f) 
            << "Hann and Rectangular windows should be very different";
    }
}

TEST_F(WindowCreationTest, CrossDeviceConsistency) {
    if (test_devices.size() < 2) {
        GTEST_SKIP() << "Need at least 2 devices for cross-device testing (found " << test_devices.size() << ")";
    }
    
    std::cout << "\n=== Testing Cross-Device Consistency ===" << std::endl;
    std::cout << "Comparing window generation across " << test_devices.size() << " devices" << std::endl;
    
    // Use CPU as reference device
    const auto& reference_device = test_devices[0]; // Always CPU
    
    for (const auto& window_type : window_types) {
        std::cout << "\n--- Testing " << window_type << " window consistency ---" << std::endl;
        
        // Generate reference windows on CPU
        auto ref_window_sym = create_window(window_type, window_size, reference_device, false);
        auto ref_window_per = create_window(window_type, window_size, reference_device, true);
        
        // Test on all other devices
        for (size_t i = 1; i < test_devices.size(); ++i) {
            const auto& test_device = test_devices[i];
            std::cout << "Comparing " << device_name(reference_device) << " vs " << device_name(test_device) << std::endl;
            
            // Generate windows on test device
            auto test_window_sym = create_window(window_type, window_size, test_device, false);
            auto test_window_per = create_window(window_type, window_size, test_device, true);
            
            // Move test device tensors back to CPU for comparison
            auto test_window_sym_cpu = test_window_sym.to(torch::kCPU);
            auto test_window_per_cpu = test_window_per.to(torch::kCPU);
            
            // Compare symmetric windows
            auto sym_diff = torch::abs(ref_window_sym - test_window_sym_cpu);
            auto max_sym_diff = torch::max(sym_diff).item<float>();
            EXPECT_LT(max_sym_diff, 1e-6f) 
                << "Symmetric " << window_type << " windows should be identical across devices. "
                << "Max difference: " << max_sym_diff;
            
            // Compare periodic windows
            auto per_diff = torch::abs(ref_window_per - test_window_per_cpu);
            auto max_per_diff = torch::max(per_diff).item<float>();
            EXPECT_LT(max_per_diff, 1e-6f) 
                << "Periodic " << window_type << " windows should be identical across devices. "
                << "Max difference: " << max_per_diff;
            
            // Compare sums
            auto ref_sym_sum = ref_window_sym.sum().item<float>();
            auto test_sym_sum = test_window_sym_cpu.sum().item<float>();
            auto ref_per_sum = ref_window_per.sum().item<float>();
            auto test_per_sum = test_window_per_cpu.sum().item<float>();
            
            EXPECT_NEAR(ref_sym_sum, test_sym_sum, 1e-4f) 
                << "Symmetric window sums should match across devices";
            EXPECT_NEAR(ref_per_sum, test_per_sum, 1e-4f) 
                << "Periodic window sums should match across devices";
            
            std::cout << "  ✓ " << window_type << " consistency verified between devices" << std::endl;
        }
    }
    
    std::cout << "\nCross-device consistency tests completed!" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#include "core_util_devices.h"
#include <vector> // Required for torch::cuda::device_count()


namespace contorchionist {
    namespace core {
        namespace util_devices {

std::pair<torch::Device, std::string> parse_torch_device(const std::string& device_str_requested) {
    torch::Device new_device(torch::kCPU); // Default to CPU
    std::string error_message;

    try {
        // First, try to create the device directly to let LibTorch handle parsing
        torch::Device parsed_device_candidate(device_str_requested);
        
        // Now validate the device type and availability
        if (parsed_device_candidate.type() == torch::kCUDA) {
            if (!torch::cuda::is_available()) {
                error_message = "CUDA specified ('" + device_str_requested + "') but not available.";
                new_device = torch::Device(torch::kCPU); // Fallback
            } else if (parsed_device_candidate.has_index()) {
                if (parsed_device_candidate.index() >= static_cast<int16_t>(torch::cuda::device_count())) {
                    error_message = "CUDA device index " + std::to_string(parsed_device_candidate.index()) +
                                    " out of range. Max index: " + std::to_string(torch::cuda::device_count() - 1) + ".";
                    new_device = torch::Device(torch::kCPU); // Fallback
                } else {
                    new_device = parsed_device_candidate; // Valid CUDA device with index
                }
            } else {
                new_device = parsed_device_candidate; // Valid CUDA device (e.g. "cuda" when available)
            }
        } else if (parsed_device_candidate.type() == torch::kCPU) {
            new_device = parsed_device_candidate; // CPU is always valid
        } else {
            // For any other device types (MPS, Metal, Vulkan, XLA, etc.)
            // We attempt to assign it and let LibTorch validate it during actual tensor operations
            // This provides forward compatibility with future device types
            try {
                // For MPS devices, check availability if possible
                if (parsed_device_candidate.type() == torch::kMPS) {
                    // Try to check MPS availability - this will work if torch::mps exists
                    try {
                        if (torch::mps::is_available()) {
                            new_device = parsed_device_candidate; // Valid MPS device
                        } else {
                            error_message = "MPS specified ('" + device_str_requested + "') but not available on this system.";
                            new_device = torch::Device(torch::kCPU); // Fallback
                        }
                    } catch (...) {
                        // torch::mps might not be available in this version, fall back to tensor test
                        torch::tensor({1.0f}, torch::TensorOptions().device(parsed_device_candidate));
                        new_device = parsed_device_candidate; // Device test passed
                    }
                } else {
                    // For other device types, test if we can create a simple tensor
                    torch::tensor({1.0f}, torch::TensorOptions().device(parsed_device_candidate));
                    new_device = parsed_device_candidate; // Device appears to work
                }
            } catch (const c10::Error& device_test_error) {
                error_message = "Device '" + device_str_requested + "' is not available or supported: " + device_test_error.what();
                new_device = torch::Device(torch::kCPU); // Fallback
            }
        }
    } catch (const c10::Error& e) {
        error_message = "LibTorch error parsing device string '" + device_str_requested + "': " + e.what();
        new_device = torch::Device(torch::kCPU); // Fallback to CPU
    } catch (const std::exception& e) {
        error_message = "Standard error while parsing device string '" + device_str_requested + "': " + e.what();
        new_device = torch::Device(torch::kCPU); // Fallback to CPU
    }

    return {new_device, error_message};
}

        } // namespace util_device
    } // namespace core
} // namespace contorchionist

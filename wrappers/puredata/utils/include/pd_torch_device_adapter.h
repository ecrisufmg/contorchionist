#ifndef PD_TORCH_DEVICE_ADAPTER_H
#define PD_TORCH_DEVICE_ADAPTER_H

#include "m_pd.h"
#include <torch/torch.h>
#include <string>
#include <algorithm> // For std::transform and std::toupper
#include "../../../core/include/core_util_devices.h" // Updated include for core_util_devices.h 

// Wrapper function for Pure Data objects to use the core device parsing utility
// and handle Pd-specific logging (post, pd_error).
static inline void pd_parse_and_set_torch_device(
    void* pd_obj, // t_object* or specific type like t_torch_rfft_tilde*
    torch::Device& target_device, // Reference to the device member in the Pd object
    const std::string& device_str_requested,
    bool verbose,
    const char* object_name, // For logging, e.g., "torch.rfft~"
    bool device_flag_was_present // Indicates if a device flag like -device, -d, -cuda was explicitly used
) {
    if (verbose) {
        post("%s: Attempting to set device. Requested string: \"%s\", Explicit flag present: %s", 
             object_name, device_str_requested.c_str(), device_flag_was_present ? "yes" : "no");
    }

    if (!device_flag_was_present && device_str_requested == "cpu") {
        // No explicit device flag was used, and the argument (or default) is "cpu".
        // We can assume the initial default of torch::kCPU for target_device is fine.
        if (verbose) {
            post("%s: No explicit device flag and default 'cpu' requested. Using pre-set CPU.", object_name);
        }
        // Ensure target_device is indeed CPU if it wasn't already.
        if (target_device.type() != torch::kCPU) {
             target_device = torch::Device(torch::kCPU);
        }
        return; // No need to call core parser
    }

    // If an explicit flag was present, or if the device_str is not "cpu" (even if no explicit flag, e.g. from a symbol arg)
    // then proceed to parse.
    auto result = contorchionist::core::util_devices::parse_torch_device(device_str_requested);
    target_device = result.first;
    std::string error_message = result.second;

    if (!error_message.empty()) {
        if (target_device.type() != torch::kCPU) { // If fallback didn't already set it to CPU
            pd_error(pd_obj, "%s: %s Falling back to CPU.", object_name, error_message.c_str());
            target_device = torch::Device(torch::kCPU); // Ensure fallback
        } else {
            post("%s: Notice - %s Using CPU.", object_name, error_message.c_str());
        }
    }

    if (verbose) {
        post("%s: Device successfully set to: %s (Requested: \"%s\")",
             object_name, target_device.str().c_str(), device_str_requested.c_str());
    }
}

// Simple function to convert a device string to torch::Device
// Returns the device and a boolean indicating success
static inline std::pair<torch::Device, bool> get_device_from_string(const std::string& device_str) {
    auto result = contorchionist::core::util_devices::parse_torch_device(device_str);
    torch::Device device = result.first;
    std::string error_message = result.second;
    
    if (!error_message.empty()) {
        // If there was an error, return CPU as fallback and false for success
        return std::make_pair(torch::Device(torch::kCPU), false);
    }
    
    return std::make_pair(device, true);
}

// Helper function to get a user-friendly device name string for any torch::Device
// This provides consistent device naming across all PureData externals
static inline std::string pd_torch_device_friendly_name(const torch::Device& device) {
    switch (device.type()) {
        case torch::kCPU:
            return "CPU";
        case torch::kCUDA:
            if (device.has_index()) {
                return "CUDA:" + std::to_string(device.index());
            } else {
                return "CUDA";
            }
        case torch::kMPS:
            return "MPS";
        case torch::kXLA:
            return "XLA";
        case torch::kMetal:
            return "Metal";
        case torch::kVulkan:
            return "Vulkan";
        default:
            // For future device types or unknown devices, fall back to LibTorch's string representation
            std::string device_str = device.str();
            // Convert to uppercase for consistency with other device names
            std::transform(device_str.begin(), device_str.end(), device_str.begin(), 
                           [](unsigned char c){ return std::toupper(c); });
            return device_str;
    }
}

#endif // PD_TORCH_DEVICE_ADAPTER_H

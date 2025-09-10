#ifndef CORE_UTIL_DEVICES_H
#define CORE_UTIL_DEVICES_H

#include <torch/torch.h>
#include <string>
#include <utility> // For std::pair
#include "contorchionist_core/contorchionist_core_export.h"

namespace contorchionist {
    namespace core {
        namespace util_devices {


/**
 * @brief Parses a device string and determines the torch::Device.
 *
 * @param device_str_requested The requested device string (e.g., "cpu", "cuda", "cuda:0").
 * @return A pair containing the determined torch::Device and an error string.
 *         The error string is empty if no error occurred.
 *         If an error occurs, it defaults to CPU and provides an error message.
 */
CONTORCHIONIST_CORE_EXPORT std::pair<torch::Device, std::string> parse_torch_device(
    const std::string& device_str_requested
);

        } // namespace util_devices
    } // namespace core
} // namespace contorchionist
#endif // CORE_UTIL_DEVICES_H

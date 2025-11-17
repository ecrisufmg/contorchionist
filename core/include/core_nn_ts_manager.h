#ifndef CORE_NN_TS_MANAGER_H
#define CORE_NN_TS_MANAGER_H

#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <string>
#include "contorchionist_core/contorchionist_core_export.h" // Include the generated export header

namespace contorchionist {
    namespace core {
        namespace nn_ts_manager {


/**
 * Load a TorchScript model from file
 * @param path Model file path
 * @param model Output model unique_ptr
 * @param device Target device for model
 * @param error_message Optional error message output
 * @return true if successful, false otherwise
 */
CONTORCHIONIST_CORE_EXPORT
bool load_torchscript_model(
    const char* path,
    std::unique_ptr<torch::jit::script::Module>& model,
    const torch::Device& device,
    std::string* error_message = nullptr);

/**
 * Get list of strings from model method (e.g., get_methods, get_attributes)
 * @param model Model pointer
 * @param method_name Method name to call
 * @param logmess Optional log message output
 * @return Vector of strings returned by method
 */
CONTORCHIONIST_CORE_EXPORT
std::vector<std::string> get_string_list_from_method(
    torch::jit::script::Module* model,
    const std::string& method_name,
    std::string* logmess = nullptr);


/**
 * @brief Extract input/output channels and model buffer size from a method in a TorchScript model
 * @param model Pointer to the TorchScript model
 * @param method_name Name of the method to extract information from
 * @param in_ch Output parameter for input channels
 * @param out_ch Output parameter for output channels
 * @param model_buffer_size Output parameter for model buffer size
 * @param error_message Optional error message output
 * @param log_messages Optional log messages output
 * @return true if successful, false otherwise
 */
CONTORCHIONIST_CORE_EXPORT
bool extract_channels_method(
    torch::jit::script::Module* model,
    const std::string& method_name,
    int& in_ch,
    int& out_ch,
    int& model_buffer_size,
    int& max_buffer_size,                
    std::string* error_message = nullptr,
    std::vector<std::string>* log_messages = nullptr);

/**
 * @brief Extract input/output shapes from a method in a TorchScript model
 * @param model Pointer to the TorchScript model
 * @param selected_method Name of the method to extract shapes from
 * @param input_shape Output vector for input shape
 * @param output_shape Output vector for output shape
 * @param input_tensor Pointer to the input tensor (optional)
 * @param error_message Optional error message output
 * @param log_messages Optional log messages output
 * @return true if successful, false otherwise
 */
CONTORCHIONIST_CORE_EXPORT
bool extract_shape_method(
    torch::jit::script::Module* model,      // model pointer
    const std::string& selected_method,     // selected method name
    std::vector<int64_t>* input_shape,      // input shape vector
    std::vector<int64_t>* output_shape,     // output shape vector
    at::Tensor* input_tensor,               // input tensor pointer (opcional)
    std::string* error_message = nullptr,   // optional error message
    std::vector<std::string>* log_messages = nullptr);  // optional log messages




/**
 * @brief Get a string attribute from a TorchScript model
 * @param model Pointer to the TorchScript model
 * @param attr_name Name of the attribute to get
 * @param out_value Output string value
 * @param error_message Optional error message output
 * @return true if successful, false otherwise
 */
CONTORCHIONIST_CORE_EXPORT
bool has_attribute(
    torch::jit::script::Module* model,
    const std::string& attr_name,
    std::string& out_value,
    std::string* error_message = nullptr);


    


        } // namespace nn_ts_manager
    } // namespace core
} // namespace contorchionist

#endif // CORE_NN_TS_MANAGER_H

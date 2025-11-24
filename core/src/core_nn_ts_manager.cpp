#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <sstream>
#include "core_nn_ts_manager.h"

 
namespace contorchionist {
    namespace core {
        namespace nn_ts_manager {

// ----------------- load a torchscript model  ----------------- //
bool load_torchscript_model(
    const char* path,
    std::unique_ptr<torch::jit::script::Module>& model,
    const torch::Device& device, 
    std::string* error_message){

    try {
        auto torch_model = std::make_unique<torch::jit::script::Module>();
        *torch_model = torch::jit::load(path);
        torch_model->eval();
        model = std::move(torch_model);
        model->to(device);
        return true;
    } catch (const std::exception& e) {
        if (error_message){
            *error_message = std::string("Failed loading the model: ") + e.what();
        }
        return false;
    }
}

//------------------------ retrieve methods and atributes string lists (ex: get_methods, get_attributes) ------------------------ //
std::vector<std::string> get_string_list_from_method(torch::jit::script::Module* model, const std::string& method_name, std::string* logmess) {
    std::vector<std::string> result_list; // string vector to store the methods, attributes or parameters names

    if (!model) { 
        if (logmess) *logmess = "No model loaded.";
        return result_list;
    }

    try {
        if (model->find_method(method_name)) { // check if the method exists
            auto result = model->get_method(method_name)({}); //get the result of the method 
            if (result.isList()) { // check if the result is a list
                auto list = result.toList(); // convert the result to a list
                for (const auto& iv : list) { // iterate over the list elements
                    const c10::IValue& val = static_cast<const c10::IValue&>(iv); // get the value
                    if (val.isString()) { // check if the value is a string
                        result_list.push_back(val.toStringRef()); // add the string to the result list
                    }
                }
                // Log the result list if logmess is provided
                if (logmess) {
                    std::ostringstream oss;
                    oss << "[";
                    for (size_t i = 0; i < result_list.size(); ++i) {
                        oss << result_list[i];
                        if (i < result_list.size() - 1) oss << ", ";
                    }
                    oss << "]";
                    *logmess = oss.str();
                }
            } else if (logmess) {
                *logmess = "Method '" + method_name + "' did not return a list";
            }
        } else if (logmess) {
            *logmess = "Method '" + method_name + "' not found in model";
        }
    } catch (const c10::Error& e) {
        if (logmess) {
            *logmess = "PyTorch error: " + std::string(e.what());
        }
    } catch (const std::exception& e) {
        if (logmess) {
            *logmess = "Error: " + std::string(e.what());
        }
    } catch (...) {
        if (logmess) {
            *logmess = "Unknown error occurred";
        }
    }
    
    return result_list;
}


// ----------------- get numbers of in_ch and out_ch from a torchscript model ----------------- //
bool extract_channels_method(
    torch::jit::script::Module* model, // model pointer
    const std::string& method_name, // method name (ex: "forward")
    int& in_ch, // input channels
    int& out_ch, // output channels
    int& model_buffer_size, // block size (model block size)
    int& max_buffer_size, // max buffer size that the model can handle
    std::string* error_message,
    std::vector<std::string>* log_messages) {

    // check if the model is valid
    if (!model) {
        if (error_message) {
            *error_message = "No model loaded.";
        }
        return false;
    }
    
    // get the numbers of input and output channels from a method model
    try {
        // search for method name + "_in_ch" attribute (ex: "forward_in_ch" for "forward" method)
        std::string in_ch_attr = method_name + "_in_ch";
        if (model->hasattr(in_ch_attr)) {
            in_ch = model->attr(in_ch_attr.c_str()).toInt();
            if (log_messages) {
                log_messages->push_back(in_ch_attr + ": input channels = " + std::to_string(in_ch));
            }
        } else {
            if (error_message) {
                *error_message = "Model has no '" + in_ch_attr + "' attribute.";
            }
            in_ch = -1;
        }
        
        // search for method name + "_out_ch" attribute (ex: "forward_out_ch" for "forward" method)
        std::string out_ch_attr = method_name + "_out_ch";
        if (model->hasattr(out_ch_attr)) {
            out_ch = model->attr(out_ch_attr.c_str()).toInt();
            if (log_messages) {
                log_messages->push_back(out_ch_attr + ": output channels = " + std::to_string(out_ch));
            }
        } else {
            if (error_message) {
                *error_message = "Model has no '" + out_ch_attr + "' attribute.";
            }
            out_ch = -1;
        }
        
        // search for model internal buffer attribute
        std::string m_buffer_size_attr = "m_buffer_size";
        if (model->hasattr(m_buffer_size_attr)) {
            model_buffer_size = model->attr(m_buffer_size_attr.c_str()).toInt();
            if (log_messages) {
                log_messages->push_back(m_buffer_size_attr + ": Model buffer size = " + std::to_string(model_buffer_size));
            }
        } else {
            if (error_message) {
                *error_message = "Model has no '" + m_buffer_size_attr + "' attribute.";
            }
            model_buffer_size = -1;
        }

        // search for model max buffer attribute
        std::string max_buffer_size_attr = "max_buffer_size";
        if (model->hasattr(max_buffer_size_attr)) {
            int max_buffer_size = model->attr(max_buffer_size_attr.c_str()).toInt();
            if (log_messages) {
                log_messages->push_back(max_buffer_size_attr + ": Model max buffer size = " + std::to_string(max_buffer_size));
            }
        } else {
            if (error_message) {
                *error_message = "Model has no '" + max_buffer_size_attr + "' attribute.";
            }
        }
        return true;
    } catch (const std::exception& e) {
        if (error_message) {
            *error_message = "Error extracting channels for method '" + method_name + "': " + e.what();
        }
        return false;
    }
}

// ----------------- retrieve model methods in/out shape  ----------------- //
bool extract_shape_method(
    torch::jit::script::Module* model, // model pointer
    const std::string& selected_method, // selected method name
    std::vector<int64_t>* input_shape, // input shape vector
    std::vector<int64_t>* output_shape, // output shape vector
    at::Tensor* input_tensor, // input tensor pointer (declared at tensors struct)
    std::string* error_message,
    std::vector<std::string>* log_messages) {

    if (!model) {
        if (error_message) *error_message = "Model pointer is null.";
        return false;
    }

    try {
        // retrieve input shape from a selected method
        if (input_shape) {
            std::string input_shape_attr = selected_method + "_input_shape";
            if (model->hasattr(input_shape_attr)) {
                auto input_attr = model->attr(input_shape_attr.c_str()).toIntList();
                input_shape->assign(input_attr.begin(), input_attr.end());
                
                // Log input shape
                if (log_messages) {
                    std::ostringstream oss;
                    oss << "Input shape for method '" << selected_method << "': [";
                    for (size_t i = 0; i < input_shape->size(); ++i) {
                        oss << (*input_shape)[i];
                        if (i < input_shape->size() - 1) oss << ", ";
                    }
                    oss << "]";
                    log_messages->push_back(oss.str());
                }
            } else {
                if (error_message) {
                    *error_message = "WARNING: model has no '" + input_shape_attr + "' attribute.";
                }
                input_shape->clear();
            }
        }

        // output_shape
        if (output_shape) {
            if (model->hasattr("output_shape")) {
                auto output_attr = model->attr("output_shape").toIntList();
                output_shape->assign(output_attr.begin(), output_attr.end());
                
                // Log output shape
                if (log_messages) {
                    std::ostringstream oss;
                    oss << "Output shape: [";
                    for (size_t i = 0; i < output_shape->size(); ++i) {
                        oss << (*output_shape)[i];
                        if (i < output_shape->size() - 1) oss << ", ";
                    }
                    oss << "]";
                    log_messages->push_back(oss.str());
                }
            } else {
                if (error_message) {
                    *error_message = "Warning: model has no 'output_shape' attribute.";
                }
                output_shape->clear();
            }
        }

        // create the input tensor based on the selected method input shape 
        if (input_tensor && input_shape && !input_shape->empty()) {
            *input_tensor = torch::randn(*input_shape, torch::TensorOptions().dtype(torch::kFloat));
            
            // Log tensor creation
            if (log_messages) {
                std::ostringstream oss;
                oss << "Input tensor created with shape " << input_tensor->sizes();
                log_messages->push_back(oss.str());
            }
        } else if (input_tensor) {
            *input_tensor = at::Tensor();
            if (error_message) {
                *error_message = "WARNING: Input tensor not created. Check input shape.";
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        if (error_message) {
            *error_message = "Error extracting shapes or creating input tensor: " + std::string(e.what());
        }
        return false;
    }
}


// ------------ get a string attribute from a torchscript model ---------- //
bool has_attribute(
    torch::jit::script::Module* model,
    const std::string& attr_name,
    std::string& out_value,
    std::string* error_message) {
        
    if (!model) {
        if (error_message) *error_message = "Model pointer is null.";
        return false;
    }
    try {
        if (model->hasattr(attr_name)) {
            out_value = model->attr(attr_name.c_str()).toStringRef();
            return true;
        } else {
            if (error_message) *error_message = "Model has no '" + attr_name + "' attribute.";
            return false;
        }
    } catch (const c10::Error& e) {
        if (error_message) *error_message = "Error getting attribute '" + attr_name + "': " + e.what();
        return false;
    }
}

        } // namespace nn_ts_manager
    } // namespace core
} // namespace contorchionist

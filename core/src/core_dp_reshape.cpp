#include <torch/torch.h>
#include <string>
#include <vector>
#include <functional>
#include <map>    
#include <stdexcept>
#include <algorithm>
#include "core_dp_reshape.h"


namespace contorchionist {
    namespace core {
        namespace dp_reshape {



// reshape method map
using ReshapeFunction = std::function<at::Tensor(const at::Tensor&, const ReshapeParams&)>;

static const std::map<std::string, ReshapeFunction>& get_reshape_method_map() {
    static const std::map<std::string, ReshapeFunction> reshape_map = {
        {"view", [](const at::Tensor& tensor, const ReshapeParams& params) {
            return tensor.view(params.shape);
        }},
        {"reshape", [](const at::Tensor& tensor, const ReshapeParams& params) {
            return tensor.reshape(params.shape);
        }},
        {"flatten", [](const at::Tensor& tensor, const ReshapeParams& params) {
            return torch::flatten(tensor, params.dim1, params.dim2);
        }},
        {"squeeze", [](const at::Tensor& tensor, const ReshapeParams& params) {
            return tensor.squeeze(params.dim1);
        }},
        {"unsqueeze", [](const at::Tensor& tensor, const ReshapeParams& params) {
            return tensor.unsqueeze(params.dim1);
        }},
        {"permute", [](const at::Tensor& tensor, const ReshapeParams& params) {
            return tensor.permute(params.dims);
        }},
        {"transpose", [](const at::Tensor& tensor, const ReshapeParams& params) {
            return tensor.transpose(params.dim1, params.dim2);
        }}
    };
    return reshape_map;
}

// list all 
std::vector<std::string> list_available_methods() {
    std::vector<std::string> methods;
    const auto& reshape_map = get_reshape_method_map();
    methods.reserve(reshape_map.size());
    for (const auto& kv : reshape_map) {
        methods.push_back(kv.first);
    }
    return methods;
}

bool is_valid_method(const std::string& method) {
    const auto& reshape_map = get_reshape_method_map();
    return reshape_map.find(method) != reshape_map.end();
}

static std::string dtype_to_string(c10::ScalarType dt) {
    return c10::toString(dt);
}

ReshapeResult ReshapeProcessor(
    const at::Tensor& input_tensor,
    const std::string& method,
    const ReshapeParams& params,
    const torch::Device& device) {

    ReshapeResult result;
    result.success = false;

    if (!input_tensor.defined()) {
        result.error_message = "Input tensor is not defined";
        return result;
    }

    at::Tensor tensor_on_device = input_tensor.to(device);

    const auto& reshape_map = get_reshape_method_map();
    auto it = reshape_map.find(method);
    if (it == reshape_map.end()) {
        result.error_message = "Unknown reshape method: " + method;
        return result;
    }

    try {
        // Validações dependentes do tensor por método
        if (method == "view" || method == "reshape") {
            if (params.shape.empty()) {
                result.error_message = method + " requires non-empty shape";
                return result;
            }
            int64_t numel = tensor_on_device.numel();
            int64_t shape_prod = 1;
            int wildcard_count = 0;
            for (auto v : params.shape) {
                if (v == -1) {
                    ++wildcard_count;
                } else {
                    if (v <= 0) {
                        result.error_message = method + " shape dimensions must be positive or -1";
                        return result;
                    }
                    shape_prod *= v;
                }
            }
            if (wildcard_count > 1) {
                result.error_message = "Only one '-1' is allowed in shape";
                return result;
            }
            if (wildcard_count == 0 && shape_prod != numel) {
                result.error_message = "Number of elements (" + std::to_string(numel) +
                    ") does not match requested shape (product=" + std::to_string(shape_prod) + ")";
                return result;
            }
            if (wildcard_count == 1 && (shape_prod == 0 || (numel % shape_prod) != 0)) {
                result.error_message = "Cannot infer dimension '-1' for shape";
                return result;
            }

        } else if (method == "flatten") {
            int64_t ndims = tensor_on_device.dim();
            if (params.dim1 < 0 || params.dim1 >= ndims) {
                result.error_message = "flatten: start_dim " + std::to_string(params.dim1) +
                    " out of bounds for tensor with " + std::to_string(ndims) + " dims";
                return result;
            }
            if (params.dim2 != -1 && (params.dim2 < params.dim1 || params.dim2 >= ndims)) {
                result.error_message = "flatten: end_dim " + std::to_string(params.dim2) +
                    " out of bounds for tensor with " + std::to_string(ndims) + " dims";
                return result;
            }

        } else if (method == "squeeze") {
            int64_t ndims = tensor_on_device.dim();
            if (params.dim1 < 0 || params.dim1 >= ndims) {
                result.error_message = "squeeze: dim " + std::to_string(params.dim1) +
                    " out of bounds for tensor with " + std::to_string(ndims) + " dims";
                return result;
            }
            if (tensor_on_device.sizes()[params.dim1] != 1) {
                result.error_message = "squeeze: dimension " + std::to_string(params.dim1) +
                    " is not of size 1 (size=" + std::to_string(tensor_on_device.sizes()[params.dim1]) + ")";
                return result;
            }

        } else if (method == "unsqueeze") {
            int64_t ndims = tensor_on_device.dim();
            if (params.dim1 < 0 || params.dim1 > ndims) {
                result.error_message = "unsqueeze: dim " + std::to_string(params.dim1) +
                    " out of bounds for tensor with " + std::to_string(ndims) + " dims";
                return result;
            }

        } else if (method == "permute") {
            int64_t ndims = tensor_on_device.dim();
            if ((int64_t)params.dims.size() != ndims) {
                result.error_message = "permute: number of dims (" + std::to_string(params.dims.size()) +
                    ") does not match tensor dims (" + std::to_string(ndims) + ")";
                return result;
            }
            std::vector<bool> seen(ndims, false);
            for (auto d : params.dims) {
                if (d < 0 || d >= ndims || seen[d]) {
                    result.error_message = "permute: invalid or repeated dim " + std::to_string(d) +
                        " for tensor with " + std::to_string(ndims) + " dims";
                    return result;
                }
                seen[d] = true;
            }

        } else if (method == "transpose") {
            int64_t ndims = tensor_on_device.dim();
            if (params.dim1 < 0 || params.dim1 >= ndims || params.dim2 < 0 || params.dim2 >= ndims) {
                result.error_message = "transpose: dims (" + std::to_string(params.dim1) + ", " +
                    std::to_string(params.dim2) + ") out of bounds for tensor with " +
                    std::to_string(ndims) + " dims";
                return result;
            }
            if (params.dim1 == params.dim2) {
                result.error_message = "transpose: dims must be different";
                return result;
            }
        }

        // Aplica a operação
        result.output_tensor = it->second(tensor_on_device, params);
        result.success = true;

    } catch (const c10::Error& e) {
        result.error_message = "PyTorch error in " + method + ": " + std::string(e.what());
    } catch (const std::exception& e) {
        result.error_message = "Error in " + method + ": " + std::string(e.what());
    }

    return result;
}

std::vector<int64_t> calculate_output_shape(
    const std::vector<int64_t>& input_shape,
    const std::string& method,
    const ReshapeParams& params) {

    if (input_shape.empty()) {
        return {};
    }

    try {
        if (method == "view" || method == "reshape") {
            return params.shape;

        } else if (method == "flatten") {
            std::vector<int64_t> output_shape;
            int64_t start_dim = params.dim1;
            int64_t end_dim = (params.dim2 == -1) ? (int64_t)input_shape.size() - 1 : params.dim2;
            // antes
            for (int64_t i = 0; i < start_dim; ++i) output_shape.push_back(input_shape[i]);
            // achatado
            int64_t flat = 1;
            for (int64_t i = start_dim; i <= end_dim; ++i) flat *= input_shape[i];
            output_shape.push_back(flat);
            // depois
            for (int64_t i = end_dim + 1; i < (int64_t)input_shape.size(); ++i) output_shape.push_back(input_shape[i]);
            return output_shape;

        } else if (method == "squeeze") {
            std::vector<int64_t> output_shape;
            for (int64_t i = 0; i < (int64_t)input_shape.size(); ++i) {
                if (i != params.dim1) output_shape.push_back(input_shape[i]);
            }
            return output_shape;

        } else if (method == "unsqueeze") {
            std::vector<int64_t> output_shape = input_shape;
            output_shape.insert(output_shape.begin() + params.dim1, 1);
            return output_shape;

        } else if (method == "permute") {
            std::vector<int64_t> output_shape;
            output_shape.reserve(params.dims.size());
            for (auto d : params.dims) output_shape.push_back(input_shape[d]);
            return output_shape;

        } else if (method == "transpose") {
            std::vector<int64_t> output_shape = input_shape;
            std::swap(output_shape[params.dim1], output_shape[params.dim2]);
            return output_shape;
        }
    } catch (...) {
        return {};
    }

    return input_shape;
}

ArgumentValidationResult parse_list_arguments(
    const std::string& method,
    const std::vector<float>& args) {

    ArgumentValidationResult result;
    result.valid = false;
    result.method_name = method;

    result.params.shape.clear();
    result.params.dims.clear();
    result.params.dim1 = 0;
    result.params.dim2 = -1;

    try {
        if (!is_valid_method(method)) {
            auto available = list_available_methods();
            std::string list;
            for (auto& m : available) { list += m + ", "; }
            if (!list.empty()) { list.pop_back(); list.pop_back(); }
            result.error_message = "Invalid reshape method: " + method + ". Available: " + list;
            return result;
        }

        if (method == "view" || method == "reshape") {
            if (args.empty()) {
                result.error_message = method + " requires at least one shape dimension";
                return result;
            }
            result.params.shape.reserve(args.size());
            for (float a : args) {
                result.params.shape.push_back(static_cast<int64_t>(a));
            }
            int wildcard_count = 0;
            for (auto d : result.params.shape) {
                if (d == -1) ++wildcard_count;
                else if (d <= 0) {
                    result.error_message = method + " shape dimensions must be positive or -1";
                    return result;
                }
            }
            if (wildcard_count > 1) {
                result.error_message = method + " allows only one '-1' wildcard dimension";
                return result;
            }

        } else if (method == "flatten") {
            result.params.dim1 = args.size() > 0 ? static_cast<int64_t>(args[0]) : 1;
            result.params.dim2 = args.size() > 1 ? static_cast<int64_t>(args[1]) : -1;
            if (result.params.dim1 < 0) {
                result.error_message = "flatten start_dim must be non-negative";
                return result;
            }
            if (result.params.dim2 != -1 && result.params.dim2 < result.params.dim1) {
                result.error_message = "flatten end_dim must be >= start_dim or -1";
                return result;
            }

        } else if (method == "squeeze") {
            if (args.size() > 1) {
                result.error_message = "squeeze takes at most one dimension argument";
                return result;
            }
            result.params.dim1 = args.size() == 1 ? static_cast<int64_t>(args[0]) : 0;
            if (result.params.dim1 < 0) {
                result.error_message = "squeeze dimension must be non-negative";
                return result;
            }

        } else if (method == "unsqueeze") {
            if (args.size() != 1) {
                result.error_message = "unsqueeze requires exactly one dimension argument";
                return result;
            }
            result.params.dim1 = static_cast<int64_t>(args[0]);
            if (result.params.dim1 < 0) {
                result.error_message = "unsqueeze dimension must be non-negative";
                return result;
            }

        } else if (method == "permute") {
            if (args.empty()) {
                result.error_message = "permute requires at least one dimension argument";
                return result;
            }
            result.params.dims.reserve(args.size());
            for (float a : args) {
                int64_t d = static_cast<int64_t>(a);
                if (d < 0) {
                    result.error_message = "permute dimensions must be non-negative";
                    return result;
                }
                result.params.dims.push_back(d);
            }
            std::vector<int64_t> sorted = result.params.dims;
            std::sort(sorted.begin(), sorted.end());
            for (size_t i = 1; i < sorted.size(); ++i) {
                if (sorted[i] == sorted[i - 1]) {
                    result.error_message = "permute dimensions must be unique";
                    return result;
                }
            }

        } else if (method == "transpose") {
            if (args.size() != 2) {
                result.error_message = "transpose requires exactly two dimension arguments";
                return result;
            }
            result.params.dim1 = static_cast<int64_t>(args[0]);
            result.params.dim2 = static_cast<int64_t>(args[1]);
            if (result.params.dim1 < 0 || result.params.dim2 < 0) {
                result.error_message = "transpose dimensions must be non-negative";
                return result;
            }
            if (result.params.dim1 == result.params.dim2) {
                result.error_message = "transpose dimensions must be different";
                return result;
            }
        }

        result.valid = true;

    } catch (const std::exception& e) {
        result.error_message = "Error parsing arguments: " + std::string(e.what());
    }

    return result;
}

ArgumentValidationResult parse_individual_arguments(
    const std::string& method, // method name
    const std::vector<int64_t>& shape, // target shape for reshape
    int64_t dim1, // dim1 for flatten/squeeze/unsqueeze
    int64_t dim2, // dim2 for flatten
    const std::vector<int64_t>& dims) { // dims for permute

    ArgumentValidationResult result;
    result.valid = false;
    result.method_name = method;

    result.params.shape.clear();
    result.params.dims.clear();
    result.params.shape.clear();
    result.params.dims.clear();
    result.params.dim1 = 0;
    result.params.dim2 = -1;

    try {
        if (!is_valid_method(method)) {
            auto available = list_available_methods();
            std::string list;
            for (auto& m : available) { list += m + ", "; }
            if (!list.empty()) { list.pop_back(); list.pop_back(); }
            result.error_message = "Invalid reshape method: " + method + ". Available: " + list;
            return result;
        }
        // reshape/view
        if (method == "view" || method == "reshape") {
            if (shape.empty()) {
                result.error_message = method + " requires at least one shape dimension";
                return result;
            }
            result.params.shape = shape;
            int wildcard_count = 0;
            for (auto d : shape) {
                if (d == -1) ++wildcard_count;
                else if (d <= 0) {
                    result.error_message = method + " shape dimensions must be positive or -1";
                    return result;
                }
            }
            if (wildcard_count > 1) {
                result.error_message = method + " allows only one '-1' wildcard dimension";
                return result;
            }
        
        // flatten 
        } else if (method == "flatten") {
            result.params.dim1 = (dim1 >= 0) ? dim1: 0;
            result.params.dim2 = (dim2 >= 0) ? dim2: -1;
            if (result.params.dim1 < 0) {
                result.error_message = "flatten start_dim must be non-negative";
                return result;
            }
            if (result.params.dim2 != -1 && result.params.dim2 < result.params.dim1) {
                result.error_message = "flatten end_dim must be >= start_dim or -1";
                return result;
            }

        } else if (method == "squeeze") {
            if (dim1 < 0) {
                result.error_message = "squeeze takes at most one dimension argument";
                return result;
            }
            result.params.dim1 = dim1;

        // unsqueeze
        } else if (method == "unsqueeze") {
            if (dim1 < 0) {
                result.error_message = "unsqueeze requires exactly one dimension argument";
                return result;
            }
            result.params.dim1 = dim1;

        // permute
        } else if (method == "permute") {
            if (dims.empty()) {
                result.error_message = "permute requires at least one dimension argument";
                return result;
            }
            for (auto d : dims) {
                if (d < 0) {
                    result.error_message = "permute dimensions must be non-negative";
                    return result;
                }
            }
            std::vector<int64_t> sorted = dims;
            std::sort(sorted.begin(), sorted.end());
            for (size_t i = 1; i < sorted.size(); ++i) {
                if (sorted[i] == sorted[i - 1]) {
                    result.error_message = "permute dimensions must be unique";
                    return result;
                }
            }
            result.params.dims = dims;

        // transpose
        } else if (method == "transpose") {
            if (dim1 < 0 || dim2 < 0) {
                result.error_message = "transpose dimensions must be non-negative";
                return result;
            }
            if (dim1 == dim2) {
                result.error_message = "transpose dimensions must be different";
                return result;
            }
            result.params.dim1 = dim1;
            result.params.dim2 = dim2;
        }

        result.valid = true;

    } catch (const std::exception& e) {
        result.error_message = "Error parsing arguments: " + std::string(e.what());
    }

    return result;
}

ReshapeConfigurationResult set_reshape_method(
    const std::string& method_name,
    const std::vector<float>& args,
    const std::vector<int64_t>& shape,
    int64_t dim1,
    int64_t dim2,
    const std::vector<int64_t>& dims) {

    ReshapeConfigurationResult out;
    out.success = false;
    out.method_name = method_name;

    if (!args.empty()) {
        // if there arguments are list
        auto parsed = parse_list_arguments(method_name, args);
        if (!parsed.valid) {
            out.error_message = parsed.error_message;
            return out;
        }
        out.params = parsed.params;
        
    } else {
        //if arguments are individuals
        bool has_individual_data = !shape.empty() || 
                                   dim1 != 0 || 
                                   dim2 != -1 || 
                                   !dims.empty();
        
        if (!has_individual_data) {
            out.error_message = "No arguments provided for method: " + method_name;
            return out;
        }
        auto parsed = parse_individual_arguments(method_name, shape, dim1, dim2, dims);
        if (!parsed.valid) {
            out.error_message = parsed.error_message;
            return out;
        }
        out.params = parsed.params;
    }

    out.success = true;
    return out;
}
       




//     }

//     out.params = parsed.params;
//     out.success = true;
//     return out;
// }






        } // namespace dp_reshape
    } // namespace core
} // namespace contorchionist
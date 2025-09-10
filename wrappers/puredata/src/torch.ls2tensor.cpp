#include "m_pd.h"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <numeric>
#include <fstream>      
#include <iostream>   
#include "s_stuff.h" 
#include "../utils/include/pd_torch_types.h"
#include "../utils/include/pd_torch_utils.h"
#include "../utils/include/pd_global_state.h"
#include "../utils/include/pd_arg_parser.h"
#include "../utils/include/pd_torch_utils.h"
#include "../../../core/include/core_dp_ls2tensor.h"

static t_class *torch_ls2tensor_class;


//* ------------------ create a tensor with the data received ------------------//
static void torch_ls2tensor_format(t_torch_ls2tensor *x) {
    // prepare config
    contorchionist::core::dp_ls2tensor::FormatConfig config;
    config.data_buffer = x->data_buffer;
    config.tensor_shape = x->tensor_shape;
    config.sample_shape = x->sample_shape;
    config.has_target = x->has_target;
    config.target_size = x->target_size;
    config.device = x->device;

    // format the tensor
    auto result = contorchionist::core::dp_ls2tensor::Ls2TensorProcessor(config);

    if (!result.success) {
        pd_error(x, "torch.ls2tensor: %s", result.error_message.c_str());
        return;
    }
    // apply results
    x->tensors_struct->dataset_tensor = result.dataset_tensor;
    x->tensors_struct->target_tensor = result.target_tensor;

    // log
    if (x->verbose) {
        if (x->has_target) {
            post("torch.ls2tensor: Split data - Input: %lld elements/sample, Target: %lld elements/sample", 
                 result.sample_data_size, result.target_size);
        }
        post("torch.ls2tensor: Dataset tensor created - Shape: %s", 
             contorchionist::core::dp_ls2tensor::shape_to_string(
                 std::vector<int64_t>(x->tensors_struct->dataset_tensor.sizes().begin(),
                                      x->tensors_struct->dataset_tensor.sizes().end())
             ).c_str());
             
        if (x->has_target && x->tensors_struct->target_tensor.defined()) {
            post("torch.ls2tensor: Target tensor created - Shape: %s", 
                 contorchionist::core::dp_ls2tensor::shape_to_string(
                     std::vector<int64_t>(x->tensors_struct->target_tensor.sizes().begin(),
                                          x->tensors_struct->target_tensor.sizes().end())
                 ).c_str());
        }
        post("torch.ls2tensor: Tensor '%s' finalized successfully.", x->name->s_name);
    }

    // if shuffling is enabled
    if (x->shuffle) {
        auto shuffle_result = contorchionist::core::dp_ls2tensor::shuffle_tensors(
            x->tensors_struct->dataset_tensor,
            x->tensors_struct->target_tensor,
            x->device,
            x->has_target
        );
        //check shuffle result
        if (!shuffle_result.success) {
            pd_error(x, "torch.ls2tensor: %s", shuffle_result.error_message.c_str());
            return;
        }
        // log
        if (x->verbose) {
            if (shuffle_result.num_samples <= 1) {
                post("torch.ls2tensor: %s", shuffle_result.error_message.c_str());
            } else {
                if (x->has_target && x->tensors_struct->target_tensor.defined()) {
                    post("torch.ls2tensor: Dataset and target tensors shuffled with same permutation (%lld samples).", 
                        shuffle_result.num_samples);
                } else {
                    post("torch.ls2tensor: Dataset tensor shuffled (%lld samples).", 
                        shuffle_result.num_samples);
                }
            }
        }
    }
    // bang when done
    outlet_bang(x->x_out2);
}



//* ------------------ receive a list of floats to fill the internal buffer ------------------ */
static void torch_ls2tensor_append_list(t_torch_ls2tensor *x, t_symbol *s, int argc, t_atom *argv) {
   if (x->tensor_shape.empty()) {
       pd_error(x, "torch.ls2tensor: Shape not set. Use [shape( message first.");
       return;
   }

    std::vector<float> new_data;
    new_data.reserve(argc);

    // get atom values
    for (int i = 0; i < argc; ++i) {
        new_data.push_back(atom_getfloat(&argv[i]));
    }

    // append new data to buffer
    auto result = contorchionist::core::dp_ls2tensor::append_to_buffer(
        x->data_buffer,
        x->buffer_write_pos,
        x->tensor_shape,
        new_data
    );

    if (!result.success) {
        pd_error(x, "torch.ls2tensor: %s", result.error_message.c_str());
        return;
    }
    
    if (x->verbose) {
        post("torch.ls2tensor: Filled %zu / %zu", 
             result.current_position, result.total_capacity);
    }
    
    if (result.buffer_full) {
        torch_ls2tensor_format(x);
    }
}


//* --------------------- set the shape of the tensor --------------------- */
static void torch_ls2tensor_shape(t_torch_ls2tensor *x, t_symbol *s, int argc, t_atom *argv) {
    if (argc < 1) {
        pd_error(x, "torch.ls2tensor: Shape cannot be empty.");
        return;
    }

    // clear previous shapes
    x->tensor_shape.clear();
    x->sample_shape.clear();
    size_t total_size = 1;

    // calculate total size and set shapes
    for (int i = 0; i < argc; ++i) {
        int64_t dim = atom_getfloat(&argv[i]);
        if (dim <= 0) {
            pd_error(x, "torch.ls2tensor: All shape dimensions must be > 0.");
            return;
        }
        x->tensor_shape.push_back(dim);
        total_size *= dim;
    }

    if (x->tensor_shape.size() > 1) {
        x->sample_shape.assign(x->tensor_shape.begin() + 1, x->tensor_shape.end());
    } else {
        x->sample_shape.push_back(1); // Scalar sample
    }

    x->data_buffer.assign(total_size, 0.0f);
    x->buffer_write_pos = 0;

    if (x->verbose) {
        post("torch.ls2tensor: Shape set. Total size: %zu.", total_size);
    }
}

//* --------------------- set target flag and size --------------------- */
static void torch_ls2tensor_target(t_torch_ls2tensor *x, t_floatarg f, t_floatarg size) {
    x->has_target = (f != 0);
    if (x->has_target) {
        if (size <= 0) {
            pd_error(x, "torch.ls2tensor: Target size must be a positive integer.");
            x->has_target = false;
            return;
        }
        x->target_size = static_cast<int64_t>(size);
    } else {
        x->target_size = 0;
    }

    if (x->verbose) {
        post("torch.ls2tensor: Target set to %s, size %lld", x->has_target ? "true" : "false", x->target_size);
    }
}

//* ------------------ clear the internal buffer and tensors ------------------ */
static void torch_ls2tensor_clear(t_torch_ls2tensor *x) {
    x->buffer_write_pos = 0;
    std::fill(x->data_buffer.begin(), x->data_buffer.end(), 0.0f);
    x->tensors_struct->dataset_tensor = torch::Tensor();
    x->tensors_struct->target_tensor = torch::Tensor();
    if(x->verbose) {
        post("torch.ls2tensor: Buffer cleared.");
    }
}

//* ---------------------- returns the tensor and its shape -------------------//
static void torch_ls2tensor_output (t_torch_ls2tensor *x){

    //check if tensors_struct is available
    if (!x->tensors_struct) {
        pd_error(x, "torch.ls2tensor: No tensors available for output.");
        return;
    }

    // check if dataset_tensor is available and full filled
    if (!x->tensors_struct->dataset_tensor.defined() || x->tensors_struct->dataset_tensor.numel() == 0) {
        pd_error(x, "torch.ls2tensor: Dataset tensor is empty or not created.");
        return;
    }

    // check if the tensor shape is defined
    if (x->tensor_shape.empty()) {
        pd_error(x, "torch.ls2tensor: No shape defined for tensor.");
        return;
    }

    size_t expected_total_size = std::accumulate(
        x->tensor_shape.begin(), 
        x->tensor_shape.end(), 
        (size_t)1, 
        std::multiplies<size_t>()
    );

    if (x->buffer_write_pos < expected_total_size) {
        pd_error(x, "torch.ls2tensor: Tensor not completely filled. Current: %zu / %zu elements.", 
                 x->buffer_write_pos, expected_total_size);
        post("torch.ls2tensor: Use [shape( message and fill with data lists first.");
        return;
    }

    // if target was set
    if (x->has_target) {
        int64_t dataset_size = x->tensors_struct->dataset_tensor.numel();
        int64_t target_size = 0;
        // Get target size if defined
        if (x->tensors_struct->target_tensor.defined()) {
            target_size = x->tensors_struct->target_tensor.numel();
        }
        // Check if dataset + target = total size expected
        if (dataset_size + target_size != (int64_t)expected_total_size) {
            pd_error(x, "torch.ls2tensor: Combined tensor size mismatch. Expected: %zu, Got: %lld (dataset) + %lld (target) = %lld", 
                     expected_total_size, dataset_size, target_size, dataset_size + target_size);
            return;
        }
        // Check if target tensor is defined
        if (!x->tensors_struct->target_tensor.defined()) {
            pd_error(x, "torch.ls2tensor: Target tensor not created despite target flag being set.");
            return;
        }
    // if target was not set
    } else {
        // check if dataset tensor size matches total expected size
        if (x->tensors_struct->dataset_tensor.numel() != (int64_t)expected_total_size) {
            pd_error(x, "torch.ls2tensor: Dataset tensor size mismatch. Expected: %zu, Got: %lld", 
                     expected_total_size, x->tensors_struct->dataset_tensor.numel());
            return;
        }
    }
    //send tensor to outlet
    try {
        at::Tensor output_tensor = x->tensors_struct->dataset_tensor;

        if (x->recursive) {
            // multi-output mode: send tensor and shape
            PdTorchUtils::send_tensor_shape(output_tensor, x->x_out1);
            PdTorchUtils::send_tensor_recursive(output_tensor, x->x_out1, 0);
            // send target if it exists
            if (x->has_target && x->tensors_struct->target_tensor.defined()) {
                t_atom target_prefix;
                SETSYMBOL(&target_prefix, gensym("target"));
                outlet_anything(x->x_out1, gensym("tensor_type"), 1, &target_prefix);
                PdTorchUtils::send_tensor_shape(x->tensors_struct->target_tensor, x->x_out1);
                PdTorchUtils::send_tensor_recursive(x->tensors_struct->target_tensor, x->x_out1, 0);
            }
        }
        else{
            // flatten the output tensor and convert to std::vector<float>
            at::Tensor flat = output_tensor.contiguous().view({-1});
             t_atom dataset_prefix;
            SETSYMBOL(&dataset_prefix, gensym("tensor"));
            outlet_anything(x->x_out1, gensym("tensor_type"), 1, &dataset_prefix);
            PdTorchUtils::send_tensor_1d(flat, x->x_out1);

            if (x->has_target && x->tensors_struct->target_tensor.defined()) {
            at::Tensor flat_target = x->tensors_struct->target_tensor.contiguous().view({-1});
            
                t_atom target_prefix;
                SETSYMBOL(&target_prefix, gensym("target"));
                outlet_anything(x->x_out1, gensym("tensor_type"), 1, &target_prefix);
                PdTorchUtils::send_tensor_1d(flat_target, x->x_out1);
            }
        }
        if (x->verbose) {
            if (x->has_target && x->tensors_struct->target_tensor.defined()) {
                post("torch.ls2tensor: Tensors output - Dataset: %lld elements, Target: %lld elements", 
                    output_tensor.numel(), x->tensors_struct->target_tensor.numel());
            } else {
                post("torch.ls2tensor: Dataset tensor output - %lld elements", output_tensor.numel());
            }
        }
    } catch (const c10::Error& e) {
        pd_error(x, "torch.ls2tensor: LibTorch error during output: %s", e.what());
    } catch (const std::exception& e) {
        pd_error(x, "torch.ls2tensor: Standard error during output: %s", e.what());
    }
}



//* ---------------------- save tensor to a text file ------------------
static void torch_ls2tensor_save(t_torch_ls2tensor *x, t_symbol *filename, t_symbol *format) {
    if (!x->tensors_struct->dataset_tensor.defined()) {
        pd_error(x, "torch.ls2tensor: No tensor to save. Fill the tensor with data first.");
        return;
    }
    const char *canvas_dir = canvas_getdir(x->m_canvas)->s_name; // get the current directory of the Pure Data canvas
    char fullpath[MAXPDSTRING];
    snprintf(fullpath, MAXPDSTRING, "%s/%s", canvas_dir, filename->s_name);
    char normalized[MAXPDSTRING];
    sys_unbashfilename(fullpath, normalized);

    std::ofstream outfile(normalized);
    if (!outfile.is_open()) {
        pd_error(x, "torch.ls2tensor: Could not open file for writing: %s", normalized);
        return;
    }
    try {
        auto dataset_tensor_cpu = x->tensors_struct->dataset_tensor.to(torch::kCPU);
        auto target_tensor_cpu = x->tensors_struct->target_tensor.defined() ? 
                                x->tensors_struct->target_tensor.to(torch::kCPU) : 
                                torch::Tensor();

        int64_t num_samples = dataset_tensor_cpu.size(0);
        int64_t sample_data_size = dataset_tensor_cpu.numel() / num_samples;

        for (int64_t i = 0; i < num_samples; ++i) {
            // Escrever parte do dataset
            auto sample_tensor = dataset_tensor_cpu.select(0, i).flatten();
            for (int64_t j = 0; j < sample_data_size; ++j) {
                outfile << sample_tensor[j].item<float>();
                if (j < sample_data_size - 1 || target_tensor_cpu.defined()) {
                    outfile << " ";
                }
            }
            if (target_tensor_cpu.defined()) {
                auto target_sample_tensor = target_tensor_cpu.select(0, i).flatten();
                int64_t target_sample_size = target_sample_tensor.numel();
                for (int64_t j = 0; j < target_sample_size; ++j) {
                    outfile << target_sample_tensor[j].item<float>();
                    if (j < target_sample_size - 1) {
                        outfile << " ";
                    }
                }
            }
            outfile << "\n";
        }
        
        outfile.close();
        
        if (x->verbose) {
            if (x->verbose) {
                post("torch.ls2tensor: Canvas dir: '%s'", canvas_dir);
                post("torch.ls2tensor: Filename: '%s'", filename->s_name);
                post("torch.ls2tensor: Full path: '%s'", fullpath);
                post("torch.ls2tensor: Normalized: '%s'", normalized);
            }
        }

    } catch (const c10::Error& e) {
        pd_error(x, "torch.ls2tensor: LibTorch error during save: %s", e.what());
        outfile.close();
    } catch (const std::exception& e) {
        pd_error(x, "torch.ls2tensor: Standard error during save: %s", e.what());
        outfile.close();
    }
}

//* ------------------------ load dataset ------------------------ *//
static void torch_ls2tensor_load(t_torch_ls2tensor *x, t_symbol *filename, t_symbol *format) {
    if (!filename) {
        pd_error(x, "torch.ls2tensor: No filename provided to load.");
        return;
    }

    //check if tensors_struct is available
    if (!x->tensors_struct) {
        pd_error(x, "torch.ls2tensor: Tensor structure not initialized.");
        return;
    }

    // Initialize tensors (dataset and target)
    x->tensors_struct->dataset_tensor = torch::Tensor();
    x->tensors_struct->target_tensor = torch::Tensor();

    char dirresult[MAXPDSTRING]; // buffer to store the directory address
    char *nameresult; // buffer to store the name of the file
    int fd; // file descriptor
    std::string fname(filename->s_name); // file name
    const char *canvas_dir = canvas_getdir(x->m_canvas)->s_name; // get the current directory of the Pure Data canvas
    fd = open_via_path(canvas_dir, fname.c_str(), "", dirresult, &nameresult, MAXPDSTRING, 1); // open the file
    //check if file descriptor is valid
    if (fd < 0) {
        pd_error(x, "torch.ls2tensor: File not found '%s'", fname.c_str());
        return;
    }
    sys_close(fd); // close the file descriptor (just need the address of the file)
    char fullpath[MAXPDSTRING]; // buffer to store the full path of the file
    snprintf(fullpath, MAXPDSTRING, "%s/%s", dirresult, nameresult); // create the full path of the file
    char normalized[MAXPDSTRING]; // buffer to store the normalized path of the file (without special characters)
    sys_unbashfilename(fullpath, normalized); // normalize the path (remove special characters)

    std::ifstream infile(normalized); // open the file for reading
    if (!infile.is_open()) { // sends an error if the file cannot be opened and return
        pd_error(x, "torch.ls2tensor: Failed to open the file '%s' for reading.", normalized);
        return;
    }
    // Read the file contents into a vector of vectors
    std::vector<std::vector<float>> data;
    std::string line;
    // Read each line from the file
    while (std::getline(infile, line)) {
        // Remove trailing semicolon if exists
        if (!line.empty() && line.back() == ';') {
            line.pop_back();
        }
        // Split the line into tokens
        std::istringstream iss(line);
        std::vector<float> row;
        std::string token;
        // Read each token from the line
        while (iss >> token) {
            try {
                row.push_back(std::stof(token));
            } catch (const std::invalid_argument& ia) {
                pd_error(x, "torch.ls2tensor: Invalid number format in file '%s': '%s'", filename->s_name, token.c_str());
            } catch (const std::out_of_range& oor) {
                pd_error(x, "torch.ls2tensor: Number out of range in file '%s': '%s'", filename->s_name, token.c_str());
            }
        }
        // If the row is not empty, add it to the data rows
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    infile.close();
    // Check if data is empty
    if (data.empty()) {
        pd_error(x, "torch.ls2tensor: Failed to load dataset '%s' (empty file or invalid format).", fname.c_str());
        return;
    }
    // Check if all rows have the same size
    size_t row_size = data[0].size();
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i].size() != row_size) {
            pd_error(x, "torch.ls2tensor: Inconsistent row sizes in file '%s'. Row %zu has %zu elements, expected %zu.", 
                     fname.c_str(), i, data[i].size(), row_size);
            return;
        }
    }
    // Check if target size is valid
    if (x->has_target && x->target_size > 0 && (int64_t)row_size <= x->target_size) {
        pd_error(x, "torch.ls2tensor: Target size (%lld) is >= number of columns in file (%zu).",
                 x->target_size, row_size);
        return;
    }
    // Check if tensor shape is valid
    if (x->tensor_shape.empty()) {
        x->tensor_shape = {(int64_t)data.size(), (int64_t)row_size};
        x->sample_shape.clear();
        if (x->tensor_shape.size() > 1) {
            x->sample_shape.assign(x->tensor_shape.begin() + 1, x->tensor_shape.end());
        } else {
            x->sample_shape.push_back(1);
        }
        // Log the auto-detected shape
        if (x->verbose) {
            post("torch.ls2tensor: Auto-detected shape [%lld, %lld] from file '%s'", 
                 x->tensor_shape[0], x->tensor_shape[1], fname.c_str());
        }
    }
    // Check if expected rows and columns are valid
    int64_t expected_rows = x->tensor_shape[0];
    int64_t expected_cols = std::accumulate(
        x->sample_shape.begin(), 
        x->sample_shape.end(), 
        (int64_t)1, 
        std::multiplies<int64_t>()
    );
    // Check if actual rows match expected rows
    if ((int64_t)data.size() != expected_rows) {
        pd_error(x, "torch.ls2tensor: File has %zu rows but tensor shape expects %lld rows.", 
                 data.size(), expected_rows);
        return;
    }
    // Check if actual columns match expected columns
    if ((int64_t)row_size != expected_cols) {
        pd_error(x, "torch.ls2tensor: File has %zu columns but tensor shape expects %lld columns.", 
                 row_size, expected_cols);
        return;
    }

    // get total size
    size_t total_size = data.size() * row_size;
    x->data_buffer.clear();
    x->data_buffer.reserve(total_size);
    // Flatten the 2D data into the 1D buffer
    for (const auto& row : data) {
        for (float value : row) {
            x->data_buffer.push_back(value);
        }
    }
    // Update buffer write position
    x->buffer_write_pos = total_size;

    if (x->verbose) {
        post("torch.ls2tensor: Dataset loaded successfully '%s' (%zu rows × %zu cols = %zu elements)", 
             fname.c_str(), data.size(), row_size, total_size);
    }

    if (x->data_buffer.empty()) {
        pd_error(x, "torch.ls2tensor: Data buffer is empty after loading.");
        return;
    }

    if (x->tensor_shape.empty()) {
        pd_error(x, "torch.ls2tensor: Tensor shape is empty after loading.");
        return;
    }

    if (x->sample_shape.empty()) {
        pd_error(x, "torch.ls2tensor: Sample shape is empty after loading.");
        return;
    }

    size_t buffer_size = x->data_buffer.size();
    size_t expected_size = std::accumulate(
        x->tensor_shape.begin(), 
        x->tensor_shape.end(), 
        (size_t)1, 
        std::multiplies<size_t>()
    );

    if (buffer_size != expected_size) {
        pd_error(x, "torch.ls2tensor: Buffer size mismatch. Buffer: %zu, Expected: %zu", 
                 buffer_size, expected_size);
        return;
    }

    size_t sample_size = std::accumulate(
        x->sample_shape.begin(), 
        x->sample_shape.end(), 
        (size_t)1, 
        std::multiplies<size_t>()
    );

    if (sample_size != row_size) {
        pd_error(x, "torch.ls2tensor: Sample shape inconsistent. Sample size: %zu, Row size: %zu", 
                 sample_size, row_size);
        return;
    }

    try {
        torch_ls2tensor_format(x);
        
        if (x->verbose) {
            post("torch.ls2tensor: Tensors created automatically from loaded data.");
        }
        
    } catch (const c10::Error& e) {
        pd_error(x, "torch.ls2tensor: LibTorch error during tensor creation: %s", e.what());
    } catch (const std::out_of_range& e) {
        pd_error(x, "torch.ls2tensor: Memory access error during tensor creation: %s", e.what());
    } catch (const std::exception& e) {
        pd_error(x, "torch.ls2tensor: Standard error during tensor creation: %s", e.what());
    } catch (...) {
        pd_error(x, "torch.ls2tensor: Unknown error during tensor creation.");
    }
}


//* ------------ constructor ------------
static void *torch_ls2tensor_new(t_symbol *s, int argc, t_atom *argv) {
    t_torch_ls2tensor *x = (t_torch_ls2tensor *)pd_new(torch_ls2tensor_class);

    if (argc < 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "torch.ls2tensor: Please provide a unique name for the tensor.");
        return NULL;
    }
    x->name = atom_getsymbol(argv);

    if (PDGlobalState::ls2tensor_registry.count(x->name->s_name) > 0) {
        pd_error(x, "torch.ls2tensor: Tensor with name '%s' already exists.", x->name->s_name);
        return NULL;
    }

    x->tensors_struct = new PdTorchTensors();
    x->has_target = false;
    x->target_size = 0;
    x->buffer_write_pos = 0;
    x->verbose = false;
    x->recursive = false;
    x->device = torch::kCPU;
    x->m_canvas = canvas_getcurrent(); 

    //---------- parse arguments ----------
    pd_utils::ArgParser parser(argc - 1, argv + 1, (t_object*)x);

    // verbose
    x->verbose = parser.has_flag("verbose v");
    // tensor output mode
    x->recursive = parser.has_flag("multioutput mout");
    if (x->verbose) {
        post("torch.ls2tensor: output mode=%s", x->recursive ? "structured" : "flattened");
    } 
    //target
    x->has_target = parser.has_flag("target t");
    if (x->has_target) {
        x->target_size = parser.get_float("target t", 0);
    }
    // shuffle mode
    x->shuffle = parser.has_flag("shuffle s");

    // tensor shape
    auto input_shape_f = parser.get_float_list("shape", {});
    x->tensor_shape.clear();
    x->sample_shape.clear();
    // check if the tensor shape is defined
    if (!input_shape_f.empty()) {
        for (auto v : input_shape_f) {
            int64_t dim = (int64_t)v;
            if (dim <= 0) {
                pd_error(x, "torch.ls2tensor: All shape dimensions must be > 0. Got: %lld", dim);
                delete x->tensors_struct;
                return NULL;
            }
            x->tensor_shape.push_back(dim);
        }
        // shape sample
        if (x->tensor_shape.size() > 1) {
            x->sample_shape.assign(x->tensor_shape.begin() + 1, x->tensor_shape.end());
        } else {
            x->sample_shape.push_back(1); // Scalar sample
        }
        // create buffer
        size_t total_size = std::accumulate(
            x->tensor_shape.begin(), 
            x->tensor_shape.end(), 
            (size_t)1, 
            std::multiplies<size_t>()
        );
        x->data_buffer.assign(total_size, 0.0f);
        // log initial shape
        if (x->verbose) {
            std::string shape_str = "[";
            for (size_t i = 0; i < x->tensor_shape.size(); ++i) {
                if (i > 0) shape_str += ", ";
                shape_str += std::to_string(x->tensor_shape[i]);
            }
            shape_str += "]";
            post("torch.ls2tensor: Initial shape set to %s (total: %zu elements)", 
                 shape_str.c_str(), total_size);
        }
    }

    // global register
    PDGlobalState::ls2tensor_registry[x->name->s_name] = x;
    pd_bind((t_pd *)x, x->name);
    // create outlets
    x->x_out1 = outlet_new(&x->x_obj, &s_anything);
    x->x_out2 = outlet_new(&x->x_obj, &s_bang);

    //LOG
    if (x->verbose) {
        std::string shape_str;
        if (x->tensor_shape.empty()) {
            shape_str = "not set";
        } else {
            shape_str = "[";
            for (size_t i = 0; i < x->tensor_shape.size(); ++i) {
                if (i > 0) shape_str += ", ";
                shape_str += std::to_string(x->tensor_shape[i]);
            }
            shape_str += "]";
        }
        post("torch.ls2tensor: Created tensor object '%s' (shape: %s, mode: %s)", 
            x->name->s_name,
            shape_str.c_str(),
            x->recursive ? "structured" : "flattened");
    }
    return (void *)x;
}


//* -------------- destructor ---------------
static void torch_ls2tensor_free(t_torch_ls2tensor *x) {
    pd_unbind((t_pd *)x, x->name);
    PDGlobalState::ls2tensor_registry.erase(x->name->s_name);
    delete x->tensors_struct; // free the tensors struct
    x->tensors_struct = nullptr;
    outlet_free(x->x_out1);
    outlet_free(x->x_out2);
}

extern "C" void setup_torch0x2els2tensor(void) {
    torch_ls2tensor_class = class_new(
        gensym("torch.ls2tensor"),
        (t_newmethod)torch_ls2tensor_new,
        (t_method)torch_ls2tensor_free,
        sizeof(t_torch_ls2tensor),
        CLASS_DEFAULT,
        A_GIMME,
        0);

    PDGlobalState::torch_ls2tensor_class = torch_ls2tensor_class;

    class_addlist(torch_ls2tensor_class, torch_ls2tensor_append_list);
    class_addbang(torch_ls2tensor_class, (t_method)torch_ls2tensor_output);
    class_addmethod(torch_ls2tensor_class, (t_method)torch_ls2tensor_shape, gensym("shape"), A_GIMME, 0);
    class_addmethod(torch_ls2tensor_class, (t_method)torch_ls2tensor_target, gensym("target"), A_FLOAT, A_DEFFLOAT, 0);
    class_addmethod(torch_ls2tensor_class, (t_method)torch_ls2tensor_clear, gensym("clear"), A_NULL, 0);
    class_addmethod(torch_ls2tensor_class, (t_method)torch_ls2tensor_save, gensym("save"), A_SYMBOL, 0);
    class_addmethod(torch_ls2tensor_class, (t_method)torch_ls2tensor_load, gensym("load"), A_SYMBOL, 0);
}

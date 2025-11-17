\
#ifndef PD_ARG_PARSER_H
#define PD_ARG_PARSER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include "m_pd.h" // Required for t_atom, t_symbol, etc.

namespace pd_utils {

// Type alias for the value associated with a flag
using ArgValue = std::variant<bool, float, std::string, std::vector<float>>;

// Type alias for the map storing parsed arguments
using ParsedArgsMap = std::unordered_map<std::string, ArgValue>;

class ArgParser {
public:
    // Constructor that takes Pd\'s argc and argv
    ArgParser(int argc, t_atom *argv, t_object *owner = nullptr);

    // Checks if a flag was present in the arguments
    bool has_flag(const std::string& flag_name) const;

    // Gets a boolean value for a flag.
    // Returns true if the flag was present without a value.
    // Returns default_value if the flag is not present or has an incompatible type.
    bool get_bool(const std::string& flag_name, bool default_value = false) const;

    // Gets a float value for a flag.
    // Returns default_value if the flag is not present or has an incompatible type.
    float get_float(const std::string& flag_name, float default_value = 0.0f) const;

    // Gets a string value for a flag.
    // Returns default_value if the flag is not present or has an incompatible type.
    std::string get_string(const std::string& flag_name, const std::string& default_value = "") const;

    // Gets a list of floats for a flag.
    // Returns default_value if the flag is not present or has an incompatible type.
    std::vector<float> get_float_list(const std::string& flag_name, const std::vector<float>& default_value = {}) const;

private:
    ParsedArgsMap parsed_args_;
    t_object* owner_obj_; // For pd_error context

    void parse(int argc, t_atom *argv);

    // Helper for logging messages
    void log_error(const char* fmt, ...) const;
    void log_post(const char* fmt, ...) const;

    template <typename T>
    T get_value_or_default(const std::string& flag_name, T default_value, const char* expected_type_name) const {
        auto it = parsed_args_.find(flag_name);
        if (it != parsed_args_.end()) {
            if (std::holds_alternative<T>(it->second)) {
                return std::get<T>(it->second);
            } else {
                std::string actual_type_name = "unknown";
                if (std::holds_alternative<bool>(it->second)) actual_type_name = "boolean";
                else if (std::holds_alternative<float>(it->second)) actual_type_name = "float";
                else if (std::holds_alternative<std::string>(it->second)) actual_type_name = "string";
                else if (std::holds_alternative<std::vector<float>>(it->second)) actual_type_name = "float list";
                
                log_error("ArgParser: Type mismatch for flag \'@%s\'. Expected %s but got %s. Returning default.", 
                          flag_name.c_str(), expected_type_name, actual_type_name.c_str());
            }
        }
        return default_value;
    }
};

} // namespace pd_utils

#endif // PD_ARG_PARSER_H

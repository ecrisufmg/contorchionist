#include "pd_arg_parser.h" // Assumes wrappers/pd/utils/include is in compiler include paths
#include <cstring> // For strcmp
#include <cstdarg> // For va_list, va_start, va_end
#include <cstdio>  // For vsnprintf

namespace pd_utils {

// Helper to convert t_symbol to std::string, removing leading '@' or '-'
static std::string normalize_flag_name(t_symbol *s) {
    if (!s || !s->s_name) {
        return "";
    }
    std::string name = s->s_name;
    if (!name.empty() && (name[0] == '@' || name[0] == '-')) {
        return name.substr(1);
    }
    return name;
}

ArgParser::ArgParser(int argc, t_atom *argv, t_object *owner) : owner_obj_(owner) {
    parse(argc, argv);
}

void ArgParser::log_error(const char* fmt, ...) const {
    char buf[MAXPDSTRING]; // MAXPDSTRING is from m_pd.h
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, MAXPDSTRING, fmt, ap);
    va_end(ap);
    if (owner_obj_) {
        // Pass the formatted buffer to pd_error. 
        // pd_error itself is variadic, so we pass buf as the format string, 
        // and no further arguments if buf contains the full message.
        pd_error(owner_obj_, "%s", buf); 
    } else {
        pd_error(nullptr, "%s", buf);
    }
}

void ArgParser::log_post(const char* fmt, ...) const {
    char buf[MAXPDSTRING]; // MAXPDSTRING is from m_pd.h
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, MAXPDSTRING, fmt, ap);
    va_end(ap);
    // post is variadic, similar to pd_error.
    post("%s", buf); 
}

void ArgParser::parse(int argc, t_atom *argv) {
    int i = 0;
    while (i < argc) {
        if (argv[i].a_type == A_SYMBOL) {
            std::string flag_name_str = normalize_flag_name(argv[i].a_w.w_symbol);
            if (flag_name_str.empty()) {
                log_error("ArgParser: Encountered an empty or invalid symbol flag at arg %d.", i);
                i++;
                continue;
            }

            if (i + 1 < argc) {
                t_atom *next_atom = &argv[i + 1];
                if (next_atom->a_type == A_FLOAT) {
                    std::vector<float> float_list;
                    float_list.push_back(next_atom->a_w.w_float);
                    i += 2; 
                    while (i < argc && argv[i].a_type == A_FLOAT) {
                        float_list.push_back(argv[i].a_w.w_float);
                        i++;
                    }
                    if (float_list.size() == 1) {
                        parsed_args_[flag_name_str] = float_list[0];
                    } else {
                        parsed_args_[flag_name_str] = float_list;
                    }
                    continue;
                } else if (next_atom->a_type == A_SYMBOL) {
                    std::string next_sym_str = atom_getsymbol(next_atom)->s_name;
                    if (!next_sym_str.empty() && (next_sym_str[0] == '@' || next_sym_str[0] == '-')) {
                        parsed_args_[flag_name_str] = true; 
                        i++; 
                    } else {
                        parsed_args_[flag_name_str] = next_sym_str;
                        i += 2; 
                    }
                } else {
                    parsed_args_[flag_name_str] = true;
                    i++; 
                }
            } else {
                parsed_args_[flag_name_str] = true;
                i++; 
            }
        } else {
            char buf[MAXPDSTRING];
            atom_string(&argv[i], buf, MAXPDSTRING);
            log_error("ArgParser: Ignoring non-flag argument \'%s\' at index %d.", buf, i);
            i++;
        }
    }
}

bool ArgParser::has_flag(const std::string& flag_name) const {
    return parsed_args_.count(flag_name);
}

bool ArgParser::get_bool(const std::string& flag_name, bool default_value) const {
    auto it = parsed_args_.find(flag_name);
    if (it != parsed_args_.end()) {
        if (std::holds_alternative<bool>(it->second)) {
            return std::get<bool>(it->second);
        }
        if (std::holds_alternative<float>(it->second) || 
            std::holds_alternative<std::string>(it->second) || 
            std::holds_alternative<std::vector<float>>(it->second)) {
            log_error("ArgParser: Type mismatch for bool flag \'@%s\'. Flag has a value. Returning default.", flag_name.c_str());
            return default_value;
        }
        return true; 
    }
    return default_value;
}

float ArgParser::get_float(const std::string& flag_name, float default_value) const {
    return get_value_or_default<float>(flag_name, default_value, "float");
}

std::string ArgParser::get_string(const std::string& flag_name, const std::string& default_value) const {
    return get_value_or_default<std::string>(flag_name, default_value, "string");
}

std::vector<float> ArgParser::get_float_list(const std::string& flag_name, const std::vector<float>& default_value) const {
    return get_value_or_default<std::vector<float>>(flag_name, default_value, "float list");
}

} // namespace pd_utils

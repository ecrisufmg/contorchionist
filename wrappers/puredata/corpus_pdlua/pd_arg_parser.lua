-- pd_arg_parser.lua
-- Utilitário para parsing de argumentos de objetos pd-lua com flags (@ ou -)
-- Inspirado no ArgParser C++ para Pure Data

local ArgParser = {}
ArgParser.__index = ArgParser

-- Remove '@' ou '-' do início de uma string
local function normalize_flag_name(str)
    if type(str) == "string" and #str > 0 then
        if str:sub(1, 1) == "@" or str:sub(1, 1) == "-" then
            return str:sub(2)
        end
    end
    return str
end

-- Verifica se um átomo é uma flag (começa com @ ou -)
local function is_flag(atom)
    if type(atom) == "string" then
        local first_char = atom:sub(1, 1)
        return first_char == "@" or first_char == "-"
    end
    return false
end

-- Cria um novo parser a partir de uma tabela de atoms
function ArgParser:new(atoms)
    local self = setmetatable({}, ArgParser)
    self.parsed_args = {}
    self.positional_args = {}
    self:parse(atoms or {})
    return self
end

-- Faz o parsing dos argumentos
function ArgParser:parse(atoms)
    local i = 1
    
    -- Primeiro, coleta argumentos posicionais (antes de qualquer flag)
    while i <= #atoms do
        local current = atoms[i]
        
        -- Para quando encontrar a primeira flag
        if type(current) == "string" and is_flag(current) then
            break
        end
        
        -- Adiciona à lista de argumentos posicionais
        table.insert(self.positional_args, current)
        i = i + 1
    end
    
    -- Agora processa argumentos nomeados (flags)
    while i <= #atoms do
        local current = atoms[i]
        
        -- Se for uma string que começa com @ ou -
        if type(current) == "string" and is_flag(current) then
            local flag_name = normalize_flag_name(current)
            
            -- Verifica se há um próximo argumento
            if i + 1 <= #atoms then
                local next_atom = atoms[i + 1]
                
                -- Se o próximo também é uma flag, este é um bool
                if type(next_atom) == "string" and is_flag(next_atom) then
                    self.parsed_args[flag_name] = true
                    i = i + 1
                else
                    -- Coleta valores mistos (números ou strings que não são flags)
                    local value_list = {}
                    i = i + 1 -- Avança para o primeiro valor
                    
                    while i <= #atoms do
                        local val = atoms[i]
                        -- Se encontrar uma flag, para
                        if type(val) == "string" and is_flag(val) then
                            break
                        end
                        table.insert(value_list, val)
                        i = i + 1
                    end
                    
                    -- Se for apenas um valor, salva como único
                    if #value_list == 1 then
                        self.parsed_args[flag_name] = value_list[1]
                    else
                        self.parsed_args[flag_name] = value_list
                    end
                end
            else
                -- Flag sem valor = boolean true
                self.parsed_args[flag_name] = true
                i = i + 1
            end
        else
            -- Se chegou aqui, algo está errado (não deveria ter não-flag após flags)
            pd.post(string.format("ArgParser: Warning - non-flag argument '%s' found after flags at index %d", tostring(current), i))
            i = i + 1
        end
    end
end

-- Divide uma string de nomes de flags separadas por espaço
local function split_flag_names(flag_names)
    local flags = {}
    for flag in flag_names:gmatch("%S+") do
        table.insert(flags, flag)
    end
    return flags
end

-- Encontra a primeira flag que existe nos argumentos
function ArgParser:find_first_matching_flag(flag_names)
    local flags = split_flag_names(flag_names)
    for _, flag in ipairs(flags) do
        if self.parsed_args[flag] then
            return flag
        end
    end
    return nil
end

-- Verifica se uma flag existe (aceita múltiplos nomes separados por espaço)
function ArgParser:has_flag(flag_names)
    return self:find_first_matching_flag(flag_names) ~= nil
end

-- Obtém um valor booleano
function ArgParser:get_bool(flag_names, default_value)
    if default_value == nil then default_value = false end
    
    local matching_flag = self:find_first_matching_flag(flag_names)
    if not matching_flag then
        return default_value
    end
    
    local value = self.parsed_args[matching_flag]
    if type(value) == "boolean" then
        return value
    end
    
    -- Se tem valor mas não é bool, retorna true (flag existe)
    return true
end

-- Obtém um valor float
function ArgParser:get_float(flag_names, default_value)
    if default_value == nil then default_value = 0.0 end
    
    local matching_flag = self:find_first_matching_flag(flag_names)
    if not matching_flag then
        return default_value
    end
    
    local value = self.parsed_args[matching_flag]
    if type(value) == "number" then
        return value
    end
    
    pd.post(string.format("ArgParser: Type mismatch for float flag '@%s'. Returning default.", matching_flag))
    return default_value
end

-- Obtém um valor string
function ArgParser:get_string(flag_names, default_value)
    if default_value == nil then default_value = "" end
    
    local matching_flag = self:find_first_matching_flag(flag_names)
    if not matching_flag then
        return default_value
    end
    
    local value = self.parsed_args[matching_flag]
    if type(value) == "string" then
        return value
    end
    
    pd.post(string.format("ArgParser: Type mismatch for string flag '@%s'. Returning default.", matching_flag))
    return default_value
end

-- Obtém o valor bruto de uma flag (pode ser qualquer tipo)
function ArgParser:get_value(flag_names, default_value)
    local matching_flag = self:find_first_matching_flag(flag_names)
    if not matching_flag then
        return default_value
    end
    return self.parsed_args[matching_flag]
end

-- Obtém uma lista de floats
function ArgParser:get_float_list(flag_names, default_value)
    if default_value == nil then default_value = {} end
    
    local matching_flag = self:find_first_matching_flag(flag_names)
    if not matching_flag then
        return default_value
    end
    
    local value = self.parsed_args[matching_flag]
    if type(value) == "table" then
        return value
    elseif type(value) == "number" then
        return {value}
    end
    
    pd.post(string.format("ArgParser: Type mismatch for float list flag '@%s'. Returning default.", matching_flag))
    return default_value
end

-- Obtém uma lista de strings
function ArgParser:get_string_list(flag_names, default_value)
    if default_value == nil then default_value = {} end
    
    local matching_flag = self:find_first_matching_flag(flag_names)
    if not matching_flag then
        return default_value
    end
    
    local value = self.parsed_args[matching_flag]
    if type(value) == "table" then
        return value
    elseif type(value) == "string" then
        return {value}
    end
    
    pd.post(string.format("ArgParser: Type mismatch for string list flag '@%s'. Returning default.", matching_flag))
    return default_value
end

-- ========== Métodos para argumentos posicionais ==========

-- Retorna o número de argumentos posicionais
function ArgParser:get_positional_count()
    return #self.positional_args
end

-- Retorna todos os argumentos posicionais
function ArgParser:get_all_positional()
    return self.positional_args
end

-- Retorna um argumento posicional específico (1-indexed)
function ArgParser:get_positional(index, default_value)
    if index < 1 or index > #self.positional_args then
        return default_value
    end
    return self.positional_args[index]
end

-- Retorna um argumento posicional como float (1-indexed)
function ArgParser:get_positional_float(index, default_value)
    if default_value == nil then default_value = 0.0 end
    
    local value = self:get_positional(index)
    if value == nil then
        return default_value
    end
    
    if type(value) == "number" then
        return value
    end
    
    pd.post(string.format("ArgParser: Type mismatch for positional arg %d (expected number). Returning default.", index))
    return default_value
end

-- Retorna um argumento posicional como string (1-indexed)
function ArgParser:get_positional_string(index, default_value)
    if default_value == nil then default_value = "" end
    
    local value = self:get_positional(index)
    if value == nil then
        return default_value
    end
    
    if type(value) == "string" then
        return value
    end
    
    pd.post(string.format("ArgParser: Type mismatch for positional arg %d (expected string). Returning default.", index))
    return default_value
end

return ArgParser

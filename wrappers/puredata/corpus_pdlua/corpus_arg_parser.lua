-- corpus_arg_parser.lua
-- Utilitário para parsing de argumentos de objetos pd-lua com flags (@ ou -)
-- Inspirado no ArgParser C++ para Pure Data
-- Renomeado para evitar conflito com ctgui

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

-- Retorna o valor de uma flag (ou nil se não existir)
function ArgParser:get(flag_name)
    return self.parsed_args[flag_name]
end

-- Retorna o valor de uma flag ou um valor padrão
function ArgParser:get_or(flag_name, default_value)
    local val = self.parsed_args[flag_name]
    if val == nil then return default_value end
    return val
end

-- Retorna um argumento posicional pelo índice (1-based)
function ArgParser:get_pos(index)
    return self.positional_args[index]
end

-- Retorna o número de argumentos posicionais
function ArgParser:count_pos()
    return #self.positional_args
end

return ArgParser
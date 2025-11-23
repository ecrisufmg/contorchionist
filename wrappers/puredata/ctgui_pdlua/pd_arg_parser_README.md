# pd_arg_parser.lua - Documentação

Parser de argumentos para objetos pd-lua com suporte a argumentos posicionais e nomeados (flags).

## Características

### 1. Argumentos Posicionais
- Devem vir **ANTES** de qualquer flag
- Podem ser números, strings ou symbols
- Acessados por índice (1-based)

### 2. Argumentos Nomeados (Flags)
- Prefixo `@` (estilo Max/MSP) ou `-` (estilo Unix)
- Suportam múltiplos aliases: `@width w @w`
- Tipos de valores:
  - **Boolean**: flag sem valor → `true`
  - **Number**: um único número
  - **String/Symbol**: uma única string
  - **Lista de números**: múltiplos números consecutivos
  - **Lista de strings**: múltiplas strings consecutivas

### 3. Uso Misto
- Posicionais primeiro, depois flags
- Flags têm **prioridade** sobre posicionais

## Exemplos de Uso

```lua
local ArgParser = require("pd_arg_parser")

-- Exemplo 1: Apenas posicionais
local p1 = ArgParser:new({-60, 300, 25})
local dbmin = p1:get_positional_float(1, -120)  -- -60
local width = p1:get_positional_float(2, 200)   -- 300
local height = p1:get_positional_float(3, 20)   -- 25

-- Exemplo 2: Apenas flags
local p2 = ArgParser:new({"@dbmin", -60, "@width", 300, "@height", 25})
local dbmin = p2:get_float("dbmin", -120)  -- -60
local width = p2:get_float("width", 200)   -- 300
local height = p2:get_float("height", 20)  -- 25

-- Exemplo 3: Misto
local p3 = ArgParser:new({-60, 300, "@height", 25})
local dbmin = p3:get_positional_float(1, -120)  -- -60
local width = p3:get_positional_float(2, 200)   -- 300
local height = p3:get_float("height", 20)       -- 25 (flag)

-- Exemplo 4: Flags booleanas
local p4 = ArgParser:new({"@visible", "@active"})
local visible = p4:get_bool("visible", false)  -- true
local active = p4:get_bool("active", false)    -- true

-- Exemplo 5: Listas
local p5 = ArgParser:new({"@values", 1, 2, 3, 4, 5})
local values = p5:get_float_list("values", {})  -- {1, 2, 3, 4, 5}

local p6 = ArgParser:new({"@tags", "alpha", "beta", "gamma"})
local tags = p6:get_string_list("tags", {})  -- {"alpha", "beta", "gamma"}

-- Exemplo 6: Múltiplos aliases
local p7 = ArgParser:new({"-w", 300, "-h", 25})
local width = p7:get_float("width w", 200)   -- 300
local height = p7:get_float("height h", 20)  -- 25
```

## API Reference

### Construtor
```lua
parser = ArgParser:new(atoms)
```

### Métodos para Argumentos Posicionais
```lua
parser:get_positional_count()                    -- retorna número de args posicionais
parser:get_all_positional()                      -- retorna tabela com todos
parser:get_positional(index, default_value)      -- retorna arg por índice
parser:get_positional_float(index, default)      -- retorna arg como número
parser:get_positional_string(index, default)     -- retorna arg como string
```

### Métodos para Flags
```lua
parser:has_flag("flag_names")                    -- verifica se flag existe
parser:get_bool("flag_names", default)           -- retorna boolean
parser:get_float("flag_names", default)          -- retorna número
parser:get_string("flag_names", default)         -- retorna string
parser:get_float_list("flag_names", default)     -- retorna lista de números
parser:get_string_list("flag_names", default)    -- retorna lista de strings
```

**Nota**: `flag_names` pode conter múltiplos aliases separados por espaço.
Exemplo: `"width w @width"` aceita `@width`, `@w`, `-width`, `-w`

## Exemplos no Pure Data

```
[lna.vu -60 300 25]
# Posicionais: dbmin=-60, width=300, height=25

[lna.vu @dbmin -60 @width 300 @height 25]
# Flags: mesmo resultado acima

[lna.vu -60 @width 300 @height 25]
# Misto: dbmin=-60 (posicional), width e height (flags)

[lna.vu @min -60 -w 300 -h 25]
# Aliases: @min, -w, -h

[lna.vu -60 300 @h 25]
# Misto com alias: dbmin e width posicionais, height flag
```

## Notas Importantes

1. **Ordem**: Argumentos posicionais devem vir ANTES das flags
2. **Prioridade**: Flags sobrescrevem valores posicionais
3. **Aliases**: Use espaços para separar aliases em `get_*()` methods
4. **Default values**: Sempre forneça um valor padrão apropriado
5. **Type safety**: Métodos verificam tipos e retornam default em caso de erro

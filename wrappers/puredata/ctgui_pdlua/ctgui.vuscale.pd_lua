-- lvuscale.pd_lua - Escala de dB para VU Meter
-- Cria uma escala de dB alinhada com lvu
--
-- Uso:
--   [lvuscale @w 200 @h 20]
--   [lvuscale @w 20 @h 200 @vert]
--
-- Parâmetros:
--   @w / @width - Largura (padrão: 200)
--   @h / @height - Altura (padrão: 20)
--   @vert / @vertical - Orientação vertical
--   @horiz / @horizontal - Orientação horizontal (padrão)
--   @fontsize / @fs - Tamanho da fonte (auto se não especificado)
--   @colorrgb - Cor do texto RGB (padrão: 200 200 200)
--   @backrgb - Cor de fundo RGB (padrão: 50 50 50)
--   @dbmin - dB mínimo (padrão: -100)
--   @dbmax - dB máximo (padrão: 12)
--   @grid - Intervalo do grid em dB (padrão: 10)
--   @dbvals - Lista customizada de valores de dB (ex: @dbvals 0 -6 -12 -24)

local ArgParser = require("pd_arg_parser")
local Colors = require("colors")

local ctguivuscale = pd.Class:new():register("ctgui.vuscale")

-- Default Colors (HSB 0-1 scale)
local C_TEXT_LIGHT = {0, 0, 0.2}
local C_BG_LIGHT = {0, 0, 0.93}

local C_TEXT_DARK = {0, 0, 0.8}
local C_BG_DARK = {0, 0, 0.2}

function ctguivuscale:initialize(sel, atoms)
    -- Parse argumentos
    local parser = ArgParser:new(atoms)
    
    -- Dimensões
    self.width = parser:get_float("width w", 200)
    self.height = parser:get_float("height h", 20)
    
    -- Orientação
    local default_orientation = "horizontal"
    if self.height > self.width then
        default_orientation = "vertical"
    end
    
    if parser:has_flag("vert vertical") then
        self.orientation = "vertical"
    elseif parser:has_flag("horiz horizontal") then
        self.orientation = "horizontal"
    else
        self.orientation = default_orientation
    end
    
    -- Range de dB
    self.db_min = parser:get_float("dbmin db_min min", -100)
    self.db_max = parser:get_float("dbmax db_max max", 12)
    
    -- Grid step
    self.grid_step = parser:get_float("grid", 10)
    
    -- Valores customizados de dB (opcional)
    self.custom_db_vals = parser:get_float_list("dbvals db_vals")
    if self.custom_db_vals and #self.custom_db_vals > 0 then
        -- Ordena os valores
        table.sort(self.custom_db_vals)
    end
    
    -- Tamanho de fonte (auto se 0)
    self.font_size = parser:get_float("fontsize fs", 0)
    
    -- Parse Colors
    
    -- Helper: RGB to HSB (0-1)
    local function rgb_to_hsb(r, g, b)
        local max = math.max(r, g, b)
        local min = math.min(r, g, b)
        local delta = max - min
        local h, s, v = 0, 0, max

        if max > 0 then s = delta / max else s = 0 end
        
        if delta > 0 then
            if r == max then h = (g - b) / delta
            elseif g == max then h = 2 + (b - r) / delta
            else h = 4 + (r - g) / delta end
            h = h * 60
            if h < 0 then h = h + 360 end
            h = h / 360
        end
        return h, s, v
    end

    -- Helper to get color from flags (RGB or HSB) or default
    local function get_color(name, default_hsb, legacy_aliases)
        -- Check for @namecolor (e.g. @textcolor)
        local val = parser:get_value(name .. "color")
        
        -- If not found, check legacy aliases (e.g. @colorrgb)
        if val == nil and legacy_aliases then
            local leg = parser:get_float_list(legacy_aliases)
            if leg and #leg >= 3 then
                val = {"rgb", leg[1], leg[2], leg[3]}
            end
        end
        
        local h, s, b = default_hsb[1], default_hsb[2], default_hsb[3]
        local source = "default"

        if type(val) == "table" then
            -- Check for format: {"rgb", r, g, b} or {"hsb", h, s, b}
            if type(val[1]) == "string" then
                local mode = val[1]
                if (mode == "rgb" or mode == "RGB") and #val >= 4 then
                    h, s, b = rgb_to_hsb(val[2], val[3], val[4])
                    source = "rgb"
                elseif (mode == "hsb" or mode == "HSB" or mode == "hsl" or mode == "HSL") and #val >= 4 then
                    h, s, b = val[2], val[3], val[4]
                    source = "hsb"
                end
            -- Check for format: {r, g, b} (implicit RGB)
            elseif type(val[1]) == "number" and #val >= 3 then
                h, s, b = rgb_to_hsb(val[1], val[2], val[3])
                source = "rgb"
            end
        end

        -- Convert to RGB (0-255) for drawing
        local r, g, b_val = Colors.hsb(h, s, b)
        return {r, g, b_val, source=source, h=h, s=s, b=b}
    end

    -- Dark Mode Defaults
    local def_text = C_TEXT_LIGHT
    local def_bg = C_BG_LIGHT

    if parser:get_bool("dark") then
        def_text = C_TEXT_DARK
        def_bg = C_BG_DARK
    end

    self.c_text = get_color("text", def_text, "colorrgb color_rgb textrgb text_rgb")
    self.c_background = get_color("bg", def_bg, "backrgb back_rgb bgcolor bg_color bg")
    
    -- Inlets/Outlets
    self.inlets = 1
    self.outlets = 1
    
    -- Armazena argumentos para getcode
    self.creation_args = atoms
    
    return true
end

function ctguivuscale:postinitialize()
    self:set_size(self.width, self.height)
end

-- Função de transferência não-linear (mesma do lvu)
-- Mapeia dB para posição visual (0.0 a 1.0)
-- 0dB fica em 85% da escala visual
function ctguivuscale:db_to_visual(db)
    if db <= self.db_min then
        return 0.0
    end
    
    if db >= self.db_max then
        return 1.0
    end
    
    -- Normaliza dB para 0-1
    local normalized = (db - self.db_min) / (self.db_max - self.db_min)
    
    -- Aplica curva não-linear para comprimir valores negativos
    -- e expandir valores próximos a 0dB
    local zero_db_visual = 0.85  -- 0dB fica em 85% da escala
    
    if db < 0 then
        -- Região negativa: comprimida
        local zero_range = 0 - self.db_min
        local db_from_min = db - self.db_min
        local ratio = db_from_min / zero_range
        -- Aplica curve para comprimir mais os valores baixos
        return zero_db_visual * (ratio * ratio)
    else
        -- Região positiva: expandida
        local max_range = self.db_max - 0
        local db_from_zero = db - 0
        local ratio = db_from_zero / max_range
        return zero_db_visual + ((1.0 - zero_db_visual) * ratio)
    end
end

-- Gera marcações de dB (mesma lógica do lvu)
function ctguivuscale:get_label_marks()
    -- Se tem valores customizados, usa eles
    if self.custom_db_vals and #self.custom_db_vals > 0 then
        return self.custom_db_vals
    end
    
    -- Caso contrário, gera automaticamente baseado no grid
    local marks = {}
    local step = self.grid_step
    
    -- Marca em 0dB sempre
    table.insert(marks, 0)
    
    -- Marcas abaixo de 0dB
    local db = -step
    while db >= self.db_min do
        table.insert(marks, db)
        db = db - step
    end
    
    -- Marcas acima de 0dB
    db = step
    while db <= self.db_max do
        table.insert(marks, db)
        db = db + step
    end
    
    -- Ordena (menor para maior)
    table.sort(marks)
    
    return marks
end

function ctguivuscale:paint(g)
    -- Safety check
    if not self.c_background then
        self.c_background = {0, 0, 0.7}
        if not self.c_text then self.c_text = {0, 0, 0} end
    end

    -- Fundo
    g:set_color(self.c_background[1], self.c_background[2], self.c_background[3])
    g:fill_all()
    
    -- Cor do texto
    g:set_color(self.c_text[1], self.c_text[2], self.c_text[3])
    
    local marks = self:get_label_marks()
    
    -- Calcula tamanho de fonte
    local font_size = self.font_size
    if font_size == 0 then
        -- Auto: baseado na dimensão
        if self.orientation == "vertical" then
            -- Vertical: ajusta para largura E altura para evitar colisão
            -- Usa a menor largura disponível como base
            font_size = math.max(6, math.min(10, math.floor(self.width * 0.6)))
            
            -- Também verifica espaçamento vertical para evitar overlap
            local marks = self:get_label_marks()
            if #marks > 1 then
                local vertical_space = self.height / #marks
                -- Limita fonte para não ultrapassar espaço vertical
                font_size = math.min(font_size, math.floor(vertical_space * 0.8))
            end
        else
            -- Horizontal: ajusta para largura
            if self.width < 200 then
                font_size = math.max(6, math.floor(10 * self.width / 200))
            else
                font_size = 10
            end
        end
    end
    
    if self.orientation == "vertical" then
        -- Vertical: texto centralizado na coctgui
        local text_height = font_size
        
        for _, db in ipairs(marks) do
            local visual_pos = self:db_to_visual(db)
            local y = self.height - (visual_pos * self.height)
            
            -- Linha horizontal pequena (marcador) - opcional, pode remover se poluir
            -- g:fill_rect(0, y, 3, 1)
            
            -- Texto do dB - centralizado verticalmente e horizontalmente
            local label = string.format("%g", db)
            local text_y = y - (text_height / 2)
            
            -- Limita para não sair do topo ou fundo
            if text_y < 0 then
                text_y = 0
            elseif text_y + text_height > self.height then
                text_y = self.height - text_height
            end
            
            -- Centraliza horizontalmente completamente na largura disponível
            local text_width = #label * (font_size / 2)
            local text_x = (self.width - text_width) / 2
            g:draw_text(label, text_x, text_y, self.width, font_size)
        end
    else
        -- Horizontal: texto abaixo ou centralizado
        for _, db in ipairs(marks) do
            local visual_pos = self:db_to_visual(db)
            local x = (visual_pos * self.width)
            
            -- Linha vertical pequena (marcador)
            g:fill_rect(x, 0, 1, 3)
            
            -- Texto do dB (centralizado)
            local label = string.format("%g", db)
            local text_width = #label * (font_size / 2)
            local text_x = x - (text_width / 2)
            
            -- Limita para não sair da esquerda ou direita
            if text_x < 0 then
                text_x = 0
            elseif text_x + (text_width * 1.5) > self.width then
                text_x = self.width - (text_width * 1.5)
            end
            
            -- Centraliza verticalmente na área disponível
            local text_y = (self.height / 2) - (font_size / 2)
            g:draw_text(label, text_x, text_y, text_width * 2, font_size)
        end
    end
    
    -- Borda
    -- g:set_color(180, 180, 180)
    -- g:stroke_rect(0, 0, self.width, self.height, 1)
end

-- Método getcode
function ctguivuscale:in_1_getcode(atoms)
    local cmd = "ctgui.vuscale"
    
    if self.creation_args and #self.creation_args > 0 then
        for _, arg in ipairs(self.creation_args) do
            cmd = cmd .. " " .. tostring(arg)
        end
    else
        cmd = string.format("ctgui.vuscale @w %g @h %g", self.width, self.height)
        if self.orientation == "vertical" then
            cmd = cmd .. " @vert"
        end
        if self.grid_step ~= 10 then
            cmd = cmd .. string.format(" @grid %g", self.grid_step)
        end
    end
    
    self:outlet(1, "list", {cmd})
    pd.post(cmd)
end

-- ctgui.vulabel.pd_lua - Labels para VU Meter multicanal
-- Cria uma sequência de rótulos numéricos para acompanhar o lvu
--
-- Uso:
--   [ctgui.vulabel @channels 8 @h 200]
--   [ctgui.vulabel @ch 4 @labels A B C D @h 150]
--
-- Parâmetros:
--   @channels / @ch - Número de canais (padrão: 1)
--   @label - Lista de labels customizados (opcional)
--   @h / @height - Altura total (obrigatório para vertical)
--   @w / @width - Largura (auto se não especificado)
--   @fontsize / @fs - Tamanho da fonte (padrão: 10)
--   @colorrgb - Cor do texto RGB (padrão: 180 180 180)
--   @backrgb - Cor de fundo RGB (padrão: 50 50 50)
--   @v / @vert / @vertical - Orientação vertical
--   @h / @horiz / @horizontal - Orientação horizontal (padrão)

local ArgParser = require("pd_arg_parser")
local Colors = require("colors")

local ctguivulabel = pd.Class:new():register("ctgui.vulabel")

-- Default Colors (HSB 0-1 scale)
local C_TEXT_LIGHT = {0, 0, 0.2}
local C_BG_LIGHT = {0, 0, 0.93}

local C_TEXT_DARK = {0, 0, 0.8}
local C_BG_DARK = {0, 0, 0.2}

function ctguivulabel:initialize(sel, atoms)
    -- Parse argumentos
    local parser = ArgParser:new(atoms)
    
    -- Número de canais
    self.num_channels = math.max(1, math.floor(parser:get_float("channels ch", 1)))
    
    -- Define inlets e outlets
    self.inlets = 1
    self.outlets = 1
    
    -- Labels customizados
    self.labels = parser:get_string_list("labels", {})
    
    -- Se não tem labels customizados, usa números
    if #self.labels == 0 then
        for i = 1, self.num_channels do
            table.insert(self.labels, tostring(i))
        end
    end
    
    -- Tamanho da fonte
    self.font_size = parser:get_float("fontsize fs", 10)
    
    -- Orientação
    if parser:has_flag("vert vertical") then
        self.orientation = "vertical"
    else
        self.orientation = "horizontal"
    end
    
    -- Calcula largura necessária baseada no label mais longo
    local max_chars = 0
    for _, label in ipairs(self.labels) do
        local label_str = tostring(label)
        if #label_str > max_chars then
            max_chars = #label_str
        end
    end
    
    -- Largura mínima: chars * 5 pixels por char + padding mínimo
    local auto_width = (max_chars * 5) + 4
    
    -- Dimensões
    self.height = parser:get_float("height h", 20)
    -- Se @w for fornecido, usa; senão usa auto_width
    local custom_width = parser:get_float("width w")
    self.width = custom_width or auto_width
    
    -- Armazena argumentos para getcode
    self.creation_args = atoms
    
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
    
    return true
end

function ctguivulabel:postinitialize()
    self:set_size(self.width, self.height)
end

function ctguivulabel:paint(g)
    -- Safety check
    if not self.c_background then
        self.c_background = {0, 0, 0.7}
        if not self.c_text then self.c_text = {0, 0, 0} end
    end

    -- Fundo
    g:set_color(self.c_background[1], self.c_background[2], self.c_background[3])
    g:fill_rect(0, 0, self.width, self.height)
    
    -- Cor do texto
    g:set_color(self.c_text[1], self.c_text[2], self.c_text[3])
    
    if self.orientation == "vertical" then
        -- Vertical: labels lado a lado (como lvu vertical)
        local channel_gap = (self.num_channels > 1) and 1 or 0
        local total_gap = channel_gap * (self.num_channels - 1)
        local channel_width = (self.width - 3 - total_gap) / self.num_channels
        
        for i = 1, self.num_channels do
            local label = self.labels[i] or tostring(i)
            local x_offset = 2 + ((i - 1) * (channel_width + channel_gap))
            
            -- Ajusta tamanho da fonte se necessário
            local fs = self.font_size
            -- Se o canal for muito estreito, reduz a fonte
            if fs > channel_width - 2 then fs = math.max(6, channel_width - 2) end
            
            -- Centraliza verticalmente
            local text_y = (self.height - fs) / 2
            
            -- Centraliza horizontalmente (estimativa)
            local char_w = fs * 0.55
            local est_w = #label * char_w
            local text_x = x_offset + (channel_width - est_w) / 2
            
            g:draw_text(label, text_x, text_y, channel_width, fs)
        end
    else
        -- Horizontal: labels empilhados (como lvu horizontal)
        local channel_gap = (self.num_channels > 1) and 1 or 0
        local total_gap = channel_gap * (self.num_channels - 1)
        local channel_height = (self.height - 2 - total_gap) / self.num_channels
        
        for i = 1, self.num_channels do
            local label = self.labels[i] or tostring(i)
            local y_offset = 2 + ((i - 1) * (channel_height + channel_gap))
            
            -- Ajusta tamanho da fonte se necessário
            local fs = self.font_size
            -- Se o canal for muito baixo, reduz a fonte
            if fs > channel_height - 1 then fs = math.max(6, channel_height - 1) end
            
            -- Centraliza verticalmente
            local text_y = y_offset + (channel_height - fs) / 2
            
            -- Centraliza horizontalmente (estimativa)
            local char_w = fs * 0.55
            local est_w = #label * char_w
            local text_x = (self.width - est_w) / 2
            
            g:draw_text(label, text_x, text_y, self.width, fs)
        end
    end
end

-- Método getcode
function ctguivulabel:in_1_getcode(atoms)
    local cmd = "ctgui.vulabel"
    
    if self.creation_args and #self.creation_args > 0 then
        for _, arg in ipairs(self.creation_args) do
            cmd = cmd .. " " .. tostring(arg)
        end
    else
        cmd = string.format("ctgui.vulabel @channels %d @h %g", self.num_channels, self.height)
        if self.width then
            cmd = cmd .. string.format(" @w %g", self.width)
        end
    end
    
    self:outlet(1, "list", {cmd})
    pd.post(cmd)
end

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

local ctguivulabel = pd.Class:new():register("ctgui.vulabel")

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
    
    -- Cor do texto
    local textrgb = parser:get_float_list("colorrgb color_rgb textrgb text_rgb")
    if textrgb and #textrgb >= 3 then
        self.text_r = math.max(0, math.min(255, textrgb[1]))
        self.text_g = math.max(0, math.min(255, textrgb[2]))
        self.text_b = math.max(0, math.min(255, textrgb[3]))
    else
        self.text_r = 180
        self.text_g = 180
        self.text_b = 180
    end
    
    -- Cor de fundo
    local backrgb = parser:get_float_list("backrgb back_rgb bgcolor bg_color bg")
    if backrgb and #backrgb >= 3 then
        self.back_r = math.max(0, math.min(255, backrgb[1]))
        self.back_g = math.max(0, math.min(255, backrgb[2]))
        self.back_b = math.max(0, math.min(255, backrgb[3]))
    else
        self.back_r = 50
        self.back_g = 50
        self.back_b = 50
    end
    
    return true
end

function ctguivulabel:postinitialize()
    self:set_size(self.width, self.height)
end

function ctguivulabel:paint(g)
    -- Fundo
    g:set_color(self.back_r, self.back_g, self.back_b)
    g:fill_rect(0, 0, self.width, self.height)
    
    -- Cor do texto
    g:set_color(self.text_r, self.text_g, self.text_b)
    
    if self.orientation == "vertical" then
        -- Vertical: labels empilhados (como lvu vertical)
        local channel_gap = (self.num_channels > 1) and 1 or 0
        local total_gap = channel_gap * (self.num_channels - 1)
        local channel_width = (self.width - 4 - total_gap) / self.num_channels
        
        for i = 1, self.num_channels do
            local label = self.labels[i] or tostring(i)
            local x_offset = 2 + ((i - 1) * (channel_width + channel_gap))
            
            -- Centraliza verticalmente
            local text_y = (self.height - self.font_size) / 2
            
            g:draw_text(label, x_offset, text_y, channel_width, self.font_size, 1)  -- centered
        end
    else
        -- Horizontal: labels lado a lado (como lvu horizontal)
        local channel_gap = (self.num_channels > 1) and 1 or 0
        local total_gap = channel_gap * (self.num_channels - 1)
        local channel_height = (self.height - 4 - total_gap) / self.num_channels
        
        for i = 1, self.num_channels do
            local label = self.labels[i] or tostring(i)
            local y_offset = 2 + ((i - 1) * (channel_height + channel_gap))
            
            -- Centraliza verticalmente dentro do canal
            g:draw_text(label, 2, y_offset, self.width - 4, channel_height, 1)  -- centered
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

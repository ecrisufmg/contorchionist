-- ctgui.vu.pd_lua - VU Meter Horizontal/Vertical
-- Cria um VU meter com gradiente de cores baseado em dB
--
-- Uso com argumentos posicionais:
--   [ctgui.vu <dbmin> <width> <height>]
--   Exemplo: [ctgui.vu -60 300 25]
--
-- Uso com argumentos nomeados (flags):
--   [ctgui.vu @dbmin <value> @width <value> @height <value>]
--   [ctgui.vu -dbmin <value> -w <value> -h <value>]
--
-- Orientação:
--   @v / @vert / @vertical - VU vertical (cresce de baixo para cima)
--   @h / @horiz / @horizontal - VU horizontal (padrão)
--   Exemplo: [ctgui.vu @v @width 30 @height 200]
--
-- Uso misto (posicionais ANTES das flags):
--   [ctgui.vu -60 @width 300 @height 25]
--
-- Padrão: dbmin=-120, dbmax=12, width=200, height=20, horizontal
-- Flags têm prioridade sobre argumentos posicionais
--
-- Mensagens suportadas:
--   bang - Reseta o indicador de pico
--   getcode - Envia comando de criação original pela outlet
--   getpositional - Envia comando em formato posicional
--   getflags - Envia comando em formato de flags
--   getinfo - Mostra informações completas no console
--
-- Botão direito: Clique com botão direito para ver comandos no console
-- Outlet: Envia comandos e informações

-- Importa o ArgParser externo
local ArgParser = require("pd_arg_parser")

local lnavu = pd.Class:new():register("ctgui.vu")

function lnavu:initialize(sel, atoms)
    -- Parse argumentos primeiro para saber quantos canais
    local parser = ArgParser:new(atoms)
    
    -- Suporta argumentos posicionais: [vu -120 200 20]
    -- ou argumentos nomeados: [vu @dbmin -120 @width 200 @height 20]
    -- ou misturados: [vu -120 @width 200 @height 20]
    -- Flags têm prioridade sobre argumentos posicionais
    
    -- Parâmetros configuráveis
    -- Tenta flags primeiro, depois posicionais, por fim usa default
    self.db_min = parser:get_float("dbmin db_min min", parser:get_positional_float(1, -120))
    self.db_max = parser:get_float("dbmax db_max max", 12)
    self.width = parser:get_float("width w", parser:get_positional_float(2, 200))
    self.height = parser:get_float("height h", parser:get_positional_float(3, 20))
    
    -- Orientação: vertical ou horizontal
    -- Padrão: baseado nas dimensões (maior dimensão define a orientação)
    local default_orientation = "horizontal"
    if self.height > self.width then
        default_orientation = "vertical"
    end
    
    -- Verifica se alguma flag de orientação foi especificada
    if parser:has_flag("vert vertical") then
        self.orientation = "vertical"
    elseif parser:has_flag("horiz horizontal") then
        self.orientation = "horizontal"
    else
        self.orientation = default_orientation  -- usa o padrão baseado nas dimensões
    end
    
    -- Grid (marcações de dB) - agora aceita valor opcional
    if parser:has_flag("grid") then
        self.grid_step = parser:get_float("grid", 10)  -- padrão 10dB
        self.show_grid = true
    else
        self.grid_step = 10
        self.show_grid = false
    end
    
    -- Modo contínuo (gradiente suave) - ativado com @cont
    if parser:has_flag("cont continuous") then
        self.led_mode = false
        self.num_leds = 14
    else
        -- Modo LED é o padrão
        self.led_mode = true
        if parser:has_flag("led") then
            self.num_leds = parser:get_float("led", 14)  -- padrão 14 LEDs
        else
            self.num_leds = 14
        end
    end
    
    -- Legenda (escala de dB)
    self.show_label = parser:has_flag("scale")
    
    -- Altura da área de escala (quando @scale está ativo)
    self.scale_height = parser:get_float("scaleh scale_height", 0)  -- 0 = automático
    
    -- Número de canais (para VUs sobrepostos)
    self.num_channels = math.max(1, math.floor(parser:get_float("channels ch", 1)))
    
    -- Define inlets e outlets (um por canal)
    self.inlets = self.num_channels
    self.outlets = self.num_channels
    
    -- Tags de canais (opcionais)
    self.channel_tags = parser:get_string_list("chtags", {})
    
    -- Mostrar labels de canal (apenas com @chlabels)
    self.show_channel_labels = parser:has_flag("chlabels chlabel")
    
    -- FPS limit (throttle de atualização visual)
    self.fps = math.max(1, parser:get_float("fps", 20))  -- padrão 20 FPS
    self.frame_counter = 0
    self.frame_skip = math.max(1, math.floor(60 / self.fps))  -- quantos frames pular
    self.needs_repaint = false
    
    -- Cor de fundo (padrão cinza 50 50 50)
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
    
    -- Cor da escala (texto de dB)
    local scalergb = parser:get_float_list("scalecolorrgb scale_color_rgb scalecolor")
    if scalergb and #scalergb >= 3 then
        self.scale_r = math.max(0, math.min(255, scalergb[1]))
        self.scale_g = math.max(0, math.min(255, scalergb[2]))
        self.scale_b = math.max(0, math.min(255, scalergb[3]))
    else
        self.scale_r = 200
        self.scale_g = 200
        self.scale_b = 200
    end
    
    -- Estado atual
    self.current_db = self.db_min
    self.peak_db = self.db_min
    self.peak_hold_time = 0
    
    -- Estados por canal (para multi-canal)
    self.channel_db = {}
    self.channel_peak_db = {}
    self.channel_peak_hold = {}
    for i = 1, self.num_channels do
        self.channel_db[i] = self.db_min
        self.channel_peak_db[i] = self.db_min
        self.channel_peak_hold[i] = 0
    end
    
    -- Armazena os argumentos originais para o menu de contexto
    self.creation_args = atoms
    
    -- Ajusta dbmax para alinhamento correto com a escala não-linear
    -- No modo LED: um LED termina exatamente em 0dB
    -- No modo contínuo: também precisa do ajuste para mapear corretamente
    if self.led_mode then
        self:adjust_dbmax_for_zero_alignment()
    end
    
    return true
end

function lnavu:postinitialize()
    -- Define o tamanho visual do objeto
    -- Se tem label, adiciona espaço extra
    local display_width = self.width - 2
    local display_height = self.height - 2
    
    if self.show_label then
        if self.orientation == "vertical" then
            -- Vertical: label à direita (proporcional à altura, mínimo 30px para 3 dígitos)
            local label_width
            if self.scale_height > 0 then
                label_width = self.scale_height
            else
                label_width = math.max(30, math.min(45, self.height / 10))
            end
            display_width = display_width + label_width
            self.label_area = label_width
        else
            -- Horizontal: label abaixo (proporcional à largura, mínimo 18px)
            local label_height
            if self.scale_height > 0 then
                label_height = self.scale_height
            else
                label_height = math.max(18, math.min(30, self.width / 25))
            end
            display_height = display_height + label_height
            self.label_area = label_height
        end
    end
    
    self:set_size(display_width, display_height)
end

-- Recebe valores de dB
-- Método genérico para processar entrada de canal
function lnavu:process_channel_input(inlet, f)
    -- Atualiza canal correspondente ao inlet (1-indexed)
    self.channel_db[inlet] = f
    
    -- Atualiza pico do canal
    if f > self.channel_peak_db[inlet] then
        self.channel_peak_db[inlet] = f
        self.channel_peak_hold[inlet] = 30
    end
    
    -- Decrementa hold
    if self.channel_peak_hold[inlet] > 0 then
        self.channel_peak_hold[inlet] = self.channel_peak_hold[inlet] - 1
    else
        self.channel_peak_db[inlet] = math.max(f, self.channel_peak_db[inlet] - 0.5)
    end
    
    -- Atualiza current_db com o máximo (para compatibilidade)
    self.current_db = self.db_min
    for i = 1, self.num_channels do
        if self.channel_db[i] > self.current_db then
            self.current_db = self.channel_db[i]
        end
    end
    
    -- Passthrough: envia valor recebido pela outlet correspondente
    self:outlet(inlet, "float", {f})
    
    self:throttled_repaint()
end

-- Métodos individuais para cada inlet (pd-lua requer isso)
function lnavu:in_1_float(f)
    self:process_channel_input(1, f)
end

function lnavu:in_2_float(f)
    self:process_channel_input(2, f)
end

function lnavu:in_3_float(f)
    self:process_channel_input(3, f)
end

function lnavu:in_4_float(f)
    self:process_channel_input(4, f)
end

function lnavu:in_5_float(f)
    self:process_channel_input(5, f)
end

function lnavu:in_6_float(f)
    self:process_channel_input(6, f)
end

function lnavu:in_7_float(f)
    self:process_channel_input(7, f)
end

function lnavu:in_8_float(f)
    self:process_channel_input(8, f)
end

function lnavu:in_9_float(f)
    self:process_channel_input(9, f)
end

function lnavu:in_10_float(f)
    self:process_channel_input(10, f)
end

function lnavu:in_11_float(f)
    self:process_channel_input(11, f)
end

function lnavu:in_12_float(f)
    self:process_channel_input(12, f)
end

function lnavu:in_13_float(f)
    self:process_channel_input(13, f)
end

function lnavu:in_14_float(f)
    self:process_channel_input(14, f)
end

function lnavu:in_15_float(f)
    self:process_channel_input(15, f)
end

function lnavu:in_16_float(f)
    self:process_channel_input(16, f)
end

-- Método de repaint com throttle (controle de FPS)
function lnavu:throttled_repaint()
    self.frame_counter = self.frame_counter + 1
    
    if self.frame_counter >= self.frame_skip then
        self.frame_counter = 0
        self:repaint()
    end
end

-- Função auxiliar para calcular cor baseada em dB
function lnavu:get_color_for_db(db)
    local red, green, blue
    
    -- Acima de 0dB: vermelho para magenta (sinal estourou!)
    if db >= 0 then
        -- Gradiente de vermelho para magenta conforme aumenta acima de 0dB
        local t = math.min(1, db / (self.db_max - 0))  -- normaliza de 0dB a dbmax
        red = 255
        green = 0
        blue = math.floor(255 * t)  -- adiciona azul para fazer magenta
    -- Entre -3dB e 0dB: laranja para vermelho
    elseif db > -3 then
        local t = (db - (-3)) / (0 - (-3))  -- normaliza de -3 a 0
        red = 255
        green = math.floor(128 * (1 - t))  -- de 128 para 0
        blue = 0
    else
        -- Normaliza dB para 0-1 (apenas na faixa de db_min a -3)
        -- O gradiente vai de db_min até 0dB (não até db_max!)
        local normalized = (db - self.db_min) / (0 - self.db_min)
        normalized = math.max(0, math.min(1, normalized))
        
        -- Gradiente de cores de db_min até -3dB:
        -- Ajusta os pontos de transição para cobrir toda a faixa db_min -> 0dB
        local transition_point = (-3 - self.db_min) / (0 - self.db_min)
        
        if normalized < 0.5 * transition_point then
            -- Cyan para Verde
            local t = normalized / (0.5 * transition_point)
            red = 0
            green = 255
            blue = math.floor(255 * (1 - t))
        elseif normalized < 0.85 * transition_point then
            -- Verde para Amarelo
            local t = (normalized - 0.5 * transition_point) / (0.35 * transition_point)
            red = math.floor(255 * t)
            green = 255
            blue = 0
        else
            -- Amarelo para Laranja (até -3dB)
            local t = (normalized - 0.85 * transition_point) / (0.15 * transition_point)
            red = 255
            green = math.floor(255 - (127 * t))
            blue = 0
        end
    end
    
    return red, green, blue
end

-- Gera lista de valores de dB para a legenda
function lnavu:get_label_marks()
    -- Pontos de referência importantes (ordem de prioridade)
    local base_marks = {-180, -120, -48, -24, -12, -6, -3, 0, 3, 6, 12, 24, 48, 90, 120, 180}
    
    -- Filtra apenas os que estão dentro do range
    local marks = {}
    local zero_included = false
    
    for _, db in ipairs(base_marks) do
        if db >= self.db_min and db <= self.db_max then
            table.insert(marks, db)
            if db == 0 then
                zero_included = true
            end
        end
    end
    
    -- Garante que 0dB sempre aparece se estiver no range
    if not zero_included and 0 >= self.db_min and 0 <= self.db_max then
        table.insert(marks, 0)
        table.sort(marks)
    end
    
    -- Se tem muitas marcas próximas, remove as menos importantes
    -- (mantém múltiplos de 6, 10 e sempre 0)
    if #marks > 12 then
        local filtered = {}
        for _, db in ipairs(marks) do
            if db == 0 or db % 6 == 0 or db % 10 == 0 then
                table.insert(filtered, db)
            end
        end
        marks = filtered
    end
    
    -- Ajusta primeira marca se estiver muito próxima de dbmin
    if #marks > 0 and marks[1] - self.db_min < 6 then
        -- Remove a primeira marca e adiciona uma 20dB acima de dbmin
        table.remove(marks, 1)
        local adjusted_min = self.db_min + 20
        -- Arredonda para múltiplo de 10
        adjusted_min = math.floor(adjusted_min / 10) * 10
        table.insert(marks, 1, adjusted_min)
    end
    
    -- Remove última marca se estiver muito próxima de dbmax
    if #marks > 0 and self.db_max - marks[#marks] < 6 then
        table.remove(marks, #marks)
    end
    
    return marks
end

-- Função de transferência não-linear para escala visual do VU
-- Converte dB real para posição visual normalizada (0-1)
-- Dá mais espaço visual para a região útil (-20dB a 0dB)
-- A curva é definida em relação a 0dB, não a dbmax
function lnavu:db_to_visual(db)
    -- Clamp db entre dbmin e dbmax
    db = math.max(self.db_min, math.min(self.db_max, db))
    
    -- Divide em duas regiões: abaixo de 0dB e acima de 0dB
    if db <= 0 then
        -- Região de dbmin até 0dB usa a curva não-linear
        -- Normaliza dB para 0-1 nesta faixa
        local linear = (db - self.db_min) / (0 - self.db_min)
        linear = math.max(0, math.min(1, linear))
        
        -- Pontos de referência (normalizados em relação a dbmin -> 0dB):
        -- -120dB -> 0.0   => visual ~0.0
        -- -60dB  -> 0.5   => visual ~0.2  (comprimido)
        -- -20dB  -> 0.833 => visual ~0.5  (região útil começa)
        -- 0dB    -> 1.0   => visual ~0.85
        
        local zero_visual = 0.85  -- 0dB ocupa 85% do espaço visual total
        
        local visual
        if linear < 0.5 then
            -- Região de dbmin a -60dB: muito comprimida (usa apenas 20% do espaço até 0dB)
            visual = (0.2 * zero_visual) * (linear / 0.5)
        elseif linear < 0.833 then
            -- Região de -60dB a -20dB: levemente comprimida (usa 30% do espaço)
            local t = (linear - 0.5) / (0.833 - 0.5)
            visual = (0.2 * zero_visual) + (0.3 * zero_visual * t)
        else
            -- Região de -20dB a 0dB: expandida (usa 50% do espaço - região útil!)
            local t = (linear - 0.833) / (1.0 - 0.833)
            visual = (0.5 * zero_visual) + (0.5 * zero_visual * t)
        end
        
        return visual
    else
        -- Região acima de 0dB: linear de 0.85 a 1.0
        local t = db / (self.db_max - 0)
        return 0.85 + (0.15 * t)
    end
end

-- Função inversa: converte posição visual (0-1) para dB
-- Inverte a função db_to_visual
function lnavu:visual_to_db(visual)
    visual = math.max(0, math.min(1, visual))
    
    local zero_visual = 0.85  -- 0dB está em 85% da posição visual
    
    if visual <= zero_visual then
        -- Região de dbmin até 0dB
        -- Inverte a curva não-linear
        local linear
        if visual < 0.2 * zero_visual then
            -- Região muito comprimida: dbmin a -60dB
            linear = 0.5 * (visual / (0.2 * zero_visual))
        elseif visual < 0.5 * zero_visual then
            -- Região levemente comprimida: -60dB a -20dB
            local t = (visual - 0.2 * zero_visual) / (0.3 * zero_visual)
            linear = 0.5 + (0.333 * t)  -- 0.833 - 0.5 = 0.333
        else
            -- Região expandida: -20dB a 0dB
            local t = (visual - 0.5 * zero_visual) / (0.5 * zero_visual)
            linear = 0.833 + (0.167 * t)  -- 1.0 - 0.833 = 0.167
        end
        
        -- Converte linear normalizado para dB (na faixa dbmin a 0dB)
        local db = self.db_min + (linear * (0 - self.db_min))
        return db
    else
        -- Região acima de 0dB (linear)
        local t = (visual - 0.85) / 0.15
        local db = 0 + (t * (self.db_max - 0))
        return db
    end
end

-- Ajusta dbmax para que um LED termine exatamente em 0dB
function lnavu:adjust_dbmax_for_zero_alignment()
    -- Calcula a posição visual de 0dB com o dbmax atual
    local zero_visual = self:db_to_visual(0)
    
    -- Cada LED ocupa 1/num_leds do espaço visual
    local led_visual_size = 1.0 / self.num_leds
    
    -- Encontra qual LED deve terminar em 0dB
    -- Queremos que zero_visual coincida com o fim de um LED
    local led_index_at_zero = math.floor(zero_visual / led_visual_size)
    
    -- O LED que contém 0dB vai de led_index_at_zero a (led_index_at_zero + 1)
    -- Queremos que 0dB seja exatamente o fim deste LED
    local target_visual = (led_index_at_zero + 1) * led_visual_size
    
    -- Agora precisamos encontrar qual dB corresponde a esta posição visual
    -- Invertendo db_to_visual para encontrar o dB que produz target_visual
    -- Como db_to_visual é não-linear, fazemos busca binária ou aproximação
    
    -- Método iterativo simples: ajusta dbmax até zero_visual coincidir com fim de LED
    local new_dbmax = self.db_max
    local tolerance = 0.01  -- tolerância para convergência
    local max_iterations = 50
    
    for i = 1, max_iterations do
        self.db_max = new_dbmax
        local current_zero_visual = self:db_to_visual(0)
        local current_led_end = (led_index_at_zero + 1) * led_visual_size
        
        local error = current_zero_visual - current_led_end
        
        if math.abs(error) < tolerance * led_visual_size then
            break  -- Convergiu!
        end
        
        -- Ajusta dbmax proporcionalmente ao erro
        -- Se 0dB está muito alto (error > 0), aumenta dbmax
        -- Se 0dB está muito baixo (error < 0), diminui dbmax
        new_dbmax = new_dbmax + error * (self.db_max - self.db_min) / 10
    end
    
    self.db_max = new_dbmax
end

-- Desenha o VU meter
function lnavu:paint(g)
    -- Fundo com cor configurável
    g:set_color(self.back_r, self.back_g, self.back_b)
    g:fill_all()
    
    -- Usa função de transferência não-linear para posição visual
    if self.num_channels > 1 then
        -- Multi-canal: desenha todos os canais sobrepostos
        if self.led_mode then
            self:paint_led_mode_multichannel(g)
        else
            self:paint_continuous_mode_multichannel(g)
        end
    else
        -- Single-canal: comportamento normal
        local visual_position = self:db_to_visual(self.current_db)
        if self.led_mode then
            -- Modo LED: barras discretas
            self:paint_led_mode(g, visual_position)
        else
            -- Modo contínuo (gradiente suave)
            self:paint_continuous_mode(g, visual_position)
        end
    end
    
    -- Borda
    g:set_color(180, 180, 180)
    g:stroke_rect(0, 0, self.width, self.height, 1)
    
    -- Desenha legenda se ativada
    if self.show_label then
        self:paint_labels(g)
    end
end

-- Desenha modo contínuo (original)
function lnavu:paint_continuous_mode(g, visual_position)
    if self.orientation == "vertical" then
        -- VU Vertical (cresce de baixo para cima)
        local bar_height = visual_position * (self.height - 4)
        
        -- Desenha gradiente de cores
        if bar_height > 0 then
            local segments = math.min(50, math.ceil(bar_height))
            local segment_height = bar_height / segments
            
            for i = 0, segments - 1 do
                local y = (self.height - 2) - ((i + 1) * segment_height)  -- De baixo para cima
                local h = segment_height
                
                -- Calcula a posição visual normalizada deste segmento
                local seg_visual_position = (i / segments) * visual_position
                
                -- Converte posição visual para dB real para obter a cor correta
                local seg_db = self:visual_to_db(seg_visual_position)
                
                -- Obtém cor para este dB
                local cr, cg, cb = self:get_color_for_db(seg_db)
                g:set_color(cr, cg, cb)
                g:fill_rect(2, y, self.width - 4, h)
            end
        end
        
        -- Desenha linha de pico
        if self.peak_db > self.db_min then
            local peak_visual = self:db_to_visual(self.peak_db)
            local peak_y = (self.height - 2) - (peak_visual * (self.height - 4))
            
            local pr, pg, pb = self:get_color_for_db(self.peak_db)
            g:set_color(pr, pg, pb)
            g:fill_rect(2, peak_y - 1, self.width - 4, 2)
        end
        
        -- Desenha linha de referência em 0dB (sempre visível)
        if 0 >= self.db_min and 0 <= self.db_max then
            local zero_visual = self:db_to_visual(0)
            local zero_y = (self.height - 2) - (zero_visual * (self.height - 4))
            g:set_color(200, 200, 200)
            g:fill_rect(0, zero_y, self.width, 1)
        end
        
        -- Desenha marcações de dB
        if self.show_grid then
            g:set_color(100, 100, 100)
            for db = self.db_min, self.db_max, self.grid_step do
                if db ~= self.db_min and db ~= self.db_max and db ~= 0 then
                    local mark_visual = self:db_to_visual(db)
                    local mark_y = (self.height - 2) - (mark_visual * (self.height - 4))
                    g:fill_rect(0, mark_y, self.width, 1)
                end
            end
        end
    else
        -- VU Horizontal (original)
        local bar_width = visual_position * (self.width - 4)
        
        -- Desenha gradiente de cores
        if bar_width > 0 then
            local segments = math.min(50, math.ceil(bar_width))
            local segment_width = bar_width / segments
            
            for i = 0, segments - 1 do
                local x = 2 + (i * segment_width)
                local w = segment_width
                
                -- Calcula a posição normalizada deste segmento (0 a visual_position)
                local seg_position = (i / segments) * visual_position
                
                -- Calcula dB real para este segmento usando a função inversa
                local seg_db = self:visual_to_db(seg_position)
                
                -- Obtém cor para este dB
                local cr, cg, cb = self:get_color_for_db(seg_db)
                g:set_color(cr, cg, cb)
                g:fill_rect(x, 2, w, self.height - 4)
            end
        end
        
        -- Desenha linha de pico
        if self.peak_db > self.db_min then
            local peak_visual = self:db_to_visual(self.peak_db)
            local peak_x = 2 + (peak_visual * (self.width - 4))
            
            local pr, pg, pb = self:get_color_for_db(self.peak_db)
            g:set_color(pr, pg, pb)
            g:fill_rect(peak_x - 1, 2, 2, self.height - 4)
        end
        
        -- Desenha linha de referência em 0dB (sempre visível)
        if 0 >= self.db_min and 0 <= self.db_max then
            local zero_visual = self:db_to_visual(0)
            local zero_x = 2 + (zero_visual * (self.width - 4))
            g:set_color(200, 200, 200)
            g:fill_rect(zero_x, 0, 1, self.height)
        end
        
        -- Desenha marcações de dB (opcional, a cada 10dB)
        if self.show_grid then
            g:set_color(100, 100, 100)
            for db = self.db_min, self.db_max, self.grid_step do
                if db ~= self.db_min and db ~= self.db_max and db ~= 0 then
                    local mark_visual = self:db_to_visual(db)
                    local mark_x = 2 + (mark_visual * (self.width - 4))
                    g:fill_rect(mark_x, 0, 1, self.height)
                end
            end
        end
    end
end

-- Desenha modo contínuo multi-canal (sobrepostos com transparência)
function lnavu:paint_continuous_mode_multichannel(g)
    -- Espaço para tags/números de canal
    local tag_space = 0
    if self.num_channels > 1 and self.show_channel_labels then
        tag_space = (self.orientation == "vertical") and 15 or 20
    end
    
    -- Espaço disponível para os canais
    if self.orientation == "vertical" then
        -- Vertical: canais lado a lado
        local channel_gap = (self.num_channels > 1) and 1 or 0  -- 1px de gap entre canais
        local total_gap = channel_gap * (self.num_channels - 1)
        local channel_width = (self.width - 4 - total_gap) / self.num_channels
        
        for ch = 1, self.num_channels do
            local visual_position = self:db_to_visual(self.channel_db[ch])
            local x_offset = 2 + ((ch - 1) * (channel_width + channel_gap))
            local bar_height = visual_position * (self.height - 4 - tag_space)
            
            -- Desenha gradiente
            if bar_height > 0 then
                local segments = math.min(50, math.ceil(bar_height))
                local segment_height = bar_height / segments
                
                for i = 0, segments - 1 do
                    local y = (self.height - 2 - tag_space) - ((i + 1) * segment_height)
                    local h = segment_height
                    local seg_visual_position = (i / segments) * visual_position
                    local seg_db = self:visual_to_db(seg_visual_position)
                    
                    local cr, cg, cb = self:get_color_for_db(seg_db)
                    g:set_color(cr, cg, cb)
                    g:fill_rect(x_offset, y, channel_width, h)
                end
            end
            
            -- Linha de pico
            if self.channel_peak_db[ch] > self.db_min then
                local peak_visual = self:db_to_visual(self.channel_peak_db[ch])
                local peak_y = (self.height - 2 - tag_space) - (peak_visual * (self.height - 4 - tag_space))
                local pr, pg, pb = self:get_color_for_db(self.channel_peak_db[ch])
                g:set_color(pr, pg, pb)
                g:fill_rect(x_offset, peak_y - 1, channel_width, 2)
            end
            
            -- Desenha tag/número do canal (abaixo)
            if self.num_channels > 1 and self.show_channel_labels then
                local tag = self.channel_tags[ch] or tostring(ch)
                g:set_color(180, 180, 180)
                g:draw_text(tag, x_offset, self.height - tag_space + 3, channel_width, tag_space - 3, 1)
            end
        end
        
        -- Desenha linhas separadoras entre canais
        if self.num_channels > 1 then
            g:set_color(40, 40, 40)
            for ch = 1, self.num_channels - 1 do
                local line_x = 2 + (ch * (channel_width + channel_gap)) - channel_gap
                g:fill_rect(line_x, 0, channel_gap, self.height)
            end
        end
    else
        -- Horizontal: canais um acima do outro
        local channel_gap = (self.num_channels > 1) and 1 or 0
        local total_gap = channel_gap * (self.num_channels - 1)
        local channel_height = (self.height - 4 - total_gap) / self.num_channels
        
        for ch = 1, self.num_channels do
            local visual_position = self:db_to_visual(self.channel_db[ch])
            local y_offset = 2 + ((ch - 1) * (channel_height + channel_gap))
            local bar_width = visual_position * (self.width - 4 - tag_space)
            
            -- Desenha gradiente
            if bar_width > 0 then
                local segments = math.min(50, math.ceil(bar_width))
                local segment_width = bar_width / segments
                
                for i = 0, segments - 1 do
                    local x = 2 + tag_space + (i * segment_width)
                    local w = segment_width
                    local seg_position = (i / segments) * visual_position
                    local seg_db = self:visual_to_db(seg_position)
                    
                    local cr, cg, cb = self:get_color_for_db(seg_db)
                    g:set_color(cr, cg, cb)
                    g:fill_rect(x, y_offset, w, channel_height)
                end
            end
            
            -- Linha de pico
            if self.channel_peak_db[ch] > self.db_min then
                local peak_visual = self:db_to_visual(self.channel_peak_db[ch])
                local peak_x = 2 + tag_space + (peak_visual * (self.width - 4 - tag_space))
                local pr, pg, pb = self:get_color_for_db(self.channel_peak_db[ch])
                g:set_color(pr, pg, pb)
                g:fill_rect(peak_x - 1, y_offset, 2, channel_height)
            end
            
            -- Desenha tag/número do canal (à esquerda)
            if self.num_channels > 1 and self.show_channel_labels then
                local tag = self.channel_tags[ch] or tostring(ch)
                g:set_color(180, 180, 180)
                g:draw_text(tag, 2, y_offset, tag_space - 2, channel_height, 1)
            end
        end
        
        -- Desenha linhas separadoras entre canais
        if self.num_channels > 1 then
            g:set_color(40, 40, 40)
            for ch = 1, self.num_channels - 1 do
                local line_y = 2 + (ch * (channel_height + channel_gap)) - channel_gap
                g:fill_rect(0, line_y, self.width, channel_gap)
            end
        end
    end
    
    -- Desenha linha de referência 0dB (apenas uma vez)
    g:set_color(200, 200, 200)
    if self.orientation == "vertical" then
        if 0 >= self.db_min and 0 <= self.db_max then
            local zero_visual = self:db_to_visual(0)
            local zero_y = (self.height - 2 - tag_space) - (zero_visual * (self.height - 4 - tag_space))
            g:fill_rect(0, zero_y, self.width, 1)
        end
    else
        if 0 >= self.db_min and 0 <= self.db_max then
            local zero_visual = self:db_to_visual(0)
            local zero_x = 2 + tag_space + (zero_visual * (self.width - 4 - tag_space))
            g:fill_rect(zero_x, 0, 1, self.height)
        end
    end
end

-- Desenha modo LED (barras discretas)
function lnavu:paint_led_mode(g, visual_position)
    -- LEDs dividem o espaço visual igualmente (não por dB)
    local led_visual_size = 1.0 / self.num_leds
    
    -- Calcula o gap baseado no tamanho disponível
    local gap = 2  -- gap padrão em pixels
    local corner_radius = 3  -- Raio dos cantos arredondados padrão
    local use_rounded = true
    local led_visual_size = 1.0 / self.num_leds
    
    -- Calcula o gap baseado no tamanho disponível
    local gap = 2  -- gap padrão em pixels
    local corner_radius = 3  -- Raio dos cantos arredondados padrão
    local use_rounded = true
    
    if self.orientation == "vertical" then
        -- Verifica se os LEDs ficariam muito pequenos
        local led_pixel_height = (self.height - 4) / self.num_leds - gap
        if led_pixel_height < 3 then
            gap = 0
            use_rounded = false
        end
    else
        local led_pixel_width = (self.width - 4) / self.num_leds - gap
        if led_pixel_width < 3 then
            gap = 0
            use_rounded = false
        end
    end
    
    if self.orientation == "vertical" then
        -- LED Vertical - cada LED ocupa 1/num_leds do espaço visual
        for i = 0, self.num_leds - 1 do
            local led_visual_min = i * led_visual_size
            local led_visual_max = (i + 1) * led_visual_size
            
            -- Verifica se este LED deve estar aceso
            if visual_position >= led_visual_min then
                local y = (self.height - 2) - (led_visual_max * (self.height - 4))
                local led_pixel_height = led_visual_size * (self.height - 4)
                
                -- Aplica gap
                if gap > 0 and led_pixel_height > gap then
                    y = y + gap / 2
                    led_pixel_height = led_pixel_height - gap
                end
                
                -- Calcula o dB que este LED representa (baseado em sua posição visual)
                -- Cada LED tem cor fixa baseada em sua própria faixa de dB
                local led_visual_center = (led_visual_min + led_visual_max) / 2
                -- Usa a função inversa para converter posição visual para dB corretamente
                local led_db = self:visual_to_db(led_visual_center)
                
                local cr, cg, cb = self:get_color_for_db(led_db)
                g:set_color(cr, cg, cb)
                if use_rounded and led_pixel_height > corner_radius * 2 then
                    g:fill_rounded_rect(2, y, self.width - 4, led_pixel_height, corner_radius)
                else
                    g:fill_rect(2, y, self.width - 4, led_pixel_height)
                end
            end
        end
        
        -- Desenha linha de pico (se estiver em um LED diferente do atual)
        if self.peak_db > self.current_db and self.peak_db > self.db_min then
            local peak_visual = self:db_to_visual(self.peak_db)
            local peak_led_index = math.floor(peak_visual / led_visual_size)
            
            local led_visual_min = peak_led_index * led_visual_size
            local led_visual_max = (peak_led_index + 1) * led_visual_size
            
            local peak_y = (self.height - 2) - (led_visual_max * (self.height - 4))
            local led_pixel_height = led_visual_size * (self.height - 4)
            
            -- Aplica gap
            if gap > 0 and led_pixel_height > gap then
                peak_y = peak_y + gap / 2
                led_pixel_height = led_pixel_height - gap
            end
            
            local pr, pg, pb = self:get_color_for_db(self.peak_db)
            g:set_color(pr, pg, pb)
            if use_rounded and led_pixel_height > corner_radius * 2 then
                g:fill_rounded_rect(2, peak_y, self.width - 4, led_pixel_height, corner_radius)
            else
                g:fill_rect(2, peak_y, self.width - 4, led_pixel_height)
            end
        end
        
        -- Desenha linha de referência em 0dB (sempre visível)
        if 0 >= self.db_min and 0 <= self.db_max then
            local zero_visual = self:db_to_visual(0)
            local zero_y = (self.height - 2) - (zero_visual * (self.height - 4))
            g:set_color(200, 200, 200)
            g:fill_rect(0, zero_y, self.width, 1)
        end
    else
        -- LED Horizontal - cada LED ocupa 1/num_leds do espaço visual
        for i = 0, self.num_leds - 1 do
            local led_visual_min = i * led_visual_size
            local led_visual_max = (i + 1) * led_visual_size
            
            -- Verifica se este LED deve estar aceso
            if visual_position >= led_visual_min then
                local x = 2 + (led_visual_min * (self.width - 4))
                local led_pixel_width = led_visual_size * (self.width - 4)
                
                -- Aplica gap
                if gap > 0 and led_pixel_width > gap then
                    x = x + gap / 2
                    led_pixel_width = led_pixel_width - gap
                end
                
                -- Calcula o dB que este LED representa (baseado em sua posição visual)
                -- Cada LED tem cor fixa baseada em sua própria faixa de dB
                local led_visual_center = (led_visual_min + led_visual_max) / 2
                -- Usa a função inversa para converter posição visual para dB corretamente
                local led_db = self:visual_to_db(led_visual_center)
                
                local cr, cg, cb = self:get_color_for_db(led_db)
                g:set_color(cr, cg, cb)
                if use_rounded and led_pixel_width > corner_radius * 2 then
                    g:fill_rounded_rect(x, 2, led_pixel_width, self.height - 4, corner_radius)
                else
                    g:fill_rect(x, 2, led_pixel_width, self.height - 4)
                end
            end
        end
        
        -- Desenha linha de pico (se estiver em um LED diferente do atual)
        if self.peak_db > self.current_db and self.peak_db > self.db_min then
            local peak_visual = self:db_to_visual(self.peak_db)
            local peak_led_index = math.floor(peak_visual / led_visual_size)
            
            local led_visual_min = peak_led_index * led_visual_size
            local led_visual_max = (peak_led_index + 1) * led_visual_size
            
            local peak_x = 2 + (led_visual_min * (self.width - 4))
            local led_pixel_width = led_visual_size * (self.width - 4)
            
            -- Aplica gap
            if gap > 0 and led_pixel_width > gap then
                peak_x = peak_x + gap / 2
                led_pixel_width = led_pixel_width - gap
            end
            
            local pr, pg, pb = self:get_color_for_db(self.peak_db)
            g:set_color(pr, pg, pb)
            if use_rounded and led_pixel_width > corner_radius * 2 then
                g:fill_rounded_rect(peak_x, 2, led_pixel_width, self.height - 4, corner_radius)
            else
                g:fill_rect(peak_x, 2, led_pixel_width, self.height - 4)
            end
        end
        
        -- Desenha linha de referência em 0dB (sempre visível)
        if 0 >= self.db_min and 0 <= self.db_max then
            local zero_visual = self:db_to_visual(0)
            local zero_x = 2 + (zero_visual * (self.width - 4))
            g:set_color(200, 200, 200)
            g:fill_rect(zero_x, 0, 1, self.height)
        end
    end
end

-- Desenha a legenda (labels de dB)
function lnavu:paint_labels(g)
    local marks = self:get_label_marks()
    
    -- Usa cores da escala ou padrão se não inicializado
    local scale_r = self.scale_r or 200
    local scale_g = self.scale_g or 200
    local scale_b = self.scale_b or 200
    g:set_color(scale_r, scale_g, scale_b)  -- Cor do texto e marcadores
    
    -- Calcula tamanho de fonte baseado na largura (para horizontal) ou altura (para vertical)
    local font_size = 10  -- padrão
    if self.orientation == "vertical" then
        -- Vertical: não precisa ajustar
    else
        -- Horizontal: reduz fonte para larguras menores que 200
        if self.width < 200 then
            font_size = math.max(6, math.floor(10 * self.width / 200))
        end
    end
    
    if self.orientation == "vertical" then
        -- Vertical: legenda à direita
        local label_area_width = self.label_area or 30
        local label_x = self.width
        local text_height = font_size - 2  -- Altura aproximada do texto
        
        for _, db in ipairs(marks) do
            local visual_pos = self:db_to_visual(db)
            local y = (self.height - 2) - (visual_pos * (self.height - 4))
            
            -- Linha horizontal pequena (marcador)
            g:fill_rect(self.width, y, 3, 1)
            
            -- Texto do dB - centralizado na área de scale
            local label = string.format("%g", db)
            local text_y = y - (text_height / 2)
            
            -- Limita para não sair do topo ou fundo
            if text_y < 0 then
                text_y = 0
            elseif text_y + text_height > self.height + label_area_width then
                text_y = self.height + label_area_width - text_height
            end
            
            -- Centraliza horizontalmente na área de scale
            local text_x = label_x + (label_area_width / 2) - 10  -- -10 para centralizar aprox.
            g:draw_text(label, text_x, text_y, 20, font_size)
        end
    else
        -- Horizontal: legenda abaixo
        local label_area_height = self.label_area or 18
        local label_y = self.height
        
        for _, db in ipairs(marks) do
            local visual_pos = self:db_to_visual(db)
            local x = 2 + (visual_pos * (self.width - 4))
            
            -- Linha vertical pequena (marcador)
            g:fill_rect(x, self.height, 1, 3)
            
            -- Texto do dB (centralizado) - ajusta posição para não sair dos limites
            local label = string.format("%g", db)
            local text_width = #label * (font_size / 2)  -- Aproximação da largura
            local text_x = x - (text_width / 2)
            
            -- Limita para não sair da esquerda ou direita
            if text_x < 0 then
                text_x = 0
            elseif text_x + (text_width * 1.5) > self.width + 28 then
                text_x = self.width + 28 - (text_width * 1.5)
            end
            
            -- Centraliza verticalmente na área de scale
            local text_y = label_y + (label_area_height / 2) - (font_size / 2)
            g:draw_text(label, text_x, text_y, text_width * 2, font_size)
        end
    end
end

-- Desenha modo LED multi-canal (sobrepostos)
function lnavu:paint_led_mode_multichannel(g)
    local led_visual_size = 1.0 / self.num_leds
    
    -- Espaço para tags/números de canal
    local tag_space = 0
    if self.num_channels > 1 and self.show_channel_labels then
        tag_space = (self.orientation == "vertical") and 15 or 20
    end
    
    -- Calcula gap
    local gap = 2
    local corner_radius = 3
    local use_rounded = true
    
    if self.orientation == "vertical" then
        -- Vertical: canais lado a lado
        local channel_gap = (self.num_channels > 1) and 1 or 0  -- 1px de gap entre canais
        local total_gap = channel_gap * (self.num_channels - 1)
        local channel_width = (self.width - 4 - total_gap) / self.num_channels
        local led_pixel_height = (self.height - 4 - tag_space) / self.num_leds - gap
        
        if led_pixel_height < 3 then
            gap = 0
            use_rounded = false
        end
        
        for ch = 1, self.num_channels do
            local visual_position = self:db_to_visual(self.channel_db[ch])
            local x_offset = 2 + ((ch - 1) * (channel_width + channel_gap))
            local led_padding = (self.num_channels > 1) and 1 or 0  -- padding lateral nos LEDs
            
            for i = 0, self.num_leds - 1 do
                local led_visual_min = i * led_visual_size
                local led_visual_max = (i + 1) * led_visual_size
                
                if visual_position >= led_visual_min then
                    local y = (self.height - 2 - tag_space) - (led_visual_max * (self.height - 4 - tag_space))
                    local led_pixel_height = led_visual_size * (self.height - 4 - tag_space)
                    
                    if gap > 0 and led_pixel_height > gap then
                        y = y + gap / 2
                        led_pixel_height = led_pixel_height - gap
                    end
                    
                    local led_visual_center = (led_visual_min + led_visual_max) / 2
                    local led_db = self:visual_to_db(led_visual_center)
                    local cr, cg, cb = self:get_color_for_db(led_db)
                    g:set_color(cr, cg, cb)
                    
                    if use_rounded and led_pixel_height > corner_radius * 2 then
                        g:fill_rounded_rect(x_offset + led_padding, y, channel_width - (led_padding * 2), led_pixel_height, corner_radius)
                    else
                        g:fill_rect(x_offset + led_padding, y, channel_width - (led_padding * 2), led_pixel_height)
                    end
                end
            end
            
            -- Desenha tag/número do canal (abaixo)
            if self.num_channels > 1 and self.show_channel_labels then
                local tag = self.channel_tags[ch] or tostring(ch)
                g:set_color(180, 180, 180)
                g:draw_text(tag, x_offset, self.height - tag_space + 3, channel_width, tag_space - 3, 1)
            end
        end
        
        -- Desenha linhas separadoras entre canais
        if self.num_channels > 1 then
            g:set_color(40, 40, 40)  -- linha escura
            for ch = 1, self.num_channels - 1 do
                local line_x = 2 + (ch * (channel_width + channel_gap)) - channel_gap
                g:fill_rect(line_x, 0, channel_gap, self.height)
            end
        end
    else
        -- Horizontal: canais um acima do outro
        local channel_gap = (self.num_channels > 1) and 1 or 0  -- 1px de gap entre canais
        local total_gap = channel_gap * (self.num_channels - 1)
        local channel_height = (self.height - 4 - total_gap) / self.num_channels
        local led_pixel_width = (self.width - 4 - tag_space) / self.num_leds - gap
        
        if led_pixel_width < 3 then
            gap = 0
            use_rounded = false
        end
        
        for ch = 1, self.num_channels do
            local visual_position = self:db_to_visual(self.channel_db[ch])
            local y_offset = 2 + ((ch - 1) * (channel_height + channel_gap))
            local led_padding = (self.num_channels > 1) and 1 or 0  -- padding vertical nos LEDs
            
            for i = 0, self.num_leds - 1 do
                local led_visual_min = i * led_visual_size
                local led_visual_max = (i + 1) * led_visual_size
                
                if visual_position >= led_visual_min then
                    local x = 2 + tag_space + (led_visual_min * (self.width - 4 - tag_space))
                    local led_pixel_width = led_visual_size * (self.width - 4 - tag_space)
                    
                    if gap > 0 and led_pixel_width > gap then
                        x = x + gap / 2
                        led_pixel_width = led_pixel_width - gap
                    end
                    
                    local led_visual_center = (led_visual_min + led_visual_max) / 2
                    local led_db = self:visual_to_db(led_visual_center)
                    local cr, cg, cb = self:get_color_for_db(led_db)
                    g:set_color(cr, cg, cb)
                    
                    if use_rounded and led_pixel_width > corner_radius * 2 then
                        g:fill_rounded_rect(x, y_offset + led_padding, led_pixel_width, channel_height - (led_padding * 2), corner_radius)
                    else
                        g:fill_rect(x, y_offset + led_padding, led_pixel_width, channel_height - (led_padding * 2))
                    end
                end
            end
            
            -- Desenha tag/número do canal (à esquerda)
            if self.num_channels > 1 and self.show_channel_labels then
                local tag = self.channel_tags[ch] or tostring(ch)
                g:set_color(180, 180, 180)
                g:draw_text(tag, 2, y_offset, tag_space - 2, channel_height, 1)
            end
        end
        
        -- Desenha linhas separadoras entre canais
        if self.num_channels > 1 then
            g:set_color(40, 40, 40)
            for ch = 1, self.num_channels - 1 do
                local line_y = 2 + (ch * (channel_height + channel_gap)) - channel_gap
                g:fill_rect(0, line_y, self.width, channel_gap)
            end
        end
    end
    
    -- Linha de referência 0dB (apenas uma vez)
    g:set_color(200, 200, 200)
    if self.orientation == "vertical" then
        if 0 >= self.db_min and 0 <= self.db_max then
            local zero_visual = self:db_to_visual(0)
            local zero_y = (self.height - 2 - tag_space) - (zero_visual * (self.height - 4 - tag_space))
            g:fill_rect(0, zero_y, self.width, 1)
        end
    else
        if 0 >= self.db_min and 0 <= self.db_max then
            local zero_visual = self:db_to_visual(0)
            local zero_x = 2 + tag_space + (zero_visual * (self.width - 4 - tag_space))
            g:fill_rect(zero_x, 0, 1, self.height)
        end
    end
end

-- Método para resetar pico
function lnavu:in_1_bang()
    self.peak_db = self.db_min
    self.peak_hold_time = 0
    
    -- Reseta todos os canais
    for i = 1, self.num_channels do
        self.channel_peak_db[i] = self.db_min
        self.channel_peak_hold[i] = 0
    end
    
    self:repaint()
end

-- Método para alterar FPS em tempo real
function lnavu:in_1_fps(atoms)
    local f = type(atoms) == "table" and atoms[1] or atoms
    if type(f) == "number" then
        self.fps = math.max(1, math.floor(f))
        self.frame_skip = math.max(1, math.floor(60 / self.fps))
        self.frame_counter = 0
        pd.post(string.format("ctgui.vu: FPS set to %d (frame_skip=%d)", self.fps, self.frame_skip))
    else
        pd.post("ctgui.vu: fps expects a number")
    end
end

-- Gera o comando de instanciação para copiar
function lnavu:get_creation_command()
    local cmd = "ctgui.vu"
    
    -- Se não tem argumentos, retorna apenas o nome
    if not self.creation_args or #self.creation_args == 0 then
        -- Gera com valores atuais se diferentes dos defaults
        if self.db_min ~= -120 or self.width ~= 200 or self.height ~= 20 then
            cmd = string.format("ctgui.vu @dbmin %g @width %g @height %g", 
                                self.db_min, self.width, self.height)
        end
        return cmd
    end
    
    -- Reconstrói o comando com os argumentos originais
    for i, arg in ipairs(self.creation_args) do
        if type(arg) == "number" then
            cmd = cmd .. " " .. tostring(arg)
        elseif type(arg) == "string" then
            cmd = cmd .. " " .. arg
        end
    end
    
    return cmd
end

function lnavu:get_positional_command()
    return string.format("ctgui.vu %g %g %g", self.db_min, self.width, self.height)
end

function lnavu:get_flags_command()
    local cmd = string.format("ctgui.vu @dbmin %g @width %g @height %g", 
                               self.db_min, self.width, self.height)
    if self.db_max ~= 12 then
        cmd = cmd .. string.format(" @dbmax %g", self.db_max)
    end
    if self.orientation == "vertical" then
        cmd = cmd .. " @v"
    end
    if self.show_grid then
        if self.grid_step ~= 10 then
            cmd = cmd .. string.format(" @grid %g", self.grid_step)
        else
            cmd = cmd .. " @grid"
        end
    end
    if self.show_label then
        cmd = cmd .. " @scale"
    end
    if self.num_channels > 1 then
        cmd = cmd .. string.format(" @channels %g", self.num_channels)
    end
    if not self.led_mode then
        -- Modo contínuo precisa ser especificado (LED é default)
        cmd = cmd .. " @cont"
    elseif self.num_leds ~= 14 then
        cmd = cmd .. string.format(" @led %g", self.num_leds)
    end
    if self.back_r ~= 50 or self.back_g ~= 50 or self.back_b ~= 50 then
        cmd = cmd .. string.format(" @backrgb %g %g %g", self.back_r, self.back_g, self.back_b)
    end
    return cmd
end

-- Mensagens para obter comandos
function lnavu:in_1_getcode(atoms)
    local cmd = self:get_creation_command()
    local cmd = self:get_creation_command()
    pd.post(cmd)
    self:outlet(1, "symbol", {cmd})
end

function lnavu:in_1_getpositional(atoms)
    local cmd = self:get_positional_command()
    pd.post(cmd)
    self:outlet(1, "symbol", {cmd})
end

function lnavu:in_1_getflags(atoms)
    local cmd = self:get_flags_command()
    pd.post(cmd)
    self:outlet(1, "symbol", {cmd})
end

function lnavu:in_1_getinfo(atoms)
    pd.post("========== ctgui.vu Object Info ==========")
    pd.post(string.format("dbmin: %g dB", self.db_min))
    pd.post(string.format("dbmax: %g dB", self.db_max))
    pd.post(string.format("width: %g px", self.width))
    pd.post(string.format("height: %g px", self.height))
    pd.post(string.format("orientation: %s", self.orientation))
    pd.post(string.format("grid: %s (step: %g dB)", self.show_grid and "yes" or "no", self.grid_step))
    pd.post(string.format("label: %s", self.show_label and "yes" or "no"))
    pd.post(string.format("channels: %g", self.num_channels))
    pd.post(string.format("mode: %s%s", self.led_mode and "LED" or "continuous", self.led_mode and string.format(" (num_leds: %g)", self.num_leds) or ""))
    pd.post(string.format("background: RGB(%g, %g, %g)", self.back_r, self.back_g, self.back_b))
    pd.post(string.format("current_db: %g dB", self.current_db))
    pd.post(string.format("peak_db: %g dB", self.peak_db))
    pd.post("Creation command: " .. self:get_creation_command())
    pd.post("========================================")
end

-- Clique com botão direito mostra info e comandos
function lnavu:mouse_down(x, y, button, mod)
    if button == 2 then  -- Botão direito (right-click)
        pd.post("")
        pd.post("========== ctgui.vu - Right Click Menu ==========")
        pd.post("Original:   " .. self:get_creation_command())
        pd.post("Positional: " .. self:get_positional_command())
        pd.post("Flags:      " .. self:get_flags_command())
        pd.post("================================================")
        pd.post("Messages: [getcode(, [getpositional(, [getflags(, [getinfo(")
        pd.post("")
        
        -- Envia também pela outlet
        self:outlet(1, "list", {"command", self:get_creation_command()})
        return true
    end
    return false
end

-- Menu de contexto (botão direito)
function lnavu:mouse_down_old(x, y, button, mod)
    if button == 2 then  -- Botão direito
        -- Cria menu popup
        self:popup_menu({
            {label = "Copy Creation Command", action = "copy_command"},
            {label = "Copy as Positional", action = "copy_positional"},
            {label = "Copy as Flags", action = "copy_flags"},
            {separator = true},
            {label = "Reset Peak", action = "reset_peak"},
        })
        return true
    end
    return false
end

-- Processa as ações do menu
function lnavu:menu_action(action)
    if action == "copy_command" then
        -- Copia o comando original
        local cmd = self:get_creation_command()
        pd.post("Copied to clipboard: " .. cmd)
        -- pd.clipboard_set(cmd)  -- Se disponível no pd-lua
        self:outlet(1, "clipboard", {cmd})
        
    elseif action == "copy_positional" then
        -- Copia no formato posicional
        local cmd = string.format("ctgui.vu %g %g %g", self.db_min, self.width, self.height)
        pd.post("Copied to clipboard: " .. cmd)
        self:outlet(1, "clipboard", {cmd})
        
    elseif action == "copy_flags" then
        -- Copia no formato com flags
        local cmd = string.format("ctgui.vu @dbmin %g @width %g @height %g", 
                                   self.db_min, self.width, self.height)
        pd.post("Copied to clipboard: " .. cmd)
        self:outlet(1, "clipboard", {cmd})
        
    elseif action == "reset_peak" then
        -- Reseta o pico
        self.peak_db = self.db_min
        self.peak_hold_time = 0
        self:repaint()
    end
end

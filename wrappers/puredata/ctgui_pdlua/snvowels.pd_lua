local snvowels = pd.Class:new():register("snvowels")

-- ========================================================
-- CONFIGURAÇÃO DO ESPAÇO VOCÁLICO (2D)
-- ========================================================

-- Definimos cada vogal como um "Ponto de Gravidade" no espaço X,Y
-- X: 0.0 (Frente) até 1.0 (Fundo)
-- Y: 0.0 (Fechada) até 1.0 (Aberta)

local VOWEL_MAP = {
    -- === FRONT VOWELS (x ~ 0.0) ===
    -- Close Front (i)
    [1] = {id="i", x=0.0, y=0.0, f1=270, f2=2290, f3=3010},
    -- Near-close Near-front (I)
    [2] = {id="I", x=0.15, y=0.2, f1=390, f2=1990, f3=2550},
    -- Close-mid Front (e)
    [3] = {id="e", x=0.05, y=0.35, f1=450, f2=2100, f3=2800},
    -- Open-mid Front (E / epsilon)
    [4] = {id="E", x=0.1, y=0.6, f1=530, f2=1840, f3=2480},
    -- Near-open Front (ae / ash)
    [5] = {id="ae", x=0.15, y=0.85, f1=660, f2=1720, f3=2410},

    -- === CENTRAL VOWELS (x ~ 0.5) ===
    -- Close Central (ix / barred-i)
    [6] = {id="ix", x=0.5, y=0.1, f1=290, f2=1500, f3=2300},
    -- Mid Central (schwa)
    [7] = {id="@", x=0.5, y=0.5, f1=500, f2=1500, f3=2500},
    -- Open Central (a) - Anchor
    [8] = {id="a", x=0.5, y=1.0, f1=730, f2=1090, f3=2440},

    -- === BACK VOWELS (x ~ 1.0) ===
    -- Close Back (u)
    [9] = {id="u", x=1.0, y=0.0, f1=300, f2=870, f3=2240},
    -- Near-close Near-back (U / upsilon)
    [10] = {id="U", x=0.85, y=0.2, f1=440, f2=1020, f3=2240},
    -- Close-mid Back (o)
    [11] = {id="o", x=0.95, y=0.4, f1=460, f2=750, f3=2300},
    -- Open-mid Back (O / open-o)
    [12] = {id="O", x=0.9, y=0.6, f1=570, f2=840, f3=2410},
    -- Open Back (alpha)
    [13] = {id="alpha", x=0.8, y=0.9, f1=700, f2=900, f3=2300}
}

-- Larguras de banda fixas (BW) para manter a inteligibilidade
local BW_CONSTANTS = { bw1=50, bw2=80, bw3=120 }

-- ========================================================
-- FUNÇÕES MATEMÁTICAS AUXILIARES
-- ========================================================

-- Calcula distância Euclidiana entre dois pontos (x1,y1) e (x2,y2)
local function get_distance(x1, y1, x2, y2)
    return math.sqrt((x2 - x1)^2 + (y2 - y1)^2)
end

-- Algoritmo IDW (Inverse Distance Weighting)
-- Calcula os formantes baseados na proximidade do cursor (px, py)
local function calculate_formants_at(px, py)
    local num_f1, num_f2, num_f3 = 0, 0, 0
    local den = 0
    local power = 2 -- Fator de decaimento (2 = quadrático, transição suave)

    for _, v in ipairs(VOWEL_MAP) do
        local d = get_distance(px, py, v.x, v.y)
        
        -- Previne divisão por zero se cair exatamente em cima da vogal
        if d < 0.001 then d = 0.001 end
        
        local weight = 1.0 / (d ^ power)
        
        num_f1 = num_f1 + (v.f1 * weight)
        num_f2 = num_f2 + (v.f2 * weight)
        num_f3 = num_f3 + (v.f3 * weight)
        
        den = den + weight
    end

    return (num_f1 / den), (num_f2 / den), (num_f3 / den)
end

-- Gera um ponto aleatório que tende a cair DENTRO do triângulo V
-- Isso evita coordenadas (0.5, 0.0) que seriam "vazias" (sem vogal definida)
local function get_random_point_in_triangle()
    -- Vamos usar o método mais simples de mistura de 2 pontas
    -- Sorteia 3 pesos para os vertices do triangulo (I, U, A)
    local w1 = math.random()
    local w2 = math.random()
    local w3 = math.random()
    local total = w1 + w2 + w3
    
    -- Posição ponderada média dos 3 vértices extremos
    -- [1]=i (0,0), [9]=u (1,0), [8]=a (0.5,1)
    local final_x = (0.0 * w1 + 1.0 * w2 + 0.5 * w3) / total
    local final_y = (0.0 * w1 + 0.0 * w2 + 1.0 * w3) / total
    
    return final_x, final_y
end

-- Encontra a vogal mais próxima do ponto (px, py)
local function get_nearest_vowel(px, py)
    local min_dist = math.huge
    local nearest_id = "?"
    
    for _, v in ipairs(VOWEL_MAP) do
        local d = get_distance(px, py, v.x, v.y)
        if d < min_dist then
            min_dist = d
            nearest_id = v.id
        end
    end
    return nearest_id
end

-- ========================================================
-- CLASSE PD
-- ========================================================

function snvowels:initialize(sel, atoms)
    math.randomseed(os.time())
    self.inlets = 1
    self.outlets = 1
    return true
end

function snvowels:in_1_list(atoms)
    -- Espera formato: <midi_float> <freq> <db> <flag> <min> <sec>
    if #atoms < 4 then return end
    
    local id = atoms[1] -- midi_float (ou identificador)
    -- local freq = atoms[2]
    -- local db = atoms[3]
    local flag = atoms[4]
    
    -- Processa apenas Note On (flag 1)
    if flag == 1 then
        -- 1. Definir Onde estamos no espaço 2D
        local px, py = get_random_point_in_triangle()
        
        -- 2. Calcular os formantes resultantes dessa posição espacial
        local res_f1, res_f2, res_f3 = calculate_formants_at(px, py)
        
        -- 3. Encontrar a vogal mais próxima (descritor IPA simplificado)
        local vowel_char = get_nearest_vowel(px, py)

        -- 4. Retornar lista
        -- Output: <vowel_char> <f1> <bw1> <f2> <bw2> <f3> <bw3> <x> <y>
        local output_list = {
            vowel_char,
            res_f1, BW_CONSTANTS.bw1,
            res_f2, BW_CONSTANTS.bw2,
            res_f3, BW_CONSTANTS.bw3,
            px,  -- X para visualização (0=Frente, 1=Fundo)
            py   -- Y para visualização (0=Fechada, 1=Aberta)
        }

        self:outlet(1, "list", output_list)
    end
end

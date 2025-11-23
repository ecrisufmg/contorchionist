local Colors = {}

-- Retorna valores RGB (0-255) a partir de entrada 0-1
function Colors.rgb(r, g, b)
    return math.max(0, math.min(255, math.floor((r or 0) * 255))), 
           math.max(0, math.min(255, math.floor((g or 0) * 255))), 
           math.max(0, math.min(255, math.floor((b or 0) * 255)))
end

-- Converte HSB (Hue 0-1, Saturation 0-1, Brightness 0-1) para RGB (0-255)
function Colors.hsb(h, s, b)
    -- Map h (0-1) to 0-360
    h = (h or 0) * 360
    s = s or 0
    b = b or 0
    
    local c = b * s
    local x = c * (1 - math.abs((h / 60) % 2 - 1))
    local m = b - c
    
    local r, g, b_val
    
    if h < 60 then
        r, g, b_val = c, x, 0
    elseif h < 120 then
        r, g, b_val = x, c, 0
    elseif h < 180 then
        r, g, b_val = 0, c, x
    elseif h < 240 then
        r, g, b_val = 0, x, c
    elseif h < 300 then
        r, g, b_val = x, 0, c
    else
        r, g, b_val = c, 0, x
    end
    
    return Colors.rgb(r + m, g + m, b_val + m)
end

return Colors

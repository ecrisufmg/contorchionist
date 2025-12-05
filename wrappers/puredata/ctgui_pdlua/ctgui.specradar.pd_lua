local Colors = require("colors")
local ArgParser = require("pd_arg_parser")

local mic_vis = pd.Class:new():register("ctgui.specradar")

-- Default Colors (HSB 0-1 scale)
local C_TEXT_LIGHT = {0, 0, 0.4}
local C_BG_LIGHT = {0, 0, 0.84}
local C_GRID_LIGHT = {0, 0, 0.7}

local C_TEXT_DARK = {0, 0, 0.8}
local C_BG_DARK = {0, 0, 0.3}
local C_GRID_DARK = {0, 0, 0.5}

function mic_vis:initialize(sel, atoms)
    self.inlets = 1
    self.outlets = 1
    self.creation_args = atoms
    
    local parser = ArgParser:new(atoms)
    
    -- State
    self.detections = {} -- List of {band_idx, angle, strength, level, age}
    
    -- Config
    self.decay_rate = 0.90
    self.max_age = 20 -- Frames
    self.gain = parser:get_float("gain", 10.0) or 10.0 -- Radius gain
    self.fps = parser:get_float("fps guifps", 20) or 20
    if self.fps <= 0 then self.fps = 20 end
    
    self.num_bands = parser:get_float("num", 4) or 4
    if self.num_bands < 1 then self.num_bands = 1 end
    
    -- Graphics
    local w = parser:get_float("w width", 180) or 180
    local h = parser:get_float("h height", 180) or 180
    local size = parser:get_float("size")
    if size then w, h = size, size end
    
    self:set_size(w, h)
    
    -- Color Parsing Helpers
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

    local function get_color(name, default_hsb, legacy_aliases)
        local val = parser:get_value(name .. "color")
        if val == nil and legacy_aliases then
            local leg = parser:get_float_list(legacy_aliases)
            if leg and #leg >= 3 then val = {"rgb", leg[1], leg[2], leg[3]} end
        end
        local h, s, b = default_hsb[1], default_hsb[2], default_hsb[3]
        local source = "default"
        if type(val) == "table" then
            if type(val[1]) == "string" then
                local mode = val[1]
                if (mode == "rgb" or mode == "RGB") and #val >= 4 then
                    h, s, b = rgb_to_hsb(val[2], val[3], val[4])
                    source = "rgb"
                elseif (mode == "hsb" or mode == "HSB") and #val >= 4 then
                    h, s, b = val[2], val[3], val[4]
                    source = "hsb"
                end
            elseif type(val[1]) == "number" and #val >= 3 then
                h, s, b = rgb_to_hsb(val[1], val[2], val[3])
                source = "rgb"
            end
        end
        local r, g, b_val = Colors.hsb(h, s, b)
        return {r, g, b_val, source=source, h=h, s=s, b=b}
    end

    -- Dark Mode
    local def_text = C_TEXT_LIGHT
    local def_bg = C_BG_LIGHT
    local def_grid = C_GRID_LIGHT

    self.dark_mode = parser:get_bool("dark")
    if self.dark_mode then
        def_text = C_TEXT_DARK
        def_bg = C_BG_DARK
        def_grid = C_GRID_DARK
    end

    self.c_text = get_color("text", def_text, "colorrgb color_rgb textrgb text_rgb")
    self.c_background = get_color("bg", def_bg, "backrgb back_rgb bgcolor bg_color bg")
    self.c_grid = get_color("grid", def_grid, "gridrgb grid_rgb")
    
    -- Clock for animation/decay
    self.clock = pd.Clock:new():register(self, "tick")
    self.clock:delay(1000 / self.fps)
    
    return true
end

function mic_vis:in_1_list(atoms)
    -- Format: <band_index> <angle_deg> <strength> <overall_level> <dominant_freq_hz>
    if #atoms < 5 then return end
    
    local band_idx = atoms[1]
    local angle = atoms[2]
    local strength = atoms[3]
    local level = atoms[4] -- Power/Level
    
    -- Add to detections
    table.insert(self.detections, {
        band_idx = band_idx,
        angle = angle,
        strength = strength,
        level = level,
        age = 0
    })
end

function mic_vis:tick()
    -- Decay detections
    local i = 1
    while i <= #self.detections do
        self.detections[i].age = self.detections[i].age + 1
        if self.detections[i].age > self.max_age then
            table.remove(self.detections, i)
        else
            i = i + 1
        end
    end
    
    self:repaint()
    self.clock:delay(1000 / self.fps)
end

function mic_vis:paint(g)
    local w, h = self:get_size()
    local cx, cy = w/2, h/2
    local radius = math.min(w, h) / 2 - 10
    
    -- Background
    g:set_color(self.c_background[1], self.c_background[2], self.c_background[3])
    g:fill_all()
    
    -- Grid Colors
    local gr, gg, gb = self.c_grid[1], self.c_grid[2], self.c_grid[3]
    
    -- Grid (Polar)
    g:set_color(gr, gg, gb)
    g:stroke_ellipse(cx - radius, cy - radius, radius*2, radius*2, 1)
    g:stroke_ellipse(cx - radius*0.5, cy - radius*0.5, radius, radius, 1)
    
    -- Radial Lines
    for deg = 0, 359, 22.5 do
        local rad = math.rad(deg)
        -- Invert X for Left=90, Right=-90
        local x2 = cx - radius * math.sin(rad)
        local y2 = cy - radius * math.cos(rad)
        
        local intensity = 1.0
        if deg % 90 == 0 then
             intensity = 1.0 -- Main axes
        elseif deg % 45 == 0 then
             intensity = 0.6 -- 45 deg
        else
             intensity = 0.3 -- 22.5 deg
        end
        
        -- Blend grid color with background color for transparency effect
        -- c = grid * intensity + bg * (1 - intensity)
        local r_line = gr * intensity + self.c_background[1] * (1 - intensity)
        local g_line = gg * intensity + self.c_background[2] * (1 - intensity)
        local b_line = gb * intensity + self.c_background[3] * (1 - intensity)
        
        g:set_color(r_line, g_line, b_line)
        g:draw_line(cx, cy, x2, y2, 1)
    end
    
    -- Draw Angle Labels
    g:set_color(self.c_text[1], self.c_text[2], self.c_text[3])
    local fs = 9
    -- Position text inside the circle
    -- 0 (Top)
    g:draw_text("0", cx - 3, cy - radius + 2, 30, fs)
    -- 90 (Left)
    g:draw_text("90", cx - radius + 4, cy - 5, 30, fs)
    -- 180 (Bottom)
    g:draw_text("180", cx - 8, cy + radius - 12, 30, fs)
    -- -90 (Right)
    g:draw_text("-90", cx + radius - 20, cy - 5, 30, fs)
    
    -- Draw Detections
    for _, d in ipairs(self.detections) do
        local angle_rad = math.rad(d.angle)
        
        -- Position (Angle)
        local r = d.strength * radius * self.gain
        r = math.min(r, radius) -- Clamp to border
        
        -- Invert X here too
        local x = cx - r * math.sin(angle_rad)
        local y = cy - r * math.cos(angle_rad)
        
        -- Color (Hue) based on Band Index and Num Bands
        -- Divide hue continuum in equal parts
        local hue = (d.band_idx / self.num_bands) % 1.0
        
        -- Alpha (Transparency) based on Strength
        -- Clamp strength to 0-1 to avoid invalid alpha
        local s_clamped = math.max(0, math.min(1, d.strength))
        local alpha = 0.3 + 0.7 * s_clamped
        
        -- Size based on Power (Level)
        -- Handle dB (negative values) or large linear values
        local norm_level = d.level
        if norm_level < 0 then
             -- Assume dB: Map -100 to 0, 0 to 1
             norm_level = math.max(0, (norm_level + 100) / 100)
        end
        -- Clamp to reasonable range (0-10) to prevent explosion
        norm_level = math.max(0, math.min(10, norm_level))
        
        local size = 5 + norm_level * 50 
        -- Clamp size to widget dimensions to prevent "Red Screen of Death"
        size = math.min(size, math.min(w, h))

        -- Get RGB
        local r_col, g_col, b_col = Colors.hsb(hue, 1, 1)
        
        -- Draw
        -- If alpha supported: g:set_color(r, g, b, alpha)
        -- We'll try passing 4 args.
        g:set_color(r_col, g_col, b_col, alpha)
        g:fill_ellipse(x - size/2, y - size/2, size, size)
    end
end

function mic_vis:in_1_decay(atoms)
    if type(atoms[1]) == "number" then
        self.decay_rate = atoms[1]
    end
end

function mic_vis:in_1_gain(atoms)
    if type(atoms[1]) == "number" then
        self.gain = atoms[1]
    end
end

function mic_vis:in_1_clear()
    self.detections = {}
    self:repaint()
end

function mic_vis:in_1_mode(atoms)
    if type(atoms[1]) == "number" then
        self.mode = atoms[1]
        self:repaint()
    end
end

function mic_vis:in_1_num(atoms)
    if type(atoms[1]) == "number" then
        self.num_bands = atoms[1]
        if self.num_bands < 1 then self.num_bands = 1 end
    end
end

function mic_vis:in_1_getcode(atoms)
    local w, h = self:get_size()
    local cmd = "ctgui.specradar"
    
    if w == h then
        cmd = cmd .. string.format(" @size %g", w)
    else
        cmd = cmd .. string.format(" @w %g @h %g", w, h)
    end
    
    cmd = cmd .. string.format(" @gain %g @fps %g @num %g", self.gain, self.fps, self.num_bands)
    
    if self.dark_mode then
        cmd = cmd .. " @dark"
    end
    
    self:outlet(1, "list", {cmd})
    pd.post(cmd)
end

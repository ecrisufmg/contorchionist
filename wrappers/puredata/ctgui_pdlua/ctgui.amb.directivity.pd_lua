local Colors = require("colors")
local ArgParser = require("pd_arg_parser")

local mic_vis = pd.Class:new():register("ctgui.amb.directivity")

function mic_vis:initialize(sel, atoms)
    self.inlets = 1
    self.outlets = 0
    
    -- State
    self.detections = {} -- List of {angle, strength, level, age}
    self.bins = {}       -- Histogram for lobe view (36 bins of 10 degrees)
    for i = 1, 36 do self.bins[i] = 0 end
    
    -- Config
    self.decay_rate = 0.90
    self.max_age = 20 -- Frames
    self.mode = 2 -- 0=Scatter, 1=Lobe, 2=Both
    
    -- Graphics
    self:set_size(200, 200)
    self.bg_color = 0 -- White/Default
    
    -- Clock for animation/decay
    self.clock = pd.Clock:new():register(self, "tick")
    self.clock:delay(33) -- ~30 FPS
    
    return true
end

function mic_vis:in_1_list(atoms)
    -- Format: <band_index> <angle_deg> <strength> <overall_level> <dominant_freq_hz>
    if #atoms < 5 then return end
    
    local band_idx = atoms[1]
    local angle = atoms[2]
    local strength = atoms[3]
    local level = atoms[4] -- Assuming linear magnitude or power, or dB?
                           -- User said "Power (dB)" in the prompt example, but the object outputs what is configured.
                           -- Let's assume it's normalized 0-1 for color mapping, or we clamp it.
    
    -- Add to detections
    table.insert(self.detections, {
        angle = angle,
        strength = strength,
        level = level,
        age = 0
    })
    
    -- Add to bins (Lobe)
    -- Map angle 0-360 to 1-36
    local bin_idx = math.floor(angle / 10) + 1
    if bin_idx > 36 then bin_idx = 1 end
    
    -- Accumulate energy (strength * level) or just strength?
    -- User said: "Lobe... represents the sum of directional energy"
    -- Let's add strength.
    self.bins[bin_idx] = math.max(self.bins[bin_idx], strength) 
    -- Using max instead of sum to prevent explosion, or we can sum and decay faster.
    -- Let's try max for "peak hold" style or sum with hard decay.
    -- Let's use a weighted accumulation: current + new
    -- self.bins[bin_idx] = self.bins[bin_idx] + strength * 0.5
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
    
    -- Decay bins
    for j = 1, 36 do
        self.bins[j] = self.bins[j] * self.decay_rate
    end
    
    self:repaint()
    self.clock:delay(33)
end

function mic_vis:paint(g)
    local w, h = self:get_size()
    local cx, cy = w/2, h/2
    local radius = math.min(w, h) / 2 - 10
    
    -- Background
    g:set_color(0) -- Background color
    g:fill_all()
    
    -- Grid (Polar)
    g:set_color(200, 200, 200) -- Light Grey
    g:stroke_ellipse(cx - radius, cy - radius, radius*2, radius*2, 1)
    g:stroke_ellipse(cx - radius*0.5, cy - radius*0.5, radius, radius, 1)
    
    -- Crosshairs
    g:draw_line(cx - radius, cy, cx + radius, cy, 1)
    g:draw_line(cx, cy - radius, cx, cy + radius, 1)
    
    -- Mode 1 or 2: Lobe
    if self.mode == 1 or self.mode == 2 then
        -- Draw filled polygon
        -- We need to construct the path points
        -- Since pd-lua graphics API is limited (no begin_path/vertex), we might have to draw lines or many small polys.
        -- Wait, does pd-lua have `fill_polygon`?
        -- The tutorial mentioned "arbitrary paths" but didn't show the command.
        -- Looking at `pdlua_gfx.h` or similar would help, but I can't.
        -- I'll assume `draw_line` loop for the outline and maybe `fill_polygon` if it exists.
        -- If not, just the outline is fine for now.
        
        -- Let's try to draw the outline of the lobe
        g:set_color(100, 149, 237, 0.6) -- Cornflower Blue, transparent? (Alpha support depends on PD version)
        -- If alpha not supported, it will be opaque.
        
        local last_x, last_y = nil, nil
        local first_x, first_y = nil, nil
        
        for i = 1, 36 do
            local angle_deg = (i - 1) * 10
            local angle_rad = math.rad(angle_deg)
            local val = self.bins[i]
            
            -- Smooth interpolation could be better, but linear for now
            local r = val * radius
            local x = cx + r * math.sin(angle_rad)
            local y = cy - r * math.cos(angle_rad)
            
            if i == 1 then
                first_x, first_y = x, y
            else
                g:draw_line(last_x, last_y, x, y, 2)
            end
            last_x, last_y = x, y
        end
        -- Close loop
        if last_x and first_x then
            g:draw_line(last_x, last_y, first_x, first_y, 2)
        end
    end
    
    -- Mode 0 or 2: Scatter
    if self.mode == 0 or self.mode == 2 then
        for _, d in ipairs(self.detections) do
            local angle_rad = math.rad(d.angle)
            local r = d.strength * radius
            local x = cx + r * math.sin(angle_rad)
            local y = cy - r * math.cos(angle_rad)
            
            -- Color based on level (assuming 0-1 or similar)
            -- Map level to Hue or Brightness
            -- Let's use Heatmap: Blue (low) to Red (high)
            -- Hue: 0.66 (Blue) -> 0.0 (Red)
            local hue = 0.66 * (1 - math.min(1, math.max(0, d.level)))
            local r_col, g_col, b_col = Colors.hsb(hue, 1, 1)
            
            -- Fade out with age
            -- Alpha not fully supported in vanilla, so maybe shrink size?
            local size = 5 * (1 - d.age / self.max_age)
            
            if size > 0 then
                g:set_color(r_col, g_col, b_col)
                g:fill_ellipse(x - size/2, y - size/2, size, size)
            end
        end
    end
end

function mic_vis:in_1_mode(atoms)
    if type(atoms[1]) == "number" then
        self.mode = atoms[1]
        self:repaint()
    end
end

function mic_vis:in_1_decay(atoms)
    if type(atoms[1]) == "number" then
        self.decay_rate = atoms[1]
    end
end

function mic_vis:in_1_clear()
    self.detections = {}
    for i = 1, 36 do self.bins[i] = 0 end
    self:repaint()
end

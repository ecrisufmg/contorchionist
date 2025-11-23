-- ctgui.clock.pd_lua - Clock Display
--
-- Usage:
--   [ctgui.clock]
--   [ctgui.clock @w 100 @h 30 @fontsize 14]
--
-- Arguments:
--   @width <val>    : Width (default: 80)
--   @height <val>   : Height (default: 20)
--   @fontsize <val> : Font size (default: auto)
--   @color <r g b>  : Text color
--   @bgcolor <r g b>: Background color
--   @dark           : Dark mode theme
--   @guifps <val>   : GUI refresh rate

local ArgParser = require("pd_arg_parser")
local Colors = require("colors")

local ctgui_clock = pd.Class:new():register("ctgui.clock")

-- Default Colors (HSB)
local C_BG_LIGHT = {0, 0, 0.93}
local C_BG_DARK = {0, 0, 0.35}
local C_TEXT_LIGHT = {0, 0, 0.2}
local C_TEXT_DARK = {0, 0, 0.9}

function ctgui_clock:initialize(sel, atoms)
    self.creation_args = atoms
    local parser = ArgParser:new(atoms)

    -- Dimensions
    self.width = parser:get_float("width w", 80)
    self.height = parser:get_float("height h", 20)
    
    -- Font size
    self.fontsize = parser:get_float("fontsize fs", 0) -- 0 means auto

    -- Colors
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

    local function get_color(name, default_hsb)
        local val = parser:get_value(name .. "color")
        
        local h, s, b = default_hsb[1], default_hsb[2], default_hsb[3]
        local source = "default"

        if type(val) == "table" then
            if type(val[1]) == "string" then
                local mode = val[1]
                if (mode == "rgb" or mode == "RGB") and #val >= 4 then
                    h, s, b = rgb_to_hsb(val[2], val[3], val[4])
                    source = "rgb"
                elseif (mode == "hsb" or mode == "HSB" or mode == "hsl" or mode == "HSL") and #val >= 4 then
                    h, s, b = val[2], val[3], val[4]
                    source = "hsb"
                end
            elseif type(val[1]) == "number" and #val >= 3 then
                h, s, b = val[1], val[2], val[3]
                source = "hsb"
            end
        end
        
        local r, g, b_val = Colors.hsb(h, s, b)
        return {r, g, b_val, source=source, h=h, s=s, b=b}
    end

    local is_dark = parser:get_bool("dark")
    
    self.c_bg = get_color("bg", is_dark and C_BG_DARK or C_BG_LIGHT)
    self.c_text = get_color("text", is_dark and C_TEXT_DARK or C_TEXT_LIGHT)
    
    -- Check for @color alias for text color
    if parser:has_flag("color") then
        local val = parser:get_value("color")
        -- Reuse get_color logic manually or just re-parse if needed, 
        -- but simpler to just check if textcolor wasn't set explicitly?
        -- Actually get_color("text") checks @textcolor. 
        -- Let's just check @color manually if needed.
        -- But for simplicity, let's assume user uses @textcolor or we add logic.
        -- Let's add a quick check for @color if @textcolor wasn't found?
        -- The parser consumes flags.
        -- Let's just use the helper properly.
    end
    -- Re-implementing get_color to check multiple keys would be better, 
    -- but for now let's stick to the pattern. 
    -- If user uses @color, we might want to support it as text color.
    local color_val = parser:get_value("color")
    if color_val then
        -- Parse color_val similar to get_color
        local h, s, b = 0, 0, 0
        if type(color_val) == "table" then
             if type(color_val[1]) == "string" then
                local mode = color_val[1]
                if (mode == "rgb" or mode == "RGB") and #color_val >= 4 then
                    h, s, b = rgb_to_hsb(color_val[2], color_val[3], color_val[4])
                elseif (mode == "hsb" or mode == "HSB") and #color_val >= 4 then
                    h, s, b = color_val[2], color_val[3], color_val[4]
                end
            elseif type(color_val[1]) == "number" and #color_val >= 3 then
                h, s, b = color_val[1], color_val[2], color_val[3]
            end
        end
        local r, g, b_val = Colors.hsb(h, s, b)
        self.c_text = {r, g, b_val, source="hsb", h=h, s=s, b=b}
    end

    -- FPS
    local gui_fps = parser:get_float("guifps guishutter fps shutter", 20)
    self.gui_fps = (gui_fps > 0) and gui_fps or 20
    
    -- Data FPS
    local data_fps = parser:get_float("datafps datashutter", 0)
    self.data_fps = (data_fps > 0) and data_fps or 0

    -- State
    self.seconds = 0

    -- Clocks
    self.repaint_clock = pd.Clock:new():register(self, "repaint_tick")
    self.repaint_clock_running = false
    self.repaint_pending = false
    
    self.data_clock = pd.Clock:new():register(self, "data_tick")
    self.data_clock_running = false
    self.data_pending = false
    self.pending_out_time = nil
    self.pending_out_changed = false
    
    -- Inlets
    self.inlets = 1
    self.outlets = 1

    -- Override repaint for throttling
    self._raw_repaint = self.repaint
    self.repaint = self.throttled_repaint

    return true
end

function ctgui_clock:postinitialize()
    self:set_size(self.width, self.height)
end

function ctgui_clock:in_1_float(f)
    self.seconds = f
    self:throttled_repaint()
    self:output_data()
end

function ctgui_clock:in_1_list(atoms)
    if type(atoms) == "table" and #atoms > 0 and type(atoms[1]) == "number" then
        self.seconds = atoms[1]
        self:throttled_repaint()
        self:output_data()
    elseif type(atoms) == "number" then
        self.seconds = atoms
        self:throttled_repaint()
        self:output_data()
    end
end

function ctgui_clock:in_1_set(f)
    if type(f) == "number" then
        self.seconds = f
        self:throttled_repaint()
    end
end

function ctgui_clock:output_data()
    if self.data_fps > 0 then
        self.pending_out_time = self.seconds
        self.pending_out_changed = true
        self:throttled_data_output()
    else
        local s = math.max(0, self.seconds)
        local m = math.floor(s / 60)
        local sec = s % 60
        self:outlet(1, "list", {m, sec})
    end
end

function ctgui_clock:throttled_data_output()
    if self.data_clock_running then
        self.data_pending = true
        return
    end
    
    if self.pending_out_changed then
        local s = math.max(0, self.pending_out_time)
        local m = math.floor(s / 60)
        local sec = s % 60
        self:outlet(1, "list", {m, sec})
        
        self.pending_out_changed = false
    end
    
    self.data_clock_running = true
    self.data_clock:delay(1000 / self.data_fps)
end

function ctgui_clock:data_tick()
    self.data_clock_running = false
    if self.data_pending then
        self.data_pending = false
        self:throttled_data_output()
    end
end

function ctgui_clock:throttled_repaint()
    if self.repaint_clock_running then
        self.repaint_pending = true
        return
    end
    self:_raw_repaint()
    self.repaint_clock_running = true
    self.repaint_clock:delay(1000 / self.gui_fps)
end

function ctgui_clock:repaint_tick()
    self.repaint_clock_running = false
    if self.repaint_pending then
        self.repaint_pending = false
        self:throttled_repaint()
    end
end

function ctgui_clock:paint(g)
    -- Background
    g:set_color(self.c_bg[1], self.c_bg[2], self.c_bg[3])
    g:fill_rect(0, 0, self.width, self.height)
    
    -- Border
    -- g:set_color(200, 200, 200)
    -- g:stroke_rect(0, 0, self.width, self.height, 1)

    -- Format Time
    local s = math.max(0, self.seconds)
    local m = math.floor(s / 60)
    local sec = math.floor(s % 60)
    local deci = math.floor((s * 10) % 10)
    
    local str = string.format("%02d'%02d.%d", m, sec, deci)
    
    -- Font Size
    local fs = self.fontsize
    if fs <= 0 then
        -- Auto calculate based on height
        fs = math.floor(self.height * 0.8)
    end
    
    -- Draw Text
    g:set_color(self.c_text[1], self.c_text[2], self.c_text[3])
    
    -- Center text manually
    -- We estimate width to center it, but pass a large width to draw_text to avoid wrapping/clipping
    local char_w = fs * 0.55
    local est_w = #str * char_w
    local x = (self.width - est_w) / 2
    local y = (self.height - fs) / 2
    
    -- Ensure x is not negative
    -- x = math.max(0, x)
    
    -- Pass a generous width to draw_text so it doesn't try to squeeze the text
    g:draw_text(str, x, y, math.max(self.width, est_w * 1.5), fs)
end

function ctgui_clock:in_1_getcode()
    local str = "ctgui.clock"
    for i, v in ipairs(self.creation_args) do
        if type(v) == "number" then
            local s = string.format("%.2f", v)
            s = s:gsub("0+$", ""):gsub("%.$", "")
            str = str .. " " .. s
        else
            str = str .. " " .. tostring(v)
        end
    end
    pd.post(str)
end

function ctgui_clock:in_1_guifps(atoms)
    local f = type(atoms) == "table" and atoms[1] or atoms
    if type(f) == "number" and f > 0 then
        self.gui_fps = f
    end
end

function ctgui_clock:in_1_guishutter(atoms)
    self:in_1_guifps(atoms)
end

function ctgui_clock:in_1_fps(atoms)
    self:in_1_guifps(atoms)
end

function ctgui_clock:in_1_shutter(atoms)
    self:in_1_guifps(atoms)
end

function ctgui_clock:in_1_datafps(atoms)
    local f = type(atoms) == "table" and atoms[1] or atoms
    if type(f) == "number" then
        self.data_fps = (f > 0) and f or 0
    end
end

function ctgui_clock:in_1_datashutter(atoms)
    self:in_1_datafps(atoms)
end

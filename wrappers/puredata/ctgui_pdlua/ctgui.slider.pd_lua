-- ctgui.slider.pd_lua - Slider with Linear, Log, and dB modes
--
-- Usage:
--   [ctgui.slider @min 0 @max 1 @mode lin]
--   [ctgui.slider @min -120 @max 12 @mode db]
--
-- Arguments:
--   @min <val>      : Minimum value (default: 0 for lin/log, -120 for db)
--   @max <val>      : Maximum value (default: 1 for lin/log, 12 for db)
--   @mode <mode>    : "lin" (linear), "log" (logarithmic), "db" (dbfader)
--   @width <val>    : Width (default: 20)
--   @height <val>   : Height (default: 120)
--   @color <r g b>  : Handle color
--   @bgcolor <r g b>: Background color
--   @slotcolor <r g b>: Slot/Track color
--   @dark           : Dark mode theme
--   @guifps <val>   : GUI refresh rate
--   @datafps <val>  : Data output rate limit

local ArgParser = require("pd_arg_parser")
local Colors = require("colors")

local ctgui_slider = pd.Class:new():register("ctgui.slider")

-- Default Colors (HSB)
local C_BG_LIGHT = {0, 0, 0.9}
local C_BG_DARK = {0, 0, 0.35}
local C_SLOT_LIGHT = {0, 0, 0.7}
local C_SLOT_DARK = {0, 0, 0.2}
local C_HANDLE_LIGHT = {0, 0, 0.45}
local C_HANDLE_DARK = {0, 0, 0.75}
local C_MARK_LIGHT = {0, 0, 0.6}
local C_MARK_DARK = {0, 0, 0.6}

function ctgui_slider:initialize(sel, atoms)
    self.creation_args = atoms
    local parser = ArgParser:new(atoms)

    -- Dimensions
    self.width = parser:get_float("width w", 20)
    self.height = parser:get_float("height h", 120)
    
    -- Orientation
    if parser:has_flag("vert vertical") then
        self.orientation = "vertical"
    elseif parser:has_flag("horiz horizontal") then
        self.orientation = "horizontal"
    else
        -- Inferred from dimensions
        self.orientation = (self.width > self.height) and "horizontal" or "vertical"
    end

    -- Mode
    local mode_str = parser:get_string("mode m", "lin")
    if mode_str == "log" or mode_str == "exponential" then
        self.mode = "log"
    elseif mode_str == "db" or mode_str == "dbfader" then
        self.mode = "db"
    else
        self.mode = "lin"
    end

    -- Min/Max
    local default_min = (self.mode == "db") and -120 or 0
    local default_max = (self.mode == "db") and 12 or 1
    self.min_val = parser:get_float("min minimum", default_min)
    self.max_val = parser:get_float("max maximum", default_max)

    -- Colors
    
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
    local function get_color(name, default_hsb)
        -- Check for @namecolor (e.g. @playcolor)
        local val = parser:get_value(name .. "color")
        
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
            -- Check for format: {h, s, b} (implicit HSB)
            elseif type(val[1]) == "number" and #val >= 3 then
                h, s, b = val[1], val[2], val[3]
                source = "hsb"
            end
        end

        -- Convert to RGB (0-255) for drawing
        local r, g, b_val = Colors.hsb(h, s, b)
        return {r, g, b_val, source=source, h=h, s=s, b=b}
    end

    -- Dark mode
    local is_dark = parser:get_bool("dark")
    
    self.c_bg = get_color("bg", is_dark and C_BG_DARK or C_BG_LIGHT)
    -- Also check "bgcolor" for backward compatibility if needed, but get_color checks "name" .. "color"
    -- So "bg" -> "bgcolor". 
    -- But wait, previous code checked "bgcolor" AND "bg".
    -- parser:get_value("bgcolor") might be needed if user uses @bgcolor.
    -- My get_color does `name .. "color"`. So `get_color("bg")` checks `@bgcolor`.
    -- If user uses `@bg`, it won't be found by `get_color("bg")` which looks for `@bgcolor`.
    -- Let's adjust get_color to check both if needed, or just pass "bg" and rely on "bgcolor".
    -- The previous code used `parse_color("bgcolor bg", ...)` which checked both.
    
    -- Let's refine get_color to take a list of keys or handle the suffix better.
    -- Or just manually check.
    
    -- Actually, let's stick to the pattern:
    -- get_color("bg") -> checks @bgcolor.
    -- If I want to check @bg, I need to pass it.
    
    -- Let's redefine get_color to be more flexible.
    
    local function get_color_flexible(keys, default_hsb)
        local val = nil
        for _, key in ipairs(keys) do
            val = parser:get_value(key)
            if val then break end
        end
        
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

    self.c_bg = get_color_flexible({"bgcolor", "bg"}, is_dark and C_BG_DARK or C_BG_LIGHT)
    self.c_slot = get_color_flexible({"slotcolor", "trackcolor"}, is_dark and C_SLOT_DARK or C_SLOT_LIGHT)
    self.c_handle = get_color_flexible({"color", "handlecolor"}, is_dark and C_HANDLE_DARK or C_HANDLE_LIGHT)
    self.c_mark = get_color_flexible({"markcolor", "mark"}, is_dark and C_MARK_DARK or C_MARK_LIGHT)

    -- FPS
    local gui_fps = parser:get_float("guifps guishutter fps shutter", 20)
    self.gui_fps = (gui_fps > 0) and gui_fps or 20
    
    local data_fps = parser:get_float("datafps datashutter", 0)
    self.data_fps = (data_fps > 0) and data_fps or 0

    -- State
    self.current_value = self.min_val
    self.visual_pos = 0 -- 0.0 to 1.0
    self.dragging = false

    -- Clocks
    self.repaint_clock = pd.Clock:new():register(self, "repaint_tick")
    self.repaint_clock_running = false
    self.repaint_pending = false
    
    self.data_clock = pd.Clock:new():register(self, "data_tick")
    self.data_clock_running = false
    self.data_pending = false
    self.pending_out_value = nil
    self.pending_out_changed = false
    self.pending_ctrl_value = nil
    self.pending_ctrl_changed = false

    -- Line Clock
    self.line_clock = pd.Clock:new():register(self, "line_tick")
    self.line_running = false
    self.line_grain = parser:get_float("linegrain", 20)
    self.default_line_ms = parser:get_float("linems", 0)
    
    -- Easing
    local curve = parser:get_value("reasing curve easing")
    if type(curve) == "number" then
        self.default_easing = curve
    elseif type(curve) == "string" then
        self.default_easing = curve
    else
        self.default_easing = "line"
    end

    -- Controller Input/Output
    -- Default to normalized 0-1
    self.v_in_min = 0
    self.v_in_max = 1
    self.v_out_min = 0
    self.v_out_max = 1

    if parser:has_flag("ctin") then
        self.v_in_min = 0
        self.v_in_max = 127
        self.v_out_min = 0
        self.v_out_max = 127
    end

    if parser:has_flag("bendin") then
        self.v_in_min = 0
        self.v_in_max = 16383
        self.v_out_min = 0
        self.v_out_max = 16383
    end

    local valin = parser:get_float_list("valin")
    if #valin >= 2 then
        self.v_in_min = valin[1]
        self.v_in_max = valin[2]
    end

    local valout = parser:get_float_list("valout")
    if #valout >= 2 then
        self.v_out_min = valout[1]
        self.v_out_max = valout[2]
    end

    -- Inlets/Outlets
    self.inlets = 2
    self.outlets = 2

    -- Initial visual pos
    self.handle_size = 10
    self:update_visual_from_value()

    -- Override repaint for throttling
    self._raw_repaint = self.repaint
    self.repaint = self.throttled_repaint

    return true
end

function ctgui_slider:get_zero_visual()
    -- Calculate visual position (0-1) that corresponds to 0dB
    -- to align with ctgui.vu which uses 0.85 of (height-2)
    
    local vu_zero_ratio = 0.85
    local vu_margin = 2
    
    local dim = (self.orientation == "vertical") and self.height or self.width
    local handle = self.handle_size or 10
    
    -- VU 0dB position from bottom (pixels)
    local vu_zero_px = vu_zero_ratio * (dim - vu_margin)
    
    -- Slider 0dB position from bottom (pixels)
    -- pos_px = handle/2 + visual * (dim - handle)
    -- We want pos_px = vu_zero_px
    
    local slider_track_len = dim - handle
    if slider_track_len <= 0 then return 0.85 end
    
    local visual = (vu_zero_px - (handle / 2)) / slider_track_len
    
    -- Clamp to reasonable bounds
    return math.max(0.1, math.min(0.9, visual))
end

function ctgui_slider:in_1_getcode()
    local str = "ctgui.slider"
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

function ctgui_slider:postinitialize()
    self:set_size(self.width, self.height)
end

-- Transfer Functions

function ctgui_slider:value_to_visual(val)
    if self.mode == "lin" then
        return (val - self.min_val) / (self.max_val - self.min_val)
    elseif self.mode == "log" then
        if val <= 0 then return 0 end
        local min_log = math.log(math.max(0.0001, self.min_val))
        local max_log = math.log(self.max_val)
        return (math.log(val) - min_log) / (max_log - min_log)
    elseif self.mode == "db" then
        return self:db_to_visual(val)
    end
    return 0
end

function ctgui_slider:visual_to_value(vis)
    vis = math.max(0, math.min(1, vis))
    if self.mode == "lin" then
        return self.min_val + vis * (self.max_val - self.min_val)
    elseif self.mode == "log" then
        local min_log = math.log(math.max(0.0001, self.min_val))
        local max_log = math.log(self.max_val)
        return math.exp(min_log + vis * (max_log - min_log))
    elseif self.mode == "db" then
        return self:visual_to_db(vis)
    end
    return self.min_val
end

-- DB Transfer Functions (Copied/Adapted from ctgui.vu)
function ctgui_slider:db_to_visual(db)
    -- Clamp db
    db = math.max(self.min_val, math.min(self.max_val, db))
    
    -- Zero point visual (0dB)
    -- Calculated dynamically to align with ctgui.vu
    local zero_visual = self:get_zero_visual()
    
    -- If 0 is not in range, this logic might be weird, but for a fader usually it is.
    -- Let's assume standard audio fader range.
    
    if db <= 0 then
        local linear = (db - self.min_val) / (0 - self.min_val)
        linear = math.max(0, math.min(1, linear))
        
        local visual
        if linear < 0.5 then
            visual = (0.2 * zero_visual) * (linear / 0.5)
        elseif linear < 0.833 then
            local t = (linear - 0.5) / (0.833 - 0.5)
            visual = (0.2 * zero_visual) + (0.3 * zero_visual * t)
        else
            local t = (linear - 0.833) / (1.0 - 0.833)
            visual = (0.5 * zero_visual) + (0.5 * zero_visual * t)
        end
        return visual
    else
        local t = db / (self.max_val - 0)
        return zero_visual + ((1.0 - zero_visual) * t)
    end
end

function ctgui_slider:visual_to_db(visual)
    visual = math.max(0, math.min(1, visual))
    local zero_visual = self:get_zero_visual()
    
    if visual <= zero_visual then
        local linear
        if visual < 0.2 * zero_visual then
            linear = 0.5 * (visual / (0.2 * zero_visual))
        elseif visual < 0.5 * zero_visual then
            local t = (visual - 0.2 * zero_visual) / (0.3 * zero_visual)
            linear = 0.5 + (0.333 * t)
        else
            local t = (visual - 0.5 * zero_visual) / (0.5 * zero_visual)
            linear = 0.833 + (0.167 * t)
        end
        local db = self.min_val + (linear * (0 - self.min_val))
        return db
    else
        local t = (visual - zero_visual) / (1.0 - zero_visual)
        local db = 0 + (t * (self.max_val - 0))
        return db
    end
end

-- Interaction

function ctgui_slider:update_visual_from_value()
    self.visual_pos = self:value_to_visual(self.current_value)
end

function ctgui_slider:set_value_from_mouse(x, y)
    local vis
    if self.orientation == "vertical" then
        -- 0 at bottom, 1 at top
        vis = 1.0 - (y / self.height)
    else
        -- 0 at left, 1 at right
        vis = x / self.width
    end
    
    -- Clamp
    vis = math.max(0, math.min(1, vis))
    self.visual_pos = vis
    self.current_value = self:visual_to_value(vis)
    
    -- Output
    self:output_value()
    self:throttled_repaint()
end

function ctgui_slider:mouse_down(x, y, button, mod)
    self:stop_line()
    self.dragging = true
    self:set_value_from_mouse(x, y)
    return true
end

function ctgui_slider:mouse_drag(x, y, button, mod)
    if self.dragging then
        self:set_value_from_mouse(x, y)
    end
end

function ctgui_slider:mouse_up(x, y, button, mod)
    self.dragging = false
end

-- Line Logic

function ctgui_slider:stop_line()
    if self.line_running then
        self.line_clock:unset()
        self.line_running = false
    end
end

function ctgui_slider:calculate_easing(t, mode)
    if mode == "linear" or mode == "line" then return t end
    
    -- Numeric (Power)
    local mode_num = tonumber(mode)
    if mode_num then
        if mode_num == 0 then return t end
        if mode_num > 0 then return t ^ mode_num end -- Ease In
        return 1 - ((1 - t) ^ math.abs(mode_num)) -- Ease Out
    end

    -- Sine
    if mode == "sine-in" then return 1 - math.cos((t * math.pi) / 2)
    elseif mode == "sine-out" then return math.sin((t * math.pi) / 2)
    elseif mode == "sine-inout" then return -(math.cos(math.pi * t) - 1) / 2
    
    -- Quad
    elseif mode == "quad-in" then return t * t
    elseif mode == "quad-out" then return 1 - (1 - t) * (1 - t)
    elseif mode == "quad-inout" then return t < 0.5 and 2 * t * t or 1 - ((-2 * t + 2)^2) / 2
    
    -- Cubic
    elseif mode == "cubic-in" then return t * t * t
    elseif mode == "cubic-out" then return 1 - (1 - t)^3
    elseif mode == "cubic-inout" then return t < 0.5 and 4 * t * t * t or 1 - ((-2 * t + 2)^3) / 2
    
    -- Quart
    elseif mode == "quart-in" then return t * t * t * t
    elseif mode == "quart-out" then return 1 - (1 - t)^4
    elseif mode == "quart-inout" then return t < 0.5 and 8 * t * t * t * t or 1 - ((-2 * t + 2)^4) / 2
    
    -- Quint
    elseif mode == "quint-in" then return t * t * t * t * t
    elseif mode == "quint-out" then return 1 - (1 - t)^5
    elseif mode == "quint-inout" then return t < 0.5 and 16 * t * t * t * t * t or 1 - ((-2 * t + 2)^5) / 2
    
    -- Sextic (Power of 6)
    elseif mode == "sextic-in" then return t^6
    elseif mode == "sextic-out" then return 1 - (1 - t)^6
    elseif mode == "sextic-inout" then return t < 0.5 and 32 * t^6 or 1 - ((-2 * t + 2)^6) / 2
    
    -- Expo
    elseif mode == "expo-in" then return t == 0 and 0 or 2^(10 * t - 10)
    elseif mode == "expo-out" then return t == 1 and 1 or 1 - 2^(-10 * t)
    elseif mode == "expo-inout" then
        if t == 0 then return 0 end
        if t == 1 then return 1 end
        if t < 0.5 then return (2^(20 * t - 10)) / 2 end
        return (2 - 2^(-20 * t + 10)) / 2
    
    -- Circ
    elseif mode == "circ-in" then return 1 - math.sqrt(1 - t^2)
    elseif mode == "circ-out" then return math.sqrt(1 - (t - 1)^2)
    elseif mode == "circ-inout" then
        if t < 0.5 then return (1 - math.sqrt(1 - (2 * t)^2)) / 2 end
        return (math.sqrt(1 - (-2 * t + 2)^2) + 1) / 2
    
    -- Back
    elseif mode == "back-in" then 
        local c1 = 1.70158; local c3 = c1 + 1
        return c3 * t * t * t - c1 * t * t
    elseif mode == "back-out" then 
        local c1 = 1.70158; local c3 = c1 + 1
        return 1 + c3 * (t - 1)^3 + c1 * (t - 1)^2
    elseif mode == "back-inout" then
        local c1 = 1.70158; local c2 = c1 * 1.525
        if t < 0.5 then
            return ((2 * t)^2 * ((c2 + 1) * 2 * t - c2)) / 2
        end
        return ((2 * t - 2)^2 * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2
    
    -- Bounce
    elseif mode == "bounce-out" then
        local n1 = 7.5625; local d1 = 2.75
        if t < 1 / d1 then return n1 * t * t
        elseif t < 2 / d1 then t = t - 1.5 / d1; return n1 * t * t + 0.75
        elseif t < 2.5 / d1 then t = t - 2.25 / d1; return n1 * t * t + 0.9375
        else t = t - 2.625 / d1; return n1 * t * t + 0.984375 end
    elseif mode == "bounce-in" then return 1 - self:calculate_easing(1 - t, "bounce-out")
    elseif mode == "bounce-inout" then
        if t < 0.5 then return (1 - self:calculate_easing(1 - 2 * t, "bounce-out")) / 2 end
        return (1 + self:calculate_easing(2 * t - 1, "bounce-out")) / 2
    
    -- Hann
    elseif mode == "hann" then return 0.5 * (1 - math.cos(math.pi * t))
    end

    return t
end

function ctgui_slider:start_line(target, time_ms, easing)
    self:stop_line()
    
    target = math.max(self.min_val, math.min(self.max_val, target))
    
    if time_ms <= 0 then
        self:in_1_float(target)
        return
    end
    
    local ticks = math.ceil(time_ms / self.line_grain)
    if ticks < 1 then ticks = 1 end
    
    self.line_target = target
    self.line_start_val = self.current_value
    self.line_total_ticks = ticks
    self.line_current_tick = 0
    self.line_easing = easing or self.default_easing
    
    self.line_running = true
    self.line_clock:delay(self.line_grain)
end

function ctgui_slider:line_tick()
    if not self.line_running then return end
    
    self.line_current_tick = self.line_current_tick + 1
    local t = self.line_current_tick / self.line_total_ticks
    if t > 1 then t = 1 end
    
    -- Easing
    local eased_t = self:calculate_easing(t, self.line_easing)
    -- "line" or 0 or unknown -> linear (t)
    
    self.current_value = self.line_start_val + (self.line_target - self.line_start_val) * eased_t
    
    -- Check bounds/completion
    local finished = false
    if self.line_current_tick >= self.line_total_ticks then
        self.current_value = self.line_target
        finished = true
    end
    
    self.current_value = math.max(self.min_val, math.min(self.max_val, self.current_value))
    
    self:update_visual_from_value()
    self:output_value()
    self:throttled_repaint()
    
    if finished then
        self.line_running = false
    else
        self.line_clock:delay(self.line_grain)
    end
end

-- Input

function ctgui_slider:in_1_list(atoms)
    if type(atoms) == "table" and #atoms > 0 and type(atoms[1]) == "number" then
        if #atoms >= 2 and type(atoms[2]) == "number" and atoms[2] > 0 then
            local easing = nil
            if #atoms >= 3 then easing = atoms[3] end
            self:start_line(atoms[1], atoms[2], easing)
        else
            if self.default_line_ms > 0 then
                self:start_line(atoms[1], self.default_line_ms)
            else
                self:in_1_float(atoms[1])
            end
        end
    elseif type(atoms) == "number" then
        self:in_1_float(atoms)
    end
end

function ctgui_slider:in_1_float(f)
    if self.default_line_ms > 0 then
        self:start_line(f, self.default_line_ms)
        return
    end

    self:stop_line()
    self.current_value = math.max(self.min_val, math.min(self.max_val, f))
    self:update_visual_from_value()
    self:throttled_repaint()
    -- Do not output on input to avoid loops, usually? 
    -- Standard PD slider outputs on set? No, usually not.
    -- But if it's a "set" message vs float.
    -- Float usually sets and outputs.
    self:output_value()
end

function ctgui_slider:in_1_set(f)
    self:stop_line()
    self.current_value = math.max(self.min_val, math.min(self.max_val, f))
    self:update_visual_from_value()
    self:throttled_repaint()
end

function ctgui_slider:in_2_list(atoms)
    if type(atoms) == "table" and #atoms > 0 and type(atoms[1]) == "number" then
        if #atoms >= 2 and type(atoms[2]) == "number" and atoms[2] > 0 then
            local f = atoms[1]
            local norm = (f - self.v_in_min) / (self.v_in_max - self.v_in_min)
            norm = math.max(0, math.min(1, norm))
            local target = self:visual_to_value(norm)
            local easing = nil
            if #atoms >= 3 then easing = atoms[3] end
            self:start_line(target, atoms[2], easing)
        else
            self:in_2_float(atoms[1])
        end
    elseif type(atoms) == "number" then
        self:in_2_float(atoms)
    end
end

function ctgui_slider:in_2_float(f)
    if self.default_line_ms and self.default_line_ms > 0 then
        local norm = (f - self.v_in_min) / (self.v_in_max - self.v_in_min)
        norm = math.max(0, math.min(1, norm))
        local target = self:visual_to_value(norm)
        self:start_line(target, self.default_line_ms)
        return
    end

    self:stop_line()
    -- Map input range to 0-1
    local norm = (f - self.v_in_min) / (self.v_in_max - self.v_in_min)
    norm = math.max(0, math.min(1, norm))
    
    self.visual_pos = norm
    self.current_value = self:visual_to_value(norm)
    
    self:output_value()
    self:throttled_repaint()
end

-- Output Throttling

function ctgui_slider:output_value()
    if self.data_fps > 0 then
        self.pending_out_value = self.current_value
        self.pending_out_changed = true
        
        local ctrl_val = self.v_out_min + self.visual_pos * (self.v_out_max - self.v_out_min)
        self.pending_ctrl_value = ctrl_val
        self.pending_ctrl_changed = true
        
        self:throttled_data_output()
    else
        self:outlet(1, "float", {self.current_value})
        local ctrl_val = self.v_out_min + self.visual_pos * (self.v_out_max - self.v_out_min)
        self:outlet(2, "float", {ctrl_val})
    end
end

function ctgui_slider:throttled_data_output()
    if self.data_clock_running then
        self.data_pending = true
        return
    end
    
    if self.pending_out_changed then
        self:outlet(1, "float", {self.pending_out_value})
        self.pending_out_changed = false
    end
    
    if self.pending_ctrl_changed then
        self:outlet(2, "float", {self.pending_ctrl_value})
        self.pending_ctrl_changed = false
    end
    
    self.data_clock_running = true
    self.data_clock:delay(1000 / self.data_fps)
end

-- Dynamic Configuration

function ctgui_slider:in_1_guifps(atoms)
    local f = type(atoms) == "table" and atoms[1] or atoms
    if type(f) == "number" and f > 0 then
        self.gui_fps = f
    end
end

function ctgui_slider:in_1_guishutter(atoms)
    self:in_1_guifps(atoms)
end

function ctgui_slider:in_1_fps(atoms)
    self:in_1_guifps(atoms)
end

function ctgui_slider:in_1_shutter(atoms)
    self:in_1_guifps(atoms)
end

function ctgui_slider:in_1_datafps(atoms)
    local f = type(atoms) == "table" and atoms[1] or atoms
    if type(f) == "number" then
        self.data_fps = (f > 0) and f or 0
    end
end

function ctgui_slider:in_1_datashutter(atoms)
    self:in_1_datafps(atoms)
end

function ctgui_slider:in_1_linems(atoms)
    local f = type(atoms) == "table" and atoms[1] or atoms
    if type(f) == "number" then
        self.default_line_ms = math.max(0, f)
    end
end

function ctgui_slider:in_1_linegrain(atoms)
    local f = type(atoms) == "table" and atoms[1] or atoms
    if type(f) == "number" then
        self.line_grain = math.max(1, f)
    end
end

function ctgui_slider:in_1_reasing(atoms)
    local v = type(atoms) == "table" and atoms[1] or atoms
    if type(v) == "number" or type(v) == "string" then
        self.default_easing = v
    end
end

function ctgui_slider:in_1_curve(atoms)
    self:in_1_reasing(atoms)
end

function ctgui_slider:in_1_easing(atoms)
    self:in_1_reasing(atoms)
end

function ctgui_slider:data_tick()
    self.data_clock_running = false
    if self.data_pending then
        self.data_pending = false
        self:throttled_data_output()
    end
end

-- Painting

function ctgui_slider:throttled_repaint()
    if self.repaint_clock_running then
        self.repaint_pending = true
        return
    end
    self:_raw_repaint()
    self.repaint_clock_running = true
    self.repaint_clock:delay(1000 / self.gui_fps)
end

function ctgui_slider:repaint_tick()
    self.repaint_clock_running = false
    if self.repaint_pending then
        self.repaint_pending = false
        self:throttled_repaint()
    end
end

function ctgui_slider:paint(g)
    -- Safety check
    if not self.c_bg then self.c_bg = {237, 237, 237} end
    if not self.c_slot then self.c_slot = {200, 200, 200} end
    if not self.c_handle then self.c_handle = {150, 150, 150} end

    -- Background
    g:set_color(self.c_bg[1], self.c_bg[2], self.c_bg[3])
    g:fill_rect(0, 0, self.width, self.height)

    -- Slot/Track
    g:set_color(self.c_slot[1], self.c_slot[2], self.c_slot[3])
    
    if self.orientation == "vertical" then
        local slot_width = 4
        local slot_x = (self.width - slot_width) / 2
        g:fill_rounded_rect(slot_x, 4, slot_width, self.height - 8, 2)
    else
        local slot_height = 4
        local slot_y = (self.height - slot_height) / 2
        g:fill_rounded_rect(4, slot_y, self.width - 8, slot_height, 2)
    end

    -- 0dB Mark
    if self.mode == "db" and self.min_val <= 0 and self.max_val >= 0 then
        if not self.c_mark then self.c_mark = {128, 128, 128} end
        g:set_color(self.c_mark[1], self.c_mark[2], self.c_mark[3])
        local zero_vis = self:value_to_visual(0)
        local handle_thickness = self.handle_size or 10
        
        if self.orientation == "vertical" then
            local track_len = self.height - handle_thickness
            local mark_y = (self.height - handle_thickness) - (zero_vis * track_len) + (handle_thickness / 2)
            -- Draw horizontal line
            g:fill_rect(2, mark_y, self.width - 4, 1)
        else
            local track_len = self.width - handle_thickness
            local mark_x = (zero_vis * track_len) + (handle_thickness / 2)
            -- Draw vertical line
            g:fill_rect(mark_x, 2, 1, self.height - 4)
        end
    end

    -- Handle
    g:set_color(self.c_handle[1], self.c_handle[2], self.c_handle[3])
    
    local handle_thickness = self.handle_size or 10 -- Pixel size along the axis
    
    if self.orientation == "vertical" then
        -- Visual pos 0 is bottom, 1 is top
        -- Y coords: 0 is top, height is bottom
        local track_len = self.height - handle_thickness
        local y = (self.height - handle_thickness) - (self.visual_pos * track_len)
        g:fill_rounded_rect(2, y, self.width - 4, handle_thickness, 2)
    else
        local track_len = self.width - handle_thickness
        local x = self.visual_pos * track_len
        g:fill_rounded_rect(x, 2, handle_thickness, self.height - 4, 2)
    end
end

-- Override repaint in initialize
-- function ctgui_slider:initialize(...)
--    ...
--    self._raw_repaint = self.repaint
--    self.repaint = self.throttled_repaint
-- end

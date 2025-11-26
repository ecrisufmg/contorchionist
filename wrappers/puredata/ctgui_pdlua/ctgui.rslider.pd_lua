-- ctgui.rslider.pd_lua - Range Slider with Linear, Log, and dB modes
--
-- Usage:
--   [ctgui.rslider @min 0 @max 1 @mode lin]
--   [ctgui.rslider @min -120 @max 12 @mode db]
--   [ctgui.rslider @route] (Single inlet/outlet mode)
--
-- Arguments:
--   @min <val>      : Minimum value (default: 0 for lin/log, -120 for db)
--   @max <val>      : Maximum value (default: 1 for lin/log, 12 for db)
--   @mode <mode>    : "lin" (linear), "log" (logarithmic), "db" (dbfader)
--   @width <val>    : Width (default: 130)
--   @height <val>   : Height (default: 20)
--   @color <r g b>  : Handle color
--   @bgcolor <r g b>: Background color
--   @slotcolor <r g b>: Slot/Track color
--   @route          : Enable single inlet/outlet route mode
--   @guifps <val>   : GUI refresh rate
--   @datafps <val>  : Data output rate limit

local ArgParser = require("pd_arg_parser")
local Colors = require("colors")

local unpack = unpack or table.unpack

local ctgui_rslider = pd.Class:new():register("ctgui.rslider")

-- Default Colors (HSB)
local C_BG_LIGHT = {0, 0, 0.9}
local C_BG_DARK = {0, 0, 0.35}
local C_SLOT_LIGHT = {0, 0, 0.7}
local C_SLOT_DARK = {0, 0, 0.2}
local C_HANDLE_LIGHT = {0, 0, 0.45}
local C_HANDLE_DARK = {0, 0, 0.75}
local C_MARK_LIGHT = {0, 0, 0.6}
local C_MARK_DARK = {0, 0, 0.6}

function ctgui_rslider:initialize(sel, atoms)
    self.creation_args = atoms
    local parser = ArgParser:new(atoms)

    -- Dimensions
    self.width = parser:get_float("width w", 130)
    self.height = parser:get_float("height h", 20)
    
    -- Orientation
    if parser:has_flag("vert vertical") then
        self.orientation = "vertical"
    elseif parser:has_flag("horiz horizontal") then
        self.orientation = "horizontal"
    else
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

    -- Route Mode
    self.route_mode = parser:has_flag("route")
    self.pos_output = parser:has_flag("posoutput")
    self.pos_mode = parser:has_flag("pos")

    -- Initial Values
    local default_init = (self.mode == "db") and self.min_val or 0.0
    local init_low = default_init
    local init_high = default_init
    
    if parser:has_flag("low lo") then
        init_low = parser:get_float("low lo", default_init)
    end
    if parser:has_flag("high hi") then
        init_high = parser:get_float("high hi", default_init)
    end
    
    -- Clamp
    init_low = math.max(self.min_val, math.min(self.max_val, init_low))
    init_high = math.max(self.min_val, math.min(self.max_val, init_high))

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

    local is_dark = parser:get_bool("dark")
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
    self.val_low = init_low
    self.val_high = init_high
    self.vis_low = 0
    self.vis_high = 0
    self.dragging = false
    self.drag_target = nil -- "low", "high", "both"

    -- Clocks
    self.repaint_clock = pd.Clock:new():register(self, "repaint_tick")
    self.repaint_clock_running = false
    self.repaint_pending = false
    
    self.data_clock = pd.Clock:new():register(self, "data_tick")
    self.data_clock_running = false
    self.data_pending = false
    self.pending_out_low = nil
    self.pending_out_high = nil
    self.pending_out_changed = false
    self.pending_ctrl_low = nil
    self.pending_ctrl_high = nil
    self.pending_ctrl_changed = false

    -- Line Clocks
    self.line_clock_low = pd.Clock:new():register(self, "line_tick_low")
    self.line_running_low = false
    
    self.line_clock_high = pd.Clock:new():register(self, "line_tick_high")
    self.line_running_high = false

    self.line_grain = parser:get_float("linegrain", 20)
    self.default_line_ms = parser:get_float("linems", 0)
    
    -- Easing
    local curve = parser:get_value("reasing curve easing")
    if type(curve) == "number" or type(curve) == "string" then
        self.default_easing = curve
    else
        self.default_easing = "line"
    end

    -- Controller Input/Output
    self.v_in_min = 0
    self.v_in_max = 1
    self.v_out_min = 0
    self.v_out_max = 1

    if parser:has_flag("ctin") then
        self.v_in_min = 0; self.v_in_max = 127
        self.v_out_min = 0; self.v_out_max = 127
    end
    if parser:has_flag("bendin") then
        self.v_in_min = 0; self.v_in_max = 16383
        self.v_out_min = 0; self.v_out_max = 16383
    end
    local valin = parser:get_float_list("valin")
    if #valin >= 2 then self.v_in_min = valin[1]; self.v_in_max = valin[2] end
    local valout = parser:get_float_list("valout")
    if #valout >= 2 then self.v_out_min = valout[1]; self.v_out_max = valout[2] end

    -- Inlets/Outlets
    if self.route_mode then
        self.inlets = 1
        self.outlets = 1
    elseif self.pos_mode then
        self.inlets = 4
        self.outlets = 4
    else
        self.inlets = 2
        self.outlets = self.pos_output and 3 or 2
    end

    -- Initial visual pos
    self:update_visual_from_value()

    -- Override repaint for throttling
    self._raw_repaint = self.repaint
    self.repaint = self.throttled_repaint

    return true
end

function ctgui_rslider:get_handle_size()
    local max_dim = (self.orientation == "vertical") and self.width or self.height
    local handle_size = (max_dim - 4) * 0.66
    if handle_size < 4 then handle_size = 4 end
    return handle_size
end

function ctgui_rslider:get_zero_visual()
    local vu_zero_ratio = 0.85
    local vu_margin = 2
    local dim = (self.orientation == "vertical") and self.height or self.width
    local handle = self:get_handle_size()
    local vu_zero_px = vu_zero_ratio * (dim - vu_margin)
    local slider_track_len = dim - handle
    if slider_track_len <= 0 then return 0.85 end
    local visual = (vu_zero_px - (handle / 2)) / slider_track_len
    return math.max(0.1, math.min(0.9, visual))
end

function ctgui_rslider:in_1_getcode()
    local str = "ctgui.rslider"
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

function ctgui_rslider:postinitialize()
    self:set_size(self.width, self.height)
end

-- Transfer Functions

function ctgui_rslider:value_to_visual(val)
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

function ctgui_rslider:visual_to_value(vis)
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

function ctgui_rslider:db_to_visual(db)
    db = math.max(self.min_val, math.min(self.max_val, db))
    local zero_visual = self:get_zero_visual()
    
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

function ctgui_rslider:visual_to_db(visual)
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
        return self.min_val + (linear * (0 - self.min_val))
    else
        local t = (visual - zero_visual) / (1.0 - zero_visual)
        return 0 + (t * (self.max_val - 0))
    end
end

-- Interaction

function ctgui_rslider:update_visual_from_value()
    self.vis_low = self:value_to_visual(self.val_low)
    self.vis_high = self:value_to_visual(self.val_high)
end

function ctgui_rslider:set_value_from_mouse(x, y)
    local vis
    if self.orientation == "vertical" then
        vis = 1.0 - (y / self.height)
    else
        vis = x / self.width
    end
    vis = math.max(0, math.min(1, vis))
    
    if self.drag_target == "low" then
        local val = self:visual_to_value(vis)
        self.val_low = math.min(val, self.val_high)
        self:update_visual_from_value()
    elseif self.drag_target == "high" then
        local val = self:visual_to_value(vis)
        self.val_high = math.max(val, self.val_low)
        self:update_visual_from_value()
    elseif self.drag_target == "both" then
        local delta = vis - self.drag_start_vis
        local new_vis_low = self.drag_start_vis_low + delta
        local new_vis_high = self.drag_start_vis_high + delta
        
        if new_vis_low < 0 then delta = -self.drag_start_vis_low
        elseif new_vis_high > 1 then delta = 1 - self.drag_start_vis_high end
        
        new_vis_low = self.drag_start_vis_low + delta
        new_vis_high = self.drag_start_vis_high + delta
        
        self.vis_low = new_vis_low
        self.vis_high = new_vis_high
        self.val_low = self:visual_to_value(new_vis_low)
        self.val_high = self:visual_to_value(new_vis_high)
    end
    
    self:output_value()
    self:throttled_repaint()
end

function ctgui_rslider:mouse_down(x, y, button, mod)
    self:stop_line_low()
    self:stop_line_high()
    self.dragging = true
    
    local vis
    if self.orientation == "vertical" then
        vis = 1.0 - (y / self.height)
    else
        vis = x / self.width
    end
    vis = math.max(0, math.min(1, vis))
    
    local margin = 0.05
    if vis > (self.vis_low + margin) and vis < (self.vis_high - margin) then
        self.drag_target = "both"
        self.drag_start_vis = vis
        self.drag_start_vis_low = self.vis_low
        self.drag_start_vis_high = self.vis_high
    else
        if vis < self.vis_low then
            self.drag_target = "low"
        elseif vis > self.vis_high then
            self.drag_target = "high"
        else
            local dist_low = math.abs(vis - self.vis_low)
            local dist_high = math.abs(vis - self.vis_high)
            if dist_low < dist_high then
                self.drag_target = "low"
            else
                self.drag_target = "high"
            end
        end
        self:set_value_from_mouse(x, y)
    end
    return true
end

function ctgui_rslider:mouse_drag(x, y, button, mod)
    if self.dragging then self:set_value_from_mouse(x, y) end
end

function ctgui_rslider:mouse_up(x, y, button, mod)
    self.dragging = false
    self.drag_target = nil
end

-- Line Logic

function ctgui_rslider:calculate_easing(t, mode, params)
    if mode == "linear" or mode == "line" then return t end
    
    params = params or {}
    
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
        local c1 = params[1] or 1.70158; local c3 = c1 + 1
        return c3 * t * t * t - c1 * t * t
    elseif mode == "back-out" then 
        local c1 = params[1] or 1.70158; local c3 = c1 + 1
        return 1 + c3 * (t - 1)^3 + c1 * (t - 1)^2
    elseif mode == "back-inout" then
        local c1 = params[1] or 1.70158; local c2 = c1 * 1.525
        if t < 0.5 then
            return ((2 * t)^2 * ((c2 + 1) * 2 * t - c2)) / 2
        end
        return ((2 * t - 2)^2 * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2

    -- Elastic
    elseif mode == "elastic-in" then
        if t == 0 then return 0 end
        if t == 1 then return 1 end
        local a = params[1] or 1
        local p = params[2] or 0.3
        local s
        if a < 1 then a = 1; s = p / 4 else s = p / (2 * math.pi) * math.asin(1 / a) end
        return -(a * 2^(10 * (t - 1)) * math.sin((t - 1 - s) * (2 * math.pi) / p))
    elseif mode == "elastic-out" then
        if t == 0 then return 0 end
        if t == 1 then return 1 end
        local a = params[1] or 1
        local p = params[2] or 0.3
        local s
        if a < 1 then a = 1; s = p / 4 else s = p / (2 * math.pi) * math.asin(1 / a) end
        return a * 2^(-10 * t) * math.sin((t - s) * (2 * math.pi) / p) + 1
    elseif mode == "elastic-inout" then
        if t == 0 then return 0 end
        if t == 1 then return 1 end
        local a = params[1] or 1
        local p = params[2] or 0.45
        local s
        if a < 1 then a = 1; s = p / 4 else s = p / (2 * math.pi) * math.asin(1 / a) end
        t = t * 2
        if t < 1 then
            return -0.5 * (a * 2^(10 * (t - 1)) * math.sin((t - 1 - s) * (2 * math.pi) / p))
        end
        return a * 2^(-10 * (t - 1)) * math.sin((t - 1 - s) * (2 * math.pi) / p) * 0.5 + 1
    
    -- Bounce
    elseif mode == "bounce-out" then
        local e = params[1] or 0.5
        -- Clamp elasticity to reasonable bounds
        if e < 0 then e = 0.1 end
        if e >= 1 then e = 0.99 end
        
        local t1 = (1 - e) / (1 + e)
        local k = 1 / (t1 * t1)
        
        if t < t1 then
            return k * t * t
        end
        
        local t_curr = t - t1
        local duration = 2 * e * t1
        local height = e * e
        
        -- Simulate bounces
        for i=1, 50 do
            if t_curr < duration then
                local half = duration / 2
                local x = t_curr - half
                return 1 - height + k * x * x
            end
            
            t_curr = t_curr - duration
            duration = duration * e
            height = height * e * e
            
            if height < 0.000001 then return 1 end
        end
        return 1

    elseif mode == "bounce-in" then return 1 - self:calculate_easing(1 - t, "bounce-out", params)
    elseif mode == "bounce-inout" then
        if t < 0.5 then return (1 - self:calculate_easing(1 - 2 * t, "bounce-out", params)) / 2 end
        return (1 + self:calculate_easing(2 * t - 1, "bounce-out", params)) / 2
    
    -- Hann
    elseif mode == "hann" then return 0.5 * (1 - math.cos(math.pi * t))
    end

    return t
end

function ctgui_rslider:stop_line_low()
    if self.line_running_low then
        self.line_clock_low:unset()
        self.line_running_low = false
    end
end

function ctgui_rslider:stop_line_high()
    if self.line_running_high then
        self.line_clock_high:unset()
        self.line_running_high = false
    end
end

function ctgui_rslider:start_line_low(target, time_ms, easing, params)
    self:stop_line_low()
    target = math.max(self.min_val, math.min(self.max_val, target))
    
    if time_ms <= 0 then
        self.val_low = math.min(target, self.val_high)
        self:update_visual_from_value()
        self:throttled_repaint()
        self:output_value()
        return
    end
    
    local ticks = math.ceil(time_ms / self.line_grain)
    if ticks < 1 then ticks = 1 end
    
    self.line_target_low = target
    self.line_start_val_low = self.val_low
    self.line_total_ticks_low = ticks
    self.line_current_tick_low = 0
    self.line_easing_low = easing or self.default_easing
    self.line_easing_params_low = params or {}
    
    self.line_running_low = true
    self.line_clock_low:delay(self.line_grain)
end

function ctgui_rslider:start_line_high(target, time_ms, easing, params)
    self:stop_line_high()
    target = math.max(self.min_val, math.min(self.max_val, target))
    
    if time_ms <= 0 then
        self.val_high = math.max(target, self.val_low)
        self:update_visual_from_value()
        self:throttled_repaint()
        self:output_value()
        return
    end
    
    local ticks = math.ceil(time_ms / self.line_grain)
    if ticks < 1 then ticks = 1 end
    
    self.line_target_high = target
    self.line_start_val_high = self.val_high
    self.line_total_ticks_high = ticks
    self.line_current_tick_high = 0
    self.line_easing_high = easing or self.default_easing
    self.line_easing_params_high = params or {}
    
    self.line_running_high = true
    self.line_clock_high:delay(self.line_grain)
end

function ctgui_rslider:line_tick_low()
    if not self.line_running_low then return end
    self.line_current_tick_low = self.line_current_tick_low + 1
    local t = self.line_current_tick_low / self.line_total_ticks_low
    if t > 1 then t = 1 end
    
    local eased_t = self:calculate_easing(t, self.line_easing_low, self.line_easing_params_low)
    local new_val = self.line_start_val_low + (self.line_target_low - self.line_start_val_low) * eased_t
    
    self.val_low = math.min(new_val, self.val_high)
    self:update_visual_from_value()
    self:output_value()
    self:throttled_repaint()
    
    if self.line_current_tick_low >= self.line_total_ticks_low then
        self.line_running_low = false
    else
        self.line_clock_low:delay(self.line_grain)
    end
end

function ctgui_rslider:line_tick_high()
    if not self.line_running_high then return end
    self.line_current_tick_high = self.line_current_tick_high + 1
    local t = self.line_current_tick_high / self.line_total_ticks_high
    if t > 1 then t = 1 end
    
    local eased_t = self:calculate_easing(t, self.line_easing_high, self.line_easing_params_high)
    local new_val = self.line_start_val_high + (self.line_target_high - self.line_start_val_high) * eased_t
    
    self.val_high = math.max(new_val, self.val_low)
    self:update_visual_from_value()
    self:output_value()
    self:throttled_repaint()
    
    if self.line_current_tick_high >= self.line_total_ticks_high then
        self.line_running_high = false
    else
        self.line_clock_high:delay(self.line_grain)
    end
end

-- Input Handlers

function ctgui_rslider:parse_line_args(atoms)
    local target = atoms[1]
    local time = (type(atoms[2]) == "number") and atoms[2] or self.default_line_ms
    local easing = (type(atoms[3]) == "string" or type(atoms[3]) == "number") and atoms[3] or nil
    local params = nil
    if #atoms >= 4 then
        params = {}
        for i=4, #atoms do table.insert(params, atoms[i]) end
    end
    return target, time, easing, params
end

-- Route Mode Handlers (Global)
function ctgui_rslider:in_1_lo(atoms)
    if type(atoms) == "table" then
        local target, time, easing, params = self:parse_line_args(atoms)
        self:start_line_low(target, time, easing, params)
    elseif type(atoms) == "number" then
        self:start_line_low(atoms, self.default_line_ms)
    end
end
function ctgui_rslider:in_1_low(atoms) self:in_1_lo(atoms) end

function ctgui_rslider:in_1_hi(atoms)
    if type(atoms) == "table" then
        local target, time, easing, params = self:parse_line_args(atoms)
        self:start_line_high(target, time, easing, params)
    elseif type(atoms) == "number" then
        self:start_line_high(atoms, self.default_line_ms)
    end
end
function ctgui_rslider:in_1_high(atoms) self:in_1_hi(atoms) end

function ctgui_rslider:in_1_lopos(atoms)
    if type(atoms) == "table" then
        local target_pos, time, easing, params = self:parse_line_args(atoms)
        local target_val = self:visual_to_value(target_pos)
        self:start_line_low(target_val, time, easing, params)
    elseif type(atoms) == "number" then
        local target_val = self:visual_to_value(atoms)
        self:start_line_low(target_val, self.default_line_ms)
    end
end
function ctgui_rslider:in_1_lowpos(atoms) self:in_1_lopos(atoms) end

function ctgui_rslider:in_1_hipos(atoms)
    if type(atoms) == "table" then
        local target_pos, time, easing, params = self:parse_line_args(atoms)
        local target_val = self:visual_to_value(target_pos)
        self:start_line_high(target_val, time, easing, params)
    elseif type(atoms) == "number" then
        local target_val = self:visual_to_value(atoms)
        self:start_line_high(target_val, self.default_line_ms)
    end
end
function ctgui_rslider:in_1_highpos(atoms) self:in_1_hipos(atoms) end

function ctgui_rslider:in_1_list(atoms)
    local sel = atoms[1]
    
    -- Global Handlers
    if sel == "lo" or sel == "low" then
        if type(atoms[2]) == "number" then
            local args = {unpack(atoms, 2)}
            local target, time, easing, params = self:parse_line_args(args)
            self:start_line_low(target, time, easing, params)
        end
        return
    elseif sel == "hi" or sel == "high" then
        if type(atoms[2]) == "number" then
            local args = {unpack(atoms, 2)}
            local target, time, easing, params = self:parse_line_args(args)
            self:start_line_high(target, time, easing, params)
        end
        return
    elseif sel == "lopos" or sel == "lowpos" then
        if type(atoms[2]) == "number" then
            local args = {unpack(atoms, 2)}
            local target_pos, time, easing, params = self:parse_line_args(args)
            local target_val = self:visual_to_value(target_pos)
            self:start_line_low(target_val, time, easing, params)
        end
        return
    elseif sel == "hipos" or sel == "highpos" then
        if type(atoms[2]) == "number" then
            local args = {unpack(atoms, 2)}
            local target_pos, time, easing, params = self:parse_line_args(args)
            local target_val = self:visual_to_value(target_pos)
            self:start_line_high(target_val, time, easing, params)
        end
        return
    end

    if not self.route_mode then
        -- Standard Mode: Inlet 1 is Low Value or Pos
        if sel == "pos" then
            if type(atoms[2]) == "number" then
                local args = {unpack(atoms, 2)}
                local target_pos, time, easing, params = self:parse_line_args(args)
                local target_val = self:visual_to_value(target_pos)
                self:start_line_low(target_val, time, easing, params)
            end
        elseif type(sel) == "number" then
            local target, time, easing, params = self:parse_line_args(atoms)
            self:start_line_low(target, time, easing, params)
        end
    end
end

function ctgui_rslider:in_1_pos(atoms)
    if self.route_mode then return end
    if type(atoms) == "table" then
        local target_pos, time, easing, params = self:parse_line_args(atoms)
        local target_val = self:visual_to_value(target_pos)
        self:start_line_low(target_val, time, easing, params)
    elseif type(atoms) == "number" then
        local target_val = self:visual_to_value(atoms)
        self:start_line_low(target_val, self.default_line_ms)
    end
end

function ctgui_rslider:in_1_float(f)
    if self.route_mode then return end
    if self.default_line_ms > 0 then
        self:start_line_low(f, self.default_line_ms)
    else
        self:stop_line_low()
        self.val_low = math.min(f, self.val_high)
        self:update_visual_from_value()
        self:throttled_repaint()
        self:output_value()
    end
end

function ctgui_rslider:in_1_set(f)
    if self.route_mode then return end
    self:stop_line_low()
    self.val_low = math.min(f, self.val_high)
    self:update_visual_from_value()
    self:throttled_repaint()
end

-- Inlet 2: High Value
function ctgui_rslider:in_2_list(atoms)
    if self.route_mode then return end
    if atoms[1] == "pos" then
        if type(atoms[2]) == "number" then
            local args = {unpack(atoms, 2)}
            local target_pos, time, easing, params = self:parse_line_args(args)
            local target_val = self:visual_to_value(target_pos)
            self:start_line_high(target_val, time, easing, params)
        end
    elseif type(atoms[1]) == "number" then
        local target, time, easing, params = self:parse_line_args(atoms)
        self:start_line_high(target, time, easing, params)
    end
end

function ctgui_rslider:in_2_pos(atoms)
    if self.route_mode then return end
    if type(atoms) == "table" then
        local target_pos, time, easing, params = self:parse_line_args(atoms)
        local target_val = self:visual_to_value(target_pos)
        self:start_line_high(target_val, time, easing, params)
    elseif type(atoms) == "number" then
        local target_val = self:visual_to_value(atoms)
        self:start_line_high(target_val, self.default_line_ms)
    end
end

function ctgui_rslider:in_2_float(f)
    if self.route_mode then return end
    if self.default_line_ms > 0 then
        self:start_line_high(f, self.default_line_ms)
    else
        self:stop_line_high()
        self.val_high = math.max(f, self.val_low)
        self:update_visual_from_value()
        self:throttled_repaint()
        self:output_value()
    end
end

function ctgui_rslider:in_2_set(f)
    if self.route_mode then return end
    self:stop_line_high()
    self.val_high = math.max(f, self.val_low)
    self:update_visual_from_value()
    self:throttled_repaint()
end

-- Inlet 3: Low Position (if @pos)
function ctgui_rslider:in_3_float(f)
    if self.route_mode or not self.pos_mode then return end
    local target_val = self:visual_to_value(f)
    if self.default_line_ms > 0 then
        self:start_line_low(target_val, self.default_line_ms)
    else
        self:stop_line_low()
        self.val_low = math.min(target_val, self.val_high)
        self:update_visual_from_value()
        self:throttled_repaint()
        self:output_value()
    end
end

function ctgui_rslider:in_3_list(atoms)
    if self.route_mode or not self.pos_mode then return end
    if type(atoms[1]) == "number" then
        local target_pos, time, easing, params = self:parse_line_args(atoms)
        local target_val = self:visual_to_value(target_pos)
        self:start_line_low(target_val, time, easing, params)
    end
end

function ctgui_rslider:in_3_set(f)
    if self.route_mode or not self.pos_mode then return end
    self:stop_line_low()
    local target_val = self:visual_to_value(f)
    self.val_low = math.min(target_val, self.val_high)
    self:update_visual_from_value()
    self:throttled_repaint()
end

-- Inlet 4: High Position (if @pos)
function ctgui_rslider:in_4_float(f)
    if self.route_mode or not self.pos_mode then return end
    local target_val = self:visual_to_value(f)
    if self.default_line_ms > 0 then
        self:start_line_high(target_val, self.default_line_ms)
    else
        self:stop_line_high()
        self.val_high = math.max(target_val, self.val_low)
        self:update_visual_from_value()
        self:throttled_repaint()
        self:output_value()
    end
end

function ctgui_rslider:in_4_list(atoms)
    if self.route_mode or not self.pos_mode then return end
    if type(atoms[1]) == "number" then
        local target_pos, time, easing, params = self:parse_line_args(atoms)
        local target_val = self:visual_to_value(target_pos)
        self:start_line_high(target_val, time, easing, params)
    end
end

function ctgui_rslider:in_4_set(f)
    if self.route_mode or not self.pos_mode then return end
    self:stop_line_high()
    local target_val = self:visual_to_value(f)
    self.val_high = math.max(target_val, self.val_low)
    self:update_visual_from_value()
    self:throttled_repaint()
end

-- Output

function ctgui_rslider:output_value()
    if self.data_fps > 0 then
        self.pending_out_low = self.val_low
        self.pending_out_high = self.val_high
        self.pending_vis_low = self.vis_low
        self.pending_vis_high = self.vis_high
        self.pending_out_changed = true
        
        self:throttled_data_output()
    else
        if self.route_mode then
            self:outlet(1, "lo", {self.val_low})
            self:outlet(1, "hi", {self.val_high})
            self:outlet(1, "lopos", {self.vis_low})
            self:outlet(1, "hipos", {self.vis_high})
        else
            self:outlet(1, "float", {self.val_low})
            self:outlet(2, "float", {self.val_high})
            if self.pos_mode then
                self:outlet(3, "float", {self.vis_low})
                self:outlet(4, "float", {self.vis_high})
            elseif self.pos_output then
                self:outlet(3, "list", {self.vis_low, self.vis_high})
            end
        end
    end
end

function ctgui_rslider:throttled_data_output()
    if self.data_clock_running then
        self.data_pending = true
        return
    end
    
    if self.pending_out_changed then
        if self.route_mode then
            self:outlet(1, "lo", {self.pending_out_low})
            self:outlet(1, "hi", {self.pending_out_high})
            local vis_l = self.pending_vis_low or self:value_to_visual(self.pending_out_low)
            local vis_h = self.pending_vis_high or self:value_to_visual(self.pending_out_high)
            self:outlet(1, "lopos", {vis_l})
            self:outlet(1, "hipos", {vis_h})
        else
            self:outlet(1, "float", {self.pending_out_low})
            self:outlet(2, "float", {self.pending_out_high})
            if self.pos_mode then
                local vis_l = self.pending_vis_low or self:value_to_visual(self.pending_out_low)
                local vis_h = self.pending_vis_high or self:value_to_visual(self.pending_out_high)
                self:outlet(3, "float", {vis_l})
                self:outlet(4, "float", {vis_h})
            elseif self.pos_output then
                local vis_l = self.pending_vis_low or self:value_to_visual(self.pending_out_low)
                local vis_h = self.pending_vis_high or self:value_to_visual(self.pending_out_high)
                self:outlet(3, "list", {vis_l, vis_h})
            end
        end
        self.pending_out_changed = false
    end
    
    if self.pending_ctrl_changed and not self.route_mode then
        self:outlet(2, "list", {self.pending_ctrl_low, self.pending_ctrl_high})
        self.pending_ctrl_changed = false
    end
    
    self.data_clock_running = true
    self.data_clock:delay(1000 / self.data_fps)
end

function ctgui_rslider:data_tick()
    self.data_clock_running = false
    if self.data_pending then
        self.data_pending = false
        self:throttled_data_output()
    end
end

-- Painting

function ctgui_rslider:throttled_repaint()
    if self.repaint_clock_running then
        self.repaint_pending = true
        return
    end
    self:_raw_repaint()
    self.repaint_clock_running = true
    self.repaint_clock:delay(1000 / self.gui_fps)
end

function ctgui_rslider:repaint_tick()
    self.repaint_clock_running = false
    if self.repaint_pending then
        self.repaint_pending = false
        self:throttled_repaint()
    end
end

function ctgui_rslider:paint(g)
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
        
        if self.orientation == "vertical" then
            local mark_y = self.height - (zero_vis * self.height)
            g:fill_rect(2, mark_y, self.width - 4, 1)
        else
            local mark_x = zero_vis * self.width
            g:fill_rect(mark_x, 2, 1, self.height - 4)
        end
    end

    -- Handle (Range Bar)
    local max_dim = (self.orientation == "vertical") and self.width or self.height
    local handle_size = self:get_handle_size()
    local bar_thickness = handle_size / 1.3
    local radius = handle_size / 2
    
    local v_min = math.min(self.vis_low, self.vis_high)
    local v_max = math.max(self.vis_low, self.vis_high)

    local bar_r, bar_g, bar_b
    if self.c_handle.h then
        bar_r, bar_g, bar_b = Colors.hsb(self.c_handle.h, self.c_handle.s, self.c_handle.b * 0.7)
    else
        bar_r = self.c_handle[1] * 0.7
        bar_g = self.c_handle[2] * 0.7
        bar_b = self.c_handle[3] * 0.7
    end
    
    if self.orientation == "vertical" then
        local track_len = self.height - handle_size
        local y_low = (self.height - handle_size) - (v_min * track_len)
        local y_high = (self.height - handle_size) - (v_max * track_len)
        
        local center_x = self.width / 2
        local bar_x = center_x - (bar_thickness / 2)
        local handle_x = center_x - (handle_size / 2)
        
        g:set_color(bar_r, bar_g, bar_b)
        local bar_top = y_high + (handle_size / 2)
        local bar_bottom = y_low + (handle_size / 2)
        g:fill_rect(bar_x, bar_top, bar_thickness, bar_bottom - bar_top)

        g:set_color(self.c_handle[1], self.c_handle[2], self.c_handle[3])
        g:fill_rounded_rect(handle_x, y_low, handle_size, handle_size, radius)
        g:fill_rounded_rect(handle_x, y_high, handle_size, handle_size, radius)
    else
        local track_len = self.width - handle_size
        local x_low = v_min * track_len
        local x_high = v_max * track_len
        
        local center_y = self.height / 2
        local bar_y = center_y - (bar_thickness / 2)
        local handle_y = center_y - (handle_size / 2)
        
        g:set_color(bar_r, bar_g, bar_b)
        local bar_left = x_low + (handle_size / 2)
        local bar_right = x_high + (handle_size / 2)
        g:fill_rect(bar_left, bar_y, bar_right - bar_left, bar_thickness)

        g:set_color(self.c_handle[1], self.c_handle[2], self.c_handle[3])
        g:fill_rounded_rect(x_low, handle_y, handle_size, handle_size, radius)
        g:fill_rounded_rect(x_high, handle_y, handle_size, handle_size, radius)
    end
end

-- Dynamic Configuration
function ctgui_rslider:in_1_guifps(atoms)
    local f = type(atoms) == "table" and atoms[1] or atoms
    if type(f) == "number" and f > 0 then self.gui_fps = f end
end
function ctgui_rslider:in_1_guishutter(atoms) self:in_1_guifps(atoms) end
function ctgui_rslider:in_1_fps(atoms) self:in_1_guifps(atoms) end
function ctgui_rslider:in_1_shutter(atoms) self:in_1_guifps(atoms) end
function ctgui_rslider:in_1_datafps(atoms)
    local f = type(atoms) == "table" and atoms[1] or atoms
    if type(f) == "number" then self.data_fps = (f > 0) and f or 0 end
end
function ctgui_rslider:in_1_datashutter(atoms) self:in_1_datafps(atoms) end
function ctgui_rslider:in_1_linems(atoms)
    local f = type(atoms) == "table" and atoms[1] or atoms
    if type(f) == "number" then self.default_line_ms = math.max(0, f) end
end
function ctgui_rslider:in_1_linegrain(atoms)
    local f = type(atoms) == "table" and atoms[1] or atoms
    if type(f) == "number" then self.line_grain = math.max(1, f) end
end
function ctgui_rslider:in_1_reasing(atoms)
    local v = type(atoms) == "table" and atoms[1] or atoms
    if type(v) == "number" or type(v) == "string" then self.default_easing = v end
end
function ctgui_rslider:in_1_curve(atoms) self:in_1_reasing(atoms) end
function ctgui_rslider:in_1_easing(atoms) self:in_1_reasing(atoms) end

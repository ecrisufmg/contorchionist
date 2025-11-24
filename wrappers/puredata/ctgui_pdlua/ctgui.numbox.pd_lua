-- ctgui.numbox.pd_lua - Numeric Box with Drag and Input
--
-- Usage:
--   [ctgui.numbox @min 0 @max 100]
--
-- Arguments:
--   @min <val>      : Minimum value (default: -1e9)
--   @max <val>      : Maximum value (default: 1e9)
--   @width <val>    : Width (default: 40)
--   @height <val>   : Height (default: 20)
--   @color <r g b>  : Text color
--   @bgcolor <r g b>: Background color
--   @bordercolor <r g b>: Border color
--   @fontsize <val> : Font size (default: 12)
--   @dark           : Dark mode theme
--   @guifps <val>   : GUI refresh rate
--   @datafps <val>  : Data output rate limit

local ArgParser = require("pd_arg_parser")
local Colors = require("colors")

local ctgui_numbox = pd.Class:new():register("ctgui.numbox")

-- Default Colors (HSB)
local C_BG_LIGHT = {0, 0, 0.93}   -- Slightly lighter/different grey
local C_BG_DARK = {0, 0, 0.35}
local C_TEXT_LIGHT = {0, 0, 0.15}
local C_TEXT_DARK = {0, 0, 0.9}
local C_BORDER_LIGHT = {0, 0, 0.5} -- Mid Grey
local C_BORDER_DARK = {0, 0, 0.6}
local C_ACTIVE_BORDER = {0.6, 0.8, 0.9} -- Highlight when active

function ctgui_numbox:initialize(sel, atoms)
    self.creation_args = atoms
    local parser = ArgParser:new(atoms)

    -- Dimensions
    self.width = parser:get_float("width w", 40)
    self.height = parser:get_float("height h", 20)
    self.fontsize = parser:get_float("fontsize size", 12)

    -- Range
    self.min_val = parser:get_float("min minimum", -1000000000)
    self.max_val = parser:get_float("max maximum", 1000000000)

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

    -- Dark mode
    local is_dark = parser:get_bool("dark")

    self.c_bg = get_color_flexible({"bgcolor", "bg"}, is_dark and C_BG_DARK or C_BG_LIGHT)
    self.c_text = get_color_flexible({"color", "textcolor", "fg"}, is_dark and C_TEXT_DARK or C_TEXT_LIGHT)
    self.c_border = get_color_flexible({"bordercolor", "border"}, is_dark and C_BORDER_DARK or C_BORDER_LIGHT)
    
    -- Highlight color for active editing
    local r, g, b = Colors.hsb(C_ACTIVE_BORDER[1], C_ACTIVE_BORDER[2], C_ACTIVE_BORDER[3])
    self.c_active = {r, g, b}

    -- FPS
    local gui_fps = parser:get_float("guifps guishutter fps shutter", 20)
    self.gui_fps = (gui_fps > 0) and gui_fps or 20

    local data_fps = parser:get_float("datafps datashutter", 0)
    self.data_fps = (data_fps > 0) and data_fps or 0

    -- State
    self.current_value = 0
    -- If a positional argument is given, set it as default value
    if parser:get_positional_count() > 0 then
        self.current_value = parser:get_positional_float(1, 0)
    end
    -- Clamp initial value
    self.current_value = math.max(self.min_val, math.min(self.max_val, self.current_value))

    self.dragging = false
    self.last_drag_y = 0
    self.last_drag_x = 0
    self.typing = false
    self.input_buffer = ""
    self.shift_down = false

    -- Clocks
    self.repaint_clock = pd.Clock:new():register(self, "repaint_tick")
    self.repaint_clock_running = false
    self.repaint_pending = false

    self.data_clock = pd.Clock:new():register(self, "data_tick")
    self.data_clock_running = false
    self.data_pending = false
    self.pending_out_value = nil
    self.pending_out_changed = false

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
    self.inlets = 2
    self.outlets = 1

    -- Override repaint for throttling
    self._raw_repaint = self.repaint
    self.repaint = self.throttled_repaint

    -- Global Key Receiver
    self.key_receiver = pd.Receive:new():register(self, "luakeys", "receive_key")

    -- Global Focus Receiver
    self.focus_receiver = pd.Receive:new():register(self, "ctgui_focus", "receive_focus")
    self.id = tostring(self)

    return true
end

function ctgui_numbox:postinitialize()
    self:set_size(self.width, self.height)
end

function ctgui_numbox:finalize()
    if self.key_receiver then self.key_receiver:destruct() end
    if self.focus_receiver then self.focus_receiver:destruct() end
end

function ctgui_numbox:get_zero_visual()
    return 0 -- Not used in numbox but kept for consistency if needed
end

-- Line Logic

function ctgui_numbox:stop_line()
    if self.line_running then
        self.line_clock:unset()
        self.line_running = false
    end
end

function ctgui_numbox:calculate_easing(t, mode, params)
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

function ctgui_numbox:start_line(target, time_ms, easing, params)
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
    self.line_easing_params = params or {}
    
    self.line_running = true
    self.line_clock:delay(self.line_grain)
end

function ctgui_numbox:line_tick()
    if not self.line_running then return end
    
    self.line_current_tick = self.line_current_tick + 1
    local t = self.line_current_tick / self.line_total_ticks
    if t > 1 then t = 1 end
    
    -- Easing
    local eased_t = self:calculate_easing(t, self.line_easing, self.line_easing_params)
    
    self.current_value = self.line_start_val + (self.line_target - self.line_start_val) * eased_t
    
    -- Check bounds/completion
    local finished = false
    if self.line_current_tick >= self.line_total_ticks then
        self.current_value = self.line_target
        finished = true
    end
    
    self.current_value = math.max(self.min_val, math.min(self.max_val, self.current_value))
    
    if self.typing then
        self.input_buffer = string.format("%g", self.current_value)
    end
    
    self:output_value()
    self:throttled_repaint()
    
    if finished then
        self.line_running = false
    else
        self.line_clock:delay(self.line_grain)
    end
end

function ctgui_numbox:in_1_getcode()
    local str = "ctgui.numbox"
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

-- Interaction (Same as before)

function ctgui_numbox:get_char_width(char)
    if char == "." then
        return self.fontsize * 0.28 -- Narrower dot
    end
    return self.fontsize * 0.6 -- Standard digit width
end

function ctgui_numbox:get_cursor_info(x)
    local val = self.current_value
    -- Base string for calculation (ensure it has a dot)
    local base_s = string.format("%.5f", val):gsub("0+$", ""):gsub("%.$", "")
    if not base_s:find("%.") then base_s = base_s .. "." end
    
    local dot_pos = base_s:find("%.")
    local start_x = 4
    local rel_x = x - start_x
    
    local current_x = 0
    local found_idx = nil
    local rect_x = 0
    local rect_w = 0
    
    -- Check inside base string
    for i = 1, #base_s do
        local char = base_s:sub(i, i)
        local w = self:get_char_width(char)
        if rel_x < current_x + w then
            found_idx = i
            rect_x = start_x + current_x
            rect_w = w
            break
        end
        current_x = current_x + w
    end
    
    -- If not found, it's beyond the string (ghost digits)
    if not found_idx then
        local i = #base_s
        while not found_idx do
            i = i + 1
            local char = "0" -- Ghost digits are 0
            local w = self:get_char_width(char)
            if rel_x < current_x + w then
                found_idx = i
                rect_x = start_x + current_x
                rect_w = w
                break
            end
            current_x = current_x + w
            -- Safety break
            if i > #base_s + 20 then 
                 found_idx = i
                 rect_x = start_x + current_x
                 rect_w = w
            end
        end
    end
    
    local idx = found_idx
    local display_s = base_s
    
    -- Extend string if cursor is past the end
    if idx > #base_s then
        local target_decimals = idx - dot_pos
        if target_decimals < 0 then target_decimals = 0 end
        display_s = string.format("%." .. target_decimals .. "f", val)
    end
    
    -- Recalculate power
    local power = 0
    if idx < dot_pos then
        power = (dot_pos - 1) - idx
    elseif idx == dot_pos then
        power = 0 -- Cursor on dot
    else
        power = dot_pos - idx
    end
    
    return {
        power = power,
        rect_x = rect_x,
        rect_w = rect_w,
        display_s = display_s
    }
end

function ctgui_numbox:mouse_down(x, y)
    -- pd-lua graphics callbacks only provide x, y. button and mod are nil.
    self:stop_line()
    self.dragging = false
    self.last_drag_y = y
    self.last_drag_x = x
    if self.typing then
        self:confirm_typing()
    end
    self.potential_drag = true
    return true
end

function ctgui_numbox:mouse_drag(x, y)
    if self.potential_drag then
        local dy = self.last_drag_y - y
        if math.abs(dy) > 1 then -- Reduced threshold
            self.dragging = true
            self.potential_drag = false
        end
    end

    if self.dragging then
        local dy = self.last_drag_y - y
        local scale = 1.0
        
        if self.shift_down then
            local info = self:get_cursor_info(x)
            scale = 10 ^ info.power
            -- Slower interaction for fine tuning? 
            -- Usually 1px = 1 unit is too fast for high powers, but okay for decimals.
            -- Let's dampen it slightly for usability
            -- scale = scale * 0.5 
        end

        local delta_val = (dy * scale)
        local new_val = self.current_value + delta_val
        self.current_value = math.max(self.min_val, math.min(self.max_val, new_val))
        
        self.last_drag_y = y
        self.last_drag_x = x
        
        self:output_value()
        self:throttled_repaint()
    end
end

function ctgui_numbox:mouse_up(x, y)
    if self.potential_drag and not self.dragging then
        self:start_typing()
    end
    self.dragging = false
    self.potential_drag = false
end

function ctgui_numbox:start_typing()
    pd.send("ctgui_focus", "claim", {self.id})
    self.typing = true
    self.input_buffer = string.format("%g", self.current_value)
    self.replace_on_type = true -- Enable overwrite
    self:throttled_repaint()
    -- pd.post("ctgui.numbox: Input mode active.")
end

function ctgui_numbox:receive_focus(sel, atoms)
    if sel == "claim" then
        local sender_id = atoms[1]
        if sender_id ~= self.id then
            if self.typing then
                self:confirm_typing()
            end
        end
    end
end

function ctgui_numbox:confirm_typing()
    self.typing = false
    local val = tonumber(self.input_buffer)
    if val then
        self.current_value = math.max(self.min_val, math.min(self.max_val, val))
        self:output_value()
    end
    self:throttled_repaint()
end

function ctgui_numbox:cancel_typing()
    self.typing = false
    self:throttled_repaint()
end

function ctgui_numbox:receive_key(sel, atoms)
    if sel == "list" and #atoms >= 2 then
        local state = atoms[1]
        local keyname = tostring(atoms[2])
        if type(state) == "number" then
            self:handle_key_name(state, keyname)
        end
    end
end

function ctgui_numbox:handle_key_name(state, keyname)
    -- Track modifiers
    if keyname == "Shift_L" or keyname == "Shift_R" then
        self.shift_down = (state ~= 0)
        return
    end

    -- Only process Key Down (state > 0) for typing
    if state == 0 then return end

    if not self.typing then return end

    -- Overwrite logic
    if self.replace_on_type then
        local is_typing_key = (keyname:match("^%d$") or keyname == "period" or keyname == "." or keyname == "minus" or keyname == "-" or keyname == "e" or keyname == "E" or keyname == "BackSpace")
        if is_typing_key then
            self.input_buffer = ""
            self.replace_on_type = false
        end
    end

    if keyname == "Return" or keyname == "Enter" then
        self:confirm_typing()
    elseif keyname == "Escape" then
        self:cancel_typing()
    elseif keyname == "BackSpace" then
        if #self.input_buffer > 0 then
            self.input_buffer = self.input_buffer:sub(1, -2)
            self:throttled_repaint()
        end
    else
        -- Character mapping
        local char = nil
        if keyname:match("^%d$") then -- 0-9
            char = keyname
        elseif keyname == "period" or keyname == "." then
            char = "."
        elseif keyname == "minus" or keyname == "-" then
            char = "-"
        elseif keyname == "e" or keyname == "E" then
            char = self.shift_down and "E" or "e"
        end

        if char then
             self.input_buffer = self.input_buffer .. char
             self:throttled_repaint()
        end
    end
end

function ctgui_numbox:in_1_key(keycode)
    if not self.typing then return end
    if keycode == 13 or keycode == 10 then -- Enter
        self:confirm_typing()
    elseif keycode == 27 then -- Esc
        self:cancel_typing()
    elseif keycode == 8 or keycode == 127 then -- Backspace
        if #self.input_buffer > 0 then
            self.input_buffer = self.input_buffer:sub(1, -2)
            self:throttled_repaint()
        end
    elseif keycode >= 32 and keycode <= 126 then -- Printable ASCII
        local char = string.char(keycode)
        if char:match("[%d%.%-eE]") then
            self.input_buffer = self.input_buffer .. char
            self:throttled_repaint()
        end
    end
end

function ctgui_numbox:in_1_list(atoms)
    if type(atoms) == "table" and #atoms > 0 and type(atoms[1]) == "number" then
        if #atoms >= 2 and type(atoms[2]) == "number" and atoms[2] > 0 then
            local easing = nil
            local params = nil
            if #atoms >= 3 then easing = atoms[3] end
            if #atoms >= 4 then
                params = {}
                for i=4, #atoms do table.insert(params, atoms[i]) end
            end
            self:start_line(atoms[1], atoms[2], easing, params)
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

function ctgui_numbox:in_1_float(f)
    if self.default_line_ms > 0 then
        self:start_line(f, self.default_line_ms)
        return
    end

    self:stop_line()
    self.current_value = math.max(self.min_val, math.min(self.max_val, f))
    if self.typing then
        self.input_buffer = string.format("%g", self.current_value)
    end
    self:output_value()
    self:throttled_repaint()
end

function ctgui_numbox:in_1_set(f)
    self:stop_line()
    self.current_value = math.max(self.min_val, math.min(self.max_val, f))
    if self.typing then
        self.input_buffer = string.format("%g", self.current_value)
    end
    self:throttled_repaint()
end

function ctgui_numbox:in_1_linems(atoms)
    local f = type(atoms) == "table" and atoms[1] or atoms
    if type(f) == "number" then
        self.default_line_ms = math.max(0, f)
    end
end

function ctgui_numbox:in_1_linegrain(atoms)
    local f = type(atoms) == "table" and atoms[1] or atoms
    if type(f) == "number" then
        self.line_grain = math.max(1, f)
    end
end

function ctgui_numbox:in_1_reasing(atoms)
    local v = type(atoms) == "table" and atoms[1] or atoms
    if type(v) == "number" or type(v) == "string" then
        self.default_easing = v
    end
end

function ctgui_numbox:in_1_curve(atoms)
    self:in_1_reasing(atoms)
end

function ctgui_numbox:in_1_easing(atoms)
    self:in_1_reasing(atoms)
end

function ctgui_numbox:in_2_float(f)
    self.current_value = math.max(self.min_val, math.min(self.max_val, f))
    if self.typing then
        self.input_buffer = string.format("%g", self.current_value)
    end
    self:throttled_repaint()
end

function ctgui_numbox:output_value()
    if self.data_fps > 0 then
        self.pending_out_value = self.current_value
        self.pending_out_changed = true
        self:throttled_data_output()
    else
        self:outlet(1, "float", {self.current_value})
    end
end

function ctgui_numbox:throttled_data_output()
    if self.data_clock_running then
        self.data_pending = true
        return
    end
    if self.pending_out_changed then
        self:outlet(1, "float", {self.pending_out_value})
        self.pending_out_changed = false
    end
    self.data_clock_running = true
    self.data_clock:delay(1000 / self.data_fps)
end

function ctgui_numbox:data_tick()
    self.data_clock_running = false
    if self.data_pending then
        self.data_pending = false
        self:throttled_data_output()
    end
end

function ctgui_numbox:in_1_guifps(atoms)
    local f = type(atoms) == "table" and atoms[1] or atoms
    if type(f) == "number" and f > 0 then
        self.gui_fps = f
    end
end

function ctgui_numbox:in_1_datafps(atoms)
    local f = type(atoms) == "table" and atoms[1] or atoms
    if type(f) == "number" then
        self.data_fps = (f > 0) and f or 0
    end
end

function ctgui_numbox:throttled_repaint()
    if self.repaint_clock_running then
        self.repaint_pending = true
        return
    end
    self:_raw_repaint()
    self.repaint_clock_running = true
    self.repaint_clock:delay(1000 / self.gui_fps)
end

function ctgui_numbox:repaint_tick()
    self.repaint_clock_running = false
    if self.repaint_pending then
        self.repaint_pending = false
        self:throttled_repaint()
    end
end

-- Painting

function ctgui_numbox:paint(g)
    -- Safety check for colors
    if not self.c_bg then self.c_bg = {237, 237, 237} end
    if not self.c_text then self.c_text = {0, 0, 0} end
    if not self.c_border then self.c_border = {100, 100, 100} end
    if not self.c_active then self.c_active = {255, 0, 0} end

    -- Draw Rounded Box (Border)
    local border_color = self.typing and self.c_active or self.c_border
    g:set_color(border_color[1], border_color[2], border_color[3])
    g:fill_rounded_rect(0, 0, self.width+2, self.height+2, 1)

    -- Draw Background (Inside, slightly smaller to show border)
    g:set_color(self.c_bg[1], self.c_bg[2], self.c_bg[3])
    g:fill_rounded_rect(1, 1, self.width , self.height , 1)

    -- Draw Text
    g:set_color(self.c_text[1], self.c_text[2], self.c_text[3])

    -- Use the 5th argument for font size!
    local font_size = self.fontsize
    local text_x = 4
    local text_y = (self.height - font_size) / 2

    local text_to_draw
    if self.typing then
        text_to_draw = self.input_buffer .. "_"
    elseif self.dragging and self.shift_down then
        local info = self:get_cursor_info(self.last_drag_x)
        text_to_draw = info.display_s
        
        -- Draw highlight box for active digit
        g:set_color(self.c_active[1], self.c_active[2], self.c_active[3]) -- Active color
        g:fill_rect(info.rect_x, text_y, info.rect_w, font_size)
        
        -- Reset text color
        g:set_color(self.c_text[1], self.c_text[2], self.c_text[3])
    else
        local v = self.current_value
        if math.abs(v) < 0.0001 and v ~= 0 then
            text_to_draw = string.format("%.2e", v)
        else
            text_to_draw = string.format("%.3f", v)
            if text_to_draw:find("%.") then
                text_to_draw = text_to_draw:gsub("0+$", ""):gsub("%.$", "")
            end
        end
    end

    -- Draw char by char (Artificial Monospace with variable width dot)
    local current_x = text_x
    for i = 1, #text_to_draw do
        local char = text_to_draw:sub(i, i)
        local w = self:get_char_width(char)
        g:draw_text(char, current_x, text_y, 100, font_size) 
        current_x = current_x + w
    end
end

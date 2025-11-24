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
local C_BG_LIGHT = {0, 0, 1}
local C_BG_DARK = {0, 0, 0.2}
local C_TEXT_LIGHT = {0, 0, 0}
local C_TEXT_DARK = {0, 0, 1}
local C_BORDER_LIGHT = {0, 0, 0.6}
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
    self.drag_start_y = 0
    self.drag_start_val = 0
    self.typing = false
    self.input_buffer = ""

    -- Clocks
    self.repaint_clock = pd.Clock:new():register(self, "repaint_tick")
    self.repaint_clock_running = false
    self.repaint_pending = false

    self.data_clock = pd.Clock:new():register(self, "data_tick")
    self.data_clock_running = false
    self.data_pending = false
    self.pending_out_value = nil
    self.pending_out_changed = false

    -- Controller Input/Output
    -- Normalizing isn't as straightforward for unbound numbox,
    -- but we keep the standard slots.
    self.inlets = 2
    self.outlets = 1 -- Only one outlet for numbox usually (value)

    -- Override repaint for throttling
    self._raw_repaint = self.repaint
    self.repaint = self.throttled_repaint

    return true
end

function ctgui_numbox:postinitialize()
    self:set_size(self.width, self.height)
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

-- Interaction

function ctgui_numbox:mouse_down(x, y, button, mod)
    if button == 1 then -- Left click
        self.dragging = false
        self.drag_start_y = y
        self.drag_start_val = self.current_value
        self.drag_accumulated_delta = 0
        -- If we were typing, confirm and exit typing
        if self.typing then
            self:confirm_typing()
        end
        -- Start tracking for potential drag
        self.potential_drag = true
        return true
    end
end

function ctgui_numbox:mouse_drag(x, y, button, mod)
    if self.potential_drag then
        local dy = self.drag_start_y - y
        if math.abs(dy) > 2 then
            self.dragging = true
            self.potential_drag = false
        end
    end

    if self.dragging then
        local dy = self.drag_start_y - y

        -- Scale factor
        local scale = 1.0
        local shift = (mod == 1) -- Shift key (check specific mod flag in pure data lua docs if 1 is shift)
        -- Actually, mod is a bitmask usually.
        -- In standard Pd, shift=1, ctrl=2, alt=4?
        -- Let's assume mod 1 includes shift.

        if shift then
            scale = 0.01
        else
            scale = 1.0
        end

        -- Sensitivity: 1 unit per 5 pixels?
        local delta_val = (dy * scale)

        local new_val = self.drag_start_val + delta_val

        -- Snap to grid if not shift? No, behavior should be continuous usually.
        -- But if it's integer mode?
        -- Let's keep it float but maybe round for display.

        self.current_value = math.max(self.min_val, math.min(self.max_val, new_val))
        self:output_value()
        self:throttled_repaint()
    end
end

function ctgui_numbox:mouse_up(x, y, button, mod)
    if self.potential_drag and not self.dragging then
        -- It was a click
        self:start_typing()
    end
    self.dragging = false
    self.potential_drag = false
end

-- Typing Logic

function ctgui_numbox:start_typing()
    self.typing = true
    self.input_buffer = string.format("%g", self.current_value) -- Start with current value
    self:throttled_repaint()
    -- Note: Since we cannot grab keyboard focus globally easily,
    -- we rely on the user to route keys if they want,
    -- OR we assume the environment might support it.
    -- However, we provide a visual cue.
    pd.post("ctgui.numbox: Input mode active. Use 'key' messages to inlet 1 if direct typing isn't supported.")
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

-- Key Input Handling (via message 'key' or 'char' into inlet 1)
-- e.g. [key] -> [pd-lua object]
-- Standard Pd [key] outputs floats (keycode). [keyname] outputs symbol keynames.

function ctgui_numbox:in_1_key(keycode)
    if not self.typing then return end

    -- Handle keys based on ASCII/Keycode
    -- 13 = Enter, 27 = Esc, 8 = Backspace, 127 = Delete
    if keycode == 13 then -- Enter
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
        -- Allow digits, ., -, and maybe e (scientific)
        if char:match("[%d%.%-eE]") then
            self.input_buffer = self.input_buffer .. char
            self:throttled_repaint()
        end
    end
end

-- Input

function ctgui_numbox:in_1_float(f)
    self.current_value = math.max(self.min_val, math.min(self.max_val, f))
    if self.typing then
        self.input_buffer = string.format("%g", self.current_value)
    end
    self:output_value() -- Usually numbox outputs when set by float?
    -- Standard Pd number box:
    -- Inlet 1 (hot): sets and outputs.
    -- Inlet 2 (cold): sets only.
    -- Here we have inlet 1.
    self:throttled_repaint()
end

function ctgui_numbox:in_1_set(f)
    self.current_value = math.max(self.min_val, math.min(self.max_val, f))
    if self.typing then
        self.input_buffer = string.format("%g", self.current_value)
    end
    self:throttled_repaint()
end

-- Handling Cold Inlet (Standard Pd style)
-- Since pd-lua dispatch uses in_N_type, we need to check N.
-- We defined self.inlets = 2.
-- Inlet 2 should be cold (set without output).

function ctgui_numbox:in_2_float(f)
    self.current_value = math.max(self.min_val, math.min(self.max_val, f))
    if self.typing then
        self.input_buffer = string.format("%g", self.current_value)
    end
    self:throttled_repaint()
end


-- Output Throttling

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

-- Dynamic Configuration (FPS, etc)

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

-- Painting

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

function ctgui_numbox:paint(g)
    -- Safety check for colors
    if not self.c_bg then self.c_bg = {237, 237, 237} end
    if not self.c_text then self.c_text = {0, 0, 0} end
    if not self.c_border then self.c_border = {100, 100, 100} end
    if not self.c_active then self.c_active = {255, 0, 0} end

    -- Draw Background
    g:set_color(self.c_bg[1], self.c_bg[2], self.c_bg[3])
    g:fill_rect(0, 0, self.width, self.height)

    -- Draw Border
    local border_color = self.typing and self.c_active or self.c_border
    g:set_color(border_color[1], border_color[2], border_color[3])
    g:stroke_rect(0, 0, self.width, self.height, 1)

    -- Triangle / notch to indicate it's a numbox (optional, mimicking Pd)
    -- g:fill_polygon... (skipping for cleaner look unless requested)

    -- Draw Text
    g:set_color(self.c_text[1], self.c_text[2], self.c_text[3])
    g:set_font_size(self.fontsize)

    local text_to_draw
    if self.typing then
        text_to_draw = self.input_buffer .. "_"
    else
        -- Format number (max 4 decimals?)
        -- text_to_draw = string.format("%.4g", self.current_value)
        -- Use more robust formatting
        local v = self.current_value
        if math.abs(v) < 0.0001 and v ~= 0 then
            text_to_draw = string.format("%.2e", v)
        else
            text_to_draw = string.format("%.3f", v)
            -- Trim trailing zeros
            if text_to_draw:find("%.") then
                text_to_draw = text_to_draw:gsub("0+$", ""):gsub("%.$", "")
            end
        end
    end

    -- Center text (simple approximation, pd-lua doesn't give text metrics easily?)
    -- Actually g:draw_text centers? No.
    -- We can guess width roughly.

    local char_w = self.fontsize * 0.6
    local text_w = #text_to_draw * char_w
    local x = (self.width - text_w) / 2
    local y = (self.height - self.fontsize) / 2 + 2 -- slight offset

    g:draw_text(text_to_draw, x, y)
end

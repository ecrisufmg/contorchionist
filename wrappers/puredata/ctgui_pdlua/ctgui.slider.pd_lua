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

-- Default Colors
local C_BG_LIGHT = {0.93, 0.93, 0.93}
local C_BG_DARK = {0.2, 0.2, 0.2}
local C_SLOT_LIGHT = {0.8, 0.8, 0.8}
local C_SLOT_DARK = {0.1, 0.1, 0.1}
local C_HANDLE_LIGHT = {0.6, 0.6, 0.6}
local C_HANDLE_DARK = {0.5, 0.5, 0.5}

function ctgui_slider:initialize(sel, atoms)
    local parser = ArgParser:new(atoms)

    -- Dimensions
    self.width = parser:get_float("width w", 20)
    self.height = parser:get_float("height h", 120)
    
    -- Orientation (inferred from dimensions)
    self.orientation = (self.width > self.height) and "horizontal" or "vertical"

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
    local function get_color(name, default_rgb)
        local val = parser:get_float_list(name .. "color")
        if #val >= 3 then
            return {val[1]/255, val[2]/255, val[3]/255} -- Normalize if 0-255 passed? No, usually 0-1 in pd-lua context or 0-255? 
            -- Wait, ctgui.vu uses Colors.hsb which returns 0-255. 
            -- Let's stick to 0-255 for internal storage if that's what paint uses.
            -- Actually, pd.Class:paint uses 0-255 for set_color? No, usually standard is whatever the lib uses.
            -- In ctgui.vu, Colors.hsb returns 0-255. g:set_color takes 0-255?
            -- Let's check ctgui.vu again. It uses g:set_color(r,g,b).
        end
        -- Check for hex or other formats? For now simple list.
        -- Let's use the helper from ctgui.vu if possible, but I'll simplify.
        -- ArgParser returns what is passed.
        -- If user passes @color 255 0 0, it's 255.
        return nil
    end

    -- Dark mode
    local is_dark = parser:get_bool("dark")
    
    -- Helper to parse color with defaults
    local function parse_color(flag_names, default_rgb_01) 
        local r, g, b = default_rgb_01[1], default_rgb_01[2], default_rgb_01[3]
        
        -- Check flags
        local val = parser:get_float_list(flag_names)
        if val and #val >= 3 then
            r, g, b = val[1], val[2], val[3]
        end
        
        -- Normalize to 0-255 if all values are <= 1.0
        -- This allows users to pass "1 0 0" for red (0-1) or "255 0 0" (0-255)
        if r <= 1.0 and g <= 1.0 and b <= 1.0 then
            r, g, b = r*255, g*255, b*255
        end
        
        return {math.max(0, math.min(255, math.floor(r))), 
                math.max(0, math.min(255, math.floor(g))), 
                math.max(0, math.min(255, math.floor(b)))}
    end

    self.c_bg = parse_color("bgcolor bg", is_dark and C_BG_DARK or C_BG_LIGHT)
    self.c_slot = parse_color("slotcolor trackcolor", is_dark and C_SLOT_DARK or C_SLOT_LIGHT)
    self.c_handle = parse_color("color handlecolor", is_dark and C_HANDLE_DARK or C_HANDLE_LIGHT)

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

    -- Inlets/Outlets
    self.inlets = 1
    self.outlets = 1

    -- Initial visual pos
    self:update_visual_from_value()

    -- Override repaint for throttling
    self._raw_repaint = self.repaint
    self.repaint = self.throttled_repaint

    return true
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
    -- In VU this is calculated or fixed at 0.85. 
    -- Here we should probably calculate it based on min/max if 0 is inside range.
    -- If max is 12 and min is -120, 0 is near top.
    -- Let's use the same logic: 0dB is the reference point.
    
    -- If 0 is not in range, this logic might be weird, but for a fader usually it is.
    -- Let's assume standard audio fader range.
    
    local zero_visual = 0.85 -- Default "unity gain" position
    
    -- If max < 0, then 0 is off scale top.
    -- If min > 0, then 0 is off scale bottom.
    
    -- Let's stick to the VU curve logic exactly for consistency.
    
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
    local zero_visual = 0.85
    
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

-- Input

function ctgui_slider:in_1_float(f)
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
    self.current_value = math.max(self.min_val, math.min(self.max_val, f))
    self:update_visual_from_value()
    self:throttled_repaint()
end

-- Output Throttling

function ctgui_slider:output_value()
    if self.data_fps > 0 then
        self.pending_out_value = self.current_value
        self.pending_out_changed = true
        self:throttled_data_output()
    else
        self:outlet(1, "float", {self.current_value})
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
    
    self.data_clock_running = true
    self.data_clock:delay(1000 / self.data_fps)
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

    -- Handle
    g:set_color(self.c_handle[1], self.c_handle[2], self.c_handle[3])
    
    local handle_thickness = 10 -- Pixel size along the axis
    
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

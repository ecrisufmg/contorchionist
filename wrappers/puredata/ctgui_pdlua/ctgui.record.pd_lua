local ArgParser = require("pd_arg_parser")
local Colors = require("colors")

local thisobj = pd.Class:new():register("ctgui.record")

-- Default Colors (HSB 0-1 scale)
local C_REC = {0, 0.6, 0.8} -- Red
local C_PAUSE = {0.166, 0.7, 0.8} -- Yellow
local C_STOP = {0.04, 0.6, 0.8} -- Red
local C_POS = {0.6, 0.6, 0.8} -- Blue
local C_POSBACK = {0, 0, 0.7} -- Grey
local C_INACTIVE = {0, 0, 0.4} -- Dark Grey
local C_BACKGROUND = {0, 0, 0.93} -- Light Grey

function thisobj:initialize(sel, atoms)
    self.inlets = 2
    self.outlets = 1 -- Outlet 1: commands and position

    -- Store creation args for reconstruction
    self.creation_args = atoms

    -- Parse arguments
    local parser = ArgParser:new(atoms)

    -- Default dimensions
    local default_w = 240
    local default_h = 20

    -- Get dimensions from flags (@w, @h) or positional args (1, 2)
    self.width = parser:get_float("width w", parser:get_positional_float(1, default_w))
    self.height = parser:get_float("height h", parser:get_positional_float(2, default_h))
    
    -- Get duration
    self.duration = parser:get_float("duration d", 0.0)

    -- FPS / Shutter (GUI)
    local gui_fps = parser:get_float("guifps", parser:get_float("guishutter", parser:get_float("fps", parser:get_float("shutter", 60))))
    self.gui_fps = (gui_fps > 0) and gui_fps or 60
    
    self.repaint_clock = pd.Clock:new():register(self, "repaint_tick")
    self.repaint_clock_running = false
    self.repaint_pending = false

    -- Override repaint for throttling
    self._raw_repaint = self.repaint
    self.repaint = self.throttled_repaint

    -- Data FPS / Shutter
    local data_fps = parser:get_float("datafps", parser:get_float("datashutter", -1))
    self.data_fps = data_fps

    self.output_clock = pd.Clock:new():register(self, "output_tick")
    self.output_clock_running = false
    self.output_pending = false

    -- Override output_position for throttling
    self._raw_output_position = self.output_position
    self.output_position = self.throttled_output_position

    -- Parse Colors
    
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
        -- Check for @namecolor (e.g. @reccolor)
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
            -- Check for format: {r, g, b} (implicit RGB)
            elseif type(val[1]) == "number" and #val >= 3 then
                h, s, b = rgb_to_hsb(val[1], val[2], val[3])
                source = "rgb"
            end
        end

        -- Legacy/Alternative: Check for @name_hsb
        local hsb_flag = parser:get_float_list(name .. "_hsb")
        if #hsb_flag == 3 then
            h, s, b = hsb_flag[1], hsb_flag[2], hsb_flag[3]
            source = "hsb"
        end
        
        -- Convert to RGB (0-255) for drawing
        local r, g, b_val = Colors.hsb(h, s, b)
        return {r, g, b_val, source=source, h=h, s=s, b=b}
    end

    -- Dark Mode Defaults
    local def_rec = C_REC
    local def_pause = C_PAUSE
    local def_stop = C_STOP
    local def_bg = C_BACKGROUND

    if parser:get_bool("dark") then
        def_rec = {0, 0.65, 0.98}
        def_pause = {0.15, 0.65, 0.98}
        def_stop = {0.04, 0.65, 0.98}
        def_bg = {0, 0, 0.35}
    end

    self.c_rec = get_color("rec", def_rec)
    self.c_pause = get_color("pause", def_pause)
    self.c_stop = get_color("stop", def_stop)
    self.c_pos = get_color("pos", C_POS)
    self.c_posback = get_color("posback", C_POSBACK)
    
    -- Inactive color (manual conversion)
    local ir, ig, ib = Colors.hsb(C_INACTIVE[1], C_INACTIVE[2], C_INACTIVE[3])
    self.c_inactive = {ir, ig, ib, h=C_INACTIVE[1], s=C_INACTIVE[2], b=C_INACTIVE[3]}
    
    self.c_background = get_color("bg", def_bg)

    -- State
    self.state = "stop" -- "record", "pause", "stop"
    self.position = 0.0 -- 0.0 to 1.0
    self.duration = 0.0 -- Total duration in seconds
    self.dragging = false
    self.tmode = parser:get_string("tmode", "sec") -- "smp", "sec", "milis"
    self.tout = 1 -- 1 (on) or 0 (off)
    self.verbose = parser:get_bool("verbose", parser:get_bool("verb", false))

    -- Layout constants
    self.btn_margin = 5
    self.slider_height = 5
    self.clock_width = 100
    
    -- Calculated positions (will be updated in postinitialize or paint)
    self:update_layout()

    self.sr = 48000 -- Default sample rate

    return true
end

function thisobj:postinitialize()
    self:set_size(self.width-2, self.height-2)
    self:repaint()
end

function thisobj:update_layout()
    -- Button size proportional to height, max 14
    self.btn_size = math.min(10, self.height * 0.75)

    local y_center = self.height / 2
    local x_start = 5
    
    -- Record Button
    self.rec_btn = {x = x_start, y = y_center - self.btn_size/2, w = self.btn_size, h = self.btn_size}
    x_start = x_start + self.btn_size + self.btn_margin
    
    -- Pause Button
    self.pause_btn = {x = x_start, y = y_center - self.btn_size/2, w = self.btn_size, h = self.btn_size}
    x_start = x_start + self.btn_size + self.btn_margin
    
    -- Stop Button
    self.stop_btn = {x = x_start, y = y_center - self.btn_size/2, w = self.btn_size, h = self.btn_size}
    x_start = x_start + self.btn_size + self.btn_margin * 2
    
    -- Slider
    -- Aligned with bottom of buttons
    local btn_bottom = self.rec_btn.y + self.rec_btn.h
    local slider_y = self.height - self.slider_height - 2

    self.slider = {
        x = x_start-5, 
        y = slider_y, 
        w = self.width - x_start + 3, 
        h = self.slider_height
    }
    
    -- Clock Area (Above slider)
    local cw = self.width - x_start - 8
    if self.width < 115 then
        cw = math.max(cw, 50)
    end

    self.clock_rect = {
        x = x_start-2,
        y = slider_y - 10, -- Just above slider
        w = cw,
        h = 9
    }

    -- Total Time Clock (Smaller)
    self.tclock_rect = {
        x = x_start,
        y = slider_y - 9, -- Aligned baseline with clock_rect
        w = self.width - x_start - 8,
        h = 7
    }
end


function thisobj:paint(g)
    -- Background
    g:set_color(self.c_background[1], self.c_background[2], self.c_background[3])
    g:fill_all()
    g:set_color(200, 200, 200)
    g:stroke_rect(0, 0, self.width, self.height, 1)

    local function set_col(c, active)
        if active then
            g:set_color(c[1], c[2], c[3])
        else
            -- Inactive: decrease saturation to 20% (reduce by 80%) and brightness to 90% (reduce by 10%)
            local h = c.h
            local s = c.s * 0.2
            local b = c.b * 0.9
            
            local r, g_val, b_val = Colors.hsb(h, s, b)
            g:set_color(r, g_val, b_val)
        end
    end

    -- Record Button
    set_col(self.c_rec, self.state == "record")
    local p = self.rec_btn
    
    -- Use ellipse for rounded look if available, else rect
    if g.fill_ellipse then
        g:fill_ellipse(p.x, p.y, p.w, p.h)
    else
        g:fill_rect(p.x, p.y, p.w, p.h)
    end

    -- Pause Button
    set_col(self.c_pause, self.state == "pause")
    local pa = self.pause_btn
    local bar_w = pa.w / 3
    g:fill_rect(bar_w * 0.3 + pa.x, pa.y, bar_w, pa.h)
    g:fill_rect(bar_w * 0.3 + pa.x + bar_w * 1.5, pa.y, bar_w, pa.h)

    -- Stop Button
    set_col(self.c_stop, self.state == "stop")
    local s = self.stop_btn
    g:fill_rect(s.x, s.y, s.w, s.h)

    if self.width >= 90 then
        -- Slider Track
        g:set_color(self.c_posback[1], self.c_posback[2], self.c_posback[3])
        local sl = self.slider
        g:fill_rect(sl.x, sl.y, sl.w, sl.h)

        -- Slider Handle/Progress
        g:set_color(self.c_pos[1], self.c_pos[2], self.c_pos[3])
        local handle_w = 3
        local pos_x = sl.x + (self.position * sl.w)
        -- Draw progress bar
        g:fill_rect(sl.x, sl.y, pos_x - sl.x, sl.h)
        
        -- Draw handle
        -- Slightly darker for handle
        local hc = self.c_pos
        g:set_color(math.max(0, hc[1]-30), math.max(0, hc[2]-30), math.max(0, hc[3]-30))
        
        local handle_h = 9
        local handle_y = sl.y + sl.h/2 - handle_h/2
        
        -- Use oval for rounded look if available, else rect
        if g.fill_oval then
            g:fill_oval(pos_x - handle_w/2, handle_y, handle_w, handle_h)
        else
            g:fill_rect(pos_x - handle_w/2, handle_y, handle_w, handle_h)
        end

            -- Draw Clock
        local function fmt_time(s)
            local m = math.floor(s / 60)
            local sec = math.floor(s % 60)
            if self.width < 155 then
                return string.format("%02d'%02d", m, sec)
            else
                local cent = math.floor((s * 100) % 100)
                return string.format("%02d'%02d.%02d", m, sec, cent)
            end
        end
        
        local cur_time = self.position * self.duration
        local tot_time = self.duration
        local str1 = fmt_time(cur_time)
        local str2 = fmt_time(tot_time)
        
        -- Text color same as info button text (complementary to bg)
        local tc_r, tc_g, tc_b
        if self.c_background.source == "hsb" then
            tc_r, tc_g, tc_b = Colors.hsb(self.c_background.h, self.c_background.s, 1 - self.c_background.b)
        else
            tc_r = 255 - self.c_background[1]
            tc_g = 255 - self.c_background[2]
            tc_b = 255 - self.c_background[3]
        end
        
        -- Mix 40% text color with 60% background for "fainter" look
        local bg_r = self.c_background[1]
        local bg_g = self.c_background[2]
        local bg_b = self.c_background[3]
        
        local f_r = tc_r * 0.6 + bg_r * 0.4
        local f_g = tc_g * 0.6 + bg_g * 0.4
        local f_b = tc_b * 0.6 + bg_b * 0.4

        -- Layout: Current time left-aligned with slider, Total time right-aligned with slider
        g:set_color(tc_r, tc_g, tc_b)
        g:draw_text(str1, self.slider.x-2, self.clock_rect.y, self.clock_rect.w, self.clock_rect.h)
        
        if self.width >= 115 then
            g:set_color(f_r, f_g, f_b)
            local offset = (self.width < 155) and 21 or 31
            g:draw_text(str2, self.slider.x + self.slider.w - offset, self.tclock_rect.y, self.tclock_rect.w, self.tclock_rect.h)
        end
    end

end

function thisobj:update_slider_from_mouse(x)
    local rel_x = x - self.slider.x
    local pos = rel_x / self.slider.w
    self.position = math.max(0, math.min(1, pos))
    self:repaint()
    self:output_position()
end

function thisobj:mouse_down(x, y, button, mod)
    local function hit(b)
        return x >= b.x and x <= b.x + b.w and y >= b.y and y <= b.y + b.h
    end

    if hit(self.rec_btn) then
        self:in_1_record()
        self:outlet(1, "record", {})
    elseif hit(self.pause_btn) then
        self:in_1_pause()
        self:outlet(1, "pause", {})
    elseif hit(self.stop_btn) then
        self:in_1_stop()
    elseif self.width >= 90 and hit(self.slider) then
        self.dragging = true
        self:update_slider_from_mouse(x)
    end
end

function thisobj:mouse_drag(x, y, button, mod)
    if self.dragging then
        self:update_slider_from_mouse(x)
    end
end

function thisobj:mouse_up(x, y, button, mod)
    self.dragging = false
end

function thisobj:in_1_record()
    self:set_state("record", true)
end

function thisobj:in_1_pause()
    self:set_state("pause", true)
end

function thisobj:in_1_stop()
    self.state = "stop"
    self.position = 0
    self:repaint()
    self:outlet(1, "stop", {})
    self:output_position()
end

function thisobj:in_1_list(atoms)
    if type(atoms) == "table" and #atoms > 0 and type(atoms[1]) == "number" then
        self:in_1_float(atoms[1])
    end
end

function thisobj:in_1_float(f)
    if f == 0 then self:in_1_stop()
    elseif f == 1 then self:in_1_record()
    elseif f == 2 then self:in_1_pause()
    end
end

function thisobj:in_1_pos(atoms)
    local p = atoms[1]
    if type(p) == "number" then
        self.position = math.max(0, math.min(1, p))
        self:repaint()
    end
end

function thisobj:in_1_duration(atoms)
    local d = atoms[1]
    if type(d) == "number" then
        self.duration = math.max(0, d)
        self:repaint()
    end
end

function thisobj:dsp(sr, bs, ch)
    self.sr = sr
    return true
end

function thisobj:output_position()
    if self.tout == 0 then return end

    local sec = self.position * self.duration
    local val = 0
    local selector = "smp"
    
    if self.tmode == "sec" then
        val = sec
        selector = "sec"
    elseif self.tmode == "milis" then
        val = sec * 1000
        selector = "ms"
    else -- "smp"
        val = math.floor(sec * self.sr)
        selector = "smp"
    end
    
    self:outlet(1, selector, {val})
end

function thisobj:in_1_tmode(atoms)
    local mode = atoms[1]
    if mode == "smp" or mode == "sec" or mode == "milis" then
        self.tmode = mode
    end
end

function thisobj:in_1_tout(atoms)
    local val = atoms[1]
    if type(val) == "number" then
        self.tout = (val ~= 0) and 1 or 0
    end
end

function thisobj:set_state(state, notify)
    self.state = state
    self:repaint()
    if notify then
        self:outlet(1, state, {})
    end
end

function thisobj:in_1_milis(atoms)
    if type(atoms[1]) == "number" and self.duration > 0 then
        local sec = atoms[1] / 1000
        self.position = math.max(0, math.min(1, sec / self.duration))
        self:repaint()
        self:output_position()
    end
end

function thisobj:in_1_sec(atoms)
    if type(atoms[1]) == "number" and self.duration > 0 then
        local sec = atoms[1]
        self.position = math.max(0, math.min(1, sec / self.duration))
        self:repaint()
        self:output_position()
    end
end

function thisobj:in_1_s(atoms)
    self:in_1_sec(atoms)
end

function thisobj:in_1_smp(atoms)
    if type(atoms[1]) == "number" and self.sr > 0 and self.duration > 0 then
        local sec = atoms[1] / self.sr
        self.position = math.max(0, math.min(1, sec / self.duration))
        self:repaint()
        self:output_position()
    end
end

function thisobj:in_1_tmilis(atoms)
    if type(atoms[1]) == "number" then
        self.duration = atoms[1] / 1000
        self:repaint()
    end
end

function thisobj:in_1_tsec(atoms)
    if type(atoms[1]) == "number" then
        self.duration = atoms[1]
        self:repaint()
    end
end

function thisobj:in_1_tsmp(atoms)
    if type(atoms[1]) == "number" and self.sr > 0 then
        self.duration = atoms[1] / self.sr
        self:repaint()
    end
end

function thisobj:in_1_set(atoms)
    local cmd = atoms[1]
    if cmd == "record" or cmd == "start" or cmd == 1 then
        self.state = "record"
        self:repaint()
    elseif cmd == "pause" or cmd == 2 then
        self.state = "pause"
        self:repaint()
    elseif cmd == "stop" or cmd == 0 then
        self.state = "stop"
        self.position = 0
        self:repaint()
    elseif cmd == "sec" or cmd == "s" then
        local val = atoms[2]
        if type(val) == "number" and self.duration > 0 then
            self.position = math.max(0, math.min(1, val / self.duration))
            self:repaint()
        end
    elseif cmd == "milis" or cmd == "ms" then
        local val = atoms[2]
        if type(val) == "number" and self.duration > 0 then
            self.position = math.max(0, math.min(1, (val / 1000) / self.duration))
            self:repaint()
        end
    elseif cmd == "smp" then
        local val = atoms[2]
        if type(val) == "number" and self.sr > 0 and self.duration > 0 then
            self.position = math.max(0, math.min(1, (val / self.sr) / self.duration))
            self:repaint()
        end
    elseif cmd == "pos" then
        local val = atoms[2]
        if type(val) == "number" then
            self.position = math.max(0, math.min(1, val))
            self:repaint()
        end
    elseif cmd == "tsec" then
        local val = atoms[2]
        if type(val) == "number" then
            self.duration = val
            self:repaint()
        end
    elseif cmd == "tmilis" then
        local val = atoms[2]
        if type(val) == "number" then
            self.duration = val / 1000
            self:repaint()
        end
    elseif cmd == "tsmp" then
        local val = atoms[2]
        if type(val) == "number" and self.sr > 0 then
            self.duration = val / self.sr
            self:repaint()
        end
    end
end

function thisobj:in_1_start()
    self:set_state("record", true)
end

function thisobj:in_1_sr(atoms)
    if type(atoms[1]) == "number" and atoms[1] > 0 then
        self.sr = atoms[1]
    end
end

function thisobj:in_1_getcode()
    local str = "ctgui.record"
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

function thisobj:throttled_repaint()
    if self.repaint_clock_running then
        self.repaint_pending = true
        return
    end
    self:_raw_repaint()
    self.repaint_clock_running = true
    self.repaint_clock:delay(1000 / self.gui_fps)
end

function thisobj:repaint_tick()
    self.repaint_clock_running = false
    if self.repaint_pending then
        self.repaint_pending = false
        self:throttled_repaint()
    end
end

function thisobj:throttled_output_position()
    if self.data_fps <= 0 then
        self:_raw_output_position()
        return
    end

    if self.output_clock_running then
        self.output_pending = true
        return
    end
    self:_raw_output_position()
    self.output_clock_running = true
    self.output_clock:delay(1000 / self.data_fps)
end

function thisobj:output_tick()
    self.output_clock_running = false
    if self.output_pending then
        self.output_pending = false
        self:throttled_output_position()
    end
end

function thisobj:in_1_guifps(atoms)
    local f = atoms[1]
    if type(f) == "number" and f > 0 then
        self.gui_fps = f
    end
end

function thisobj:in_1_guishutter(atoms)
    self:in_1_guifps(atoms)
end

function thisobj:in_1_fps(atoms)
    self:in_1_guifps(atoms)
end

function thisobj:in_1_shutter(atoms)
    self:in_1_guifps(atoms)
end

function thisobj:in_1_datafps(atoms)
    local f = atoms[1]
    if type(f) == "number" then
        self.data_fps = f
    end
end

function thisobj:in_1_datashutter(atoms)
    self:in_1_datafps(atoms)
end

function thisobj:in_2_tsec(atoms)
    local val = atoms[1]
    if type(val) == "number" then
        self.duration = val
        self:repaint()
    end
end

function thisobj:in_2_sec(atoms)
    local val = atoms[1]
    if type(val) == "number" and self.duration > 0 then
        self.position = math.max(0, math.min(1, val / self.duration))
        self:repaint()
    end
end

function thisobj:in_2_state(atoms)
    local s = atoms[1]
    if s == 0 then
        self.state = "stop"
        self.position = 0
        self:repaint()
    elseif s == 1 then
        self:set_state("record", false)
    elseif s == 2 then
        self:set_state("pause", false)
    end
end

function thisobj:in_2_tmode(atoms)
    local mode = atoms[1]
    if mode == "smp" or mode == "sec" or mode == "milis" then
        self.tmode = mode
    end
end

function thisobj:in_2_milis(atoms)
    local val = atoms[1]
    if type(val) == "number" and self.duration > 0 then
        self.position = math.max(0, math.min(1, (val / 1000) / self.duration))
        self:repaint()
    end
end

function thisobj:in_2_tmilis(atoms)
    local val = atoms[1]
    if type(val) == "number" then
        self.duration = val / 1000
        self:repaint()
    end
end

function thisobj:in_2_smp(atoms)
    local val = atoms[1]
    if type(val) == "number" and self.sr > 0 and self.duration > 0 then
        self.position = math.max(0, math.min(1, (val / self.sr) / self.duration))
        self:repaint()
    end
end

function thisobj:in_2_tsmp(atoms)
    local val = atoms[1]
    if type(val) == "number" and self.sr > 0 then
        self.duration = val / self.sr
        self:repaint()
    end
end

function thisobj:in_1_verbose(atoms)
    local v = atoms[1]
    if type(v) == "number" then
        self.verbose = (v ~= 0)
    else
        self.verbose = true
    end
end

function thisobj:in_1_verb(atoms)
    self:in_1_verbose(atoms)
end

function thisobj:in_2(sel, atoms)
    if self.verbose then
        pd.post(string.format("%s: no method for '%s' at inlet 2", self.object_name or "ctgui.record", sel))
    end
end

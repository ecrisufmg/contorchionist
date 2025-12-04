local ArgParser = require("pd_arg_parser")

local sntv3 = pd.Class:new():register("sntv3")

function sntv3:initialize(sel, atoms)
    self.creation_args = atoms
    local parser = ArgParser:new(atoms)

    self.openness = 0.5 -- Default openness (0 = cluster, 1 = spread)

    -- Defaults
    self.ranges = {
        S = {min = parser:get_float("Smin", 60), max = parser:get_float("Smax", 79)},
        A = {min = parser:get_float("Amin", 53), max = parser:get_float("Amax", 72)},
        T = {min = parser:get_float("Tmin", 48), max = parser:get_float("Tmax", 69)},
        B = {min = parser:get_float("Bmin", 40), max = parser:get_float("Bmax", 60)},
    }

    self.counts = {
        S = parser:get_float("Snum", 2),
        A = parser:get_float("Anum", 2),
        T = parser:get_float("Tnum", 2),
        B = parser:get_float("Bnum", 2),
    }    

    self.min_db = parser:get_float("mindb", -70)

    -- Delay parameters
    self.dephase = 0
    self.jitter_min = 0
    self.jitter_max = 0

    -- New parameters for sntv3
    self.freeze = false
    self.max_duration = 0 -- 0 = infinite
    self.min_duration = 0 -- 0 = none
    self.cooldown_time = 0.5 -- seconds

    pd.post("sntv3: Initialized")
    pd.post("S: ".. self.counts.S .. "x | range: [" .. self.ranges.S.min .. "-" .. self.ranges.S.max .. "]")
    pd.post("A: ".. self.counts.A .. "x | range: [" .. self.ranges.A.min .. "-" .. self.ranges.A.max .. "]")
    pd.post("T: ".. self.counts.T .. "x | range: [" .. self.ranges.T.min .. "-" .. self.ranges.T.max .. "]")
    pd.post("B: ".. self.counts.B .. "x | range: [" .. self.ranges.B.min .. "-" .. self.ranges.B.max .. "]")

    pd.post("Min dB threshold: " .. self.min_db)
    pd.post("Max Duration: " .. self.max_duration .. " s")
    pd.post("Min Duration: " .. self.min_duration .. " s")
    pd.post("Cooldown Time: " .. self.cooldown_time .. " s")

    -- Initialize voices
    self.voices = {}
    local types = {"S", "A", "T", "B"}
    for _, t in ipairs(types) do
        self.voices[t] = {}
    end

    -- Assign global indices bottom-up (B -> T -> A -> S)
    local types_ordered = {"B", "T", "A", "S"}
    local global_idx_counter = 0
    
    -- Calculate total voices for dephase normalization
    self.total_voice_count = self.counts.S + self.counts.A + self.counts.T + self.counts.B
    
    -- Single clock for all voices
    self.clock = pd.Clock:new():register(self, "tick")

    for _, t in ipairs(types_ordered) do
        for k = 1, self.counts[t] do
            -- Initialize last_midi_note to center of range to avoid initial bias
            local center = (self.ranges[t].min + self.ranges[t].max) / 2
            
            local v = {
                active = false,
                partial_id = -1,
                octave_offset = 0, -- Stores the transposition (k * 12)
                last_midi_note = center,
                type = t,
                index = k,
                global_index = global_idx_counter,
                pending_event = nil,
                queue = {}, -- Queue for delayed events
                start_time = 0, -- For max_duration tracking
                cooldown_until = 0, -- For rotation logic
                pending_release_time = nil -- For min_duration logic
            }
            
            table.insert(self.voices[t], v)
            global_idx_counter = global_idx_counter + 1
        end
    end

    self.current_time = {min = 0, sec = 0}
    self.is_ticking = false

    -- Inlets
    self.inlets = 2
    -- Outlets
    self.outlets = 1
    return true
end

-- Main clock tick function
function sntv3:tick()
    if self.is_ticking then return end
    self.is_ticking = true

    local now = os.clock()
    local next_wake_time = math.huge
    
    for t, v_list in pairs(self.voices) do
        for _, v in ipairs(v_list) do
            
            -- Pending Release Logic (Min Duration)
            if v.active and v.pending_release_time then
                if now >= v.pending_release_time then
                    -- Update state BEFORE output to prevent recursion loops
                    v.active = false
                    v.partial_id = -1
                    v.pending_release_time = nil
                    
                    self:schedule_output(v, v.last_midi_note, 0, -144, -1)
                elseif v.pending_release_time < next_wake_time then
                    next_wake_time = v.pending_release_time
                end
            end

            -- Voice Lifetime / Rotation Logic
            if v.active and self.max_duration > 0 then
                local duration = now - v.start_time
                if duration > self.max_duration then
                    -- Force release to allow rotation
                    -- Update state BEFORE output to prevent recursion loops
                    v.active = false
                    v.partial_id = -1
                    v.cooldown_until = now + self.cooldown_time
                    
                    self:schedule_output(v, v.last_midi_note, 0, -144, -1)
                end
            end

            -- Process this voice's queue
            while #v.queue > 0 do
                local head = v.queue[1]
                
                -- Calculate target time dynamically
                local dephase_delay_ms = 0
                if self.total_voice_count > 1 then
                    dephase_delay_ms = (v.global_index / (self.total_voice_count - 1)) * self.dephase
                end
                
                local total_delay_ms = dephase_delay_ms + head.jitter_val
                local target_time = head.arrival_time + (total_delay_ms / 1000.0)
                local remaining = target_time - now
                
                if remaining <= 0.0005 then
                    -- Ready to output
                    
                    -- Smart Burst Logic: Skip intermediate updates if next event is also ready
                    local skip = false
                    if #v.queue >= 2 then
                        local next_evt = v.queue[2]
                        local next_total_delay = dephase_delay_ms + next_evt.jitter_val
                        local next_target = next_evt.arrival_time + (next_total_delay / 1000.0)
                        
                        if (next_target - now) <= 0.0005 then
                            if head.data[5] == 0 then -- Current is Update
                                skip = true
                            end
                        end
                    end
                    
                    if not skip then
                        self:outlet(1, "list", head.data)
                    end
                    
                    table.remove(v.queue, 1)
                    -- Continue loop to check next event for this voice
                else
                    -- Not ready yet
                    if target_time < next_wake_time then
                        next_wake_time = target_time
                    end
                    break -- Stop checking this voice (FIFO assumption)
                end
            end
        end
    end
    
    -- Schedule next wake up
    if next_wake_time < math.huge then
        local delay_ms = (next_wake_time - now) * 1000.0
        if delay_ms < 0.5 then delay_ms = 0.5 end -- Minimum delay to avoid busy loop
        self.clock:delay(delay_ms)
    elseif self.max_duration > 0 then
        -- If we have max_duration enabled, we need to keep ticking to check for timeouts
        -- even if the queue is empty.
        self.clock:delay(100) -- Check every 100ms
    end
    
    self.is_ticking = false
end

-- Trigger queue processing for all voices (e.g. when params change)
function sntv3:update_all_queues()
    self:tick()
end

-- Inlet 2: Control Parameters (Openness, Ranges, Dephase, Jitter)
function sntv3:in_2_float(f)
    self.openness = math.max(0, math.min(1, f))
end

function sntv3:in_2_mindb(atoms)
    if #atoms > 0 then
        self.min_db = atoms[1]
    end
end

function sntv3:in_2_freeze(atoms)
    if #atoms > 0 then
        self.freeze = (atoms[1] ~= 0)
    end
end

function sntv3:in_2_max_duration(atoms)
    if #atoms > 0 then
        self.max_duration = atoms[1] / 1000.0 -- Convert ms to seconds
        self:tick() -- Trigger check immediately
    end
end

function sntv3:in_2_maxduration(atoms)
    self:in_2_max_duration(atoms)
end

function sntv3:in_2_cooldown(atoms)
    if #atoms > 0 then
        self.cooldown_time = atoms[1] / 1000.0 -- Convert ms to seconds
    end
end

function sntv3:in_2_min_duration(atoms)
    if #atoms > 0 then
        self.min_duration = atoms[1] / 1000.0
    end
end

function sntv3:in_2_minduration(atoms)
    self:in_2_min_duration(atoms)
end

function sntv3:in_2_openness(atoms)
    if #atoms > 0 then
        self.openness = math.max(0, math.min(1, atoms[1]))
    end
end

function sntv3:in_2_time(atoms)
    if #atoms >= 2 then
        self.current_time.min = atoms[1]
        self.current_time.sec = atoms[2]
    end
end

function sntv3:in_2_dephase(atoms)
    if #atoms > 0 then
        self.dephase = math.max(0, atoms[1])
        self:update_all_queues()
    end
end

function sntv3:in_2_jitter_min(atoms)
    if #atoms > 0 then
        self.jitter_min = math.max(0, atoms[1])
    end
end

function sntv3:in_2_jitter_max(atoms)
    if #atoms > 0 then
        self.jitter_max = math.max(0, atoms[1])
    end
end

function sntv3:in_2_S(atoms)
    if #atoms >= 2 then
        self.ranges.S.min = atoms[1]
        self.ranges.S.max = atoms[2]
    end
end

function sntv3:in_2_A(atoms)
    if #atoms >= 2 then
        self.ranges.A.min = atoms[1]
        self.ranges.A.max = atoms[2]
    end
end

function sntv3:in_2_T(atoms)
    if #atoms >= 2 then
        self.ranges.T.min = atoms[1]
        self.ranges.T.max = atoms[2]
    end
end

function sntv3:in_2_B(atoms)
    if #atoms >= 2 then
        self.ranges.B.min = atoms[1]
        self.ranges.B.max = atoms[2]
    end
end

function sntv3:in_2_list(atoms)
    if #atoms > 0 then
        local selector = atoms[1]
        if type(selector) == "string" then
            if (selector == "S_range" or selector == "S") and #atoms >= 3 then
                self.ranges.S.min = atoms[2]; self.ranges.S.max = atoms[3]
            elseif (selector == "A_range" or selector == "A") and #atoms >= 3 then
                self.ranges.A.min = atoms[2]; self.ranges.A.max = atoms[3]
            elseif (selector == "T_range" or selector == "T") and #atoms >= 3 then
                self.ranges.T.min = atoms[2]; self.ranges.T.max = atoms[3]
            elseif (selector == "B_range" or selector == "B") and #atoms >= 3 then
                self.ranges.B.min = atoms[2]; self.ranges.B.max = atoms[3]
            elseif selector == "mindb" and #atoms >= 2 then
                self.min_db = atoms[2]
            elseif selector == "freeze" and #atoms >= 2 then
                self.freeze = (atoms[2] ~= 0)
            elseif (selector == "max_duration" or selector == "maxduration") and #atoms >= 2 then
                self.max_duration = atoms[2] / 1000.0
                self:tick()
            elseif (selector == "min_duration" or selector == "minduration") and #atoms >= 2 then
                self.min_duration = atoms[2] / 1000.0
            elseif selector == "cooldown" and #atoms >= 2 then
                self.cooldown_time = atoms[2] / 1000.0
            elseif selector == "time" and #atoms >= 3 then
                self.current_time.min = atoms[2]
                self.current_time.sec = atoms[3]
            elseif selector == "dephase" and #atoms >= 2 then
                self.dephase = math.max(0, atoms[2])
                self:update_all_queues()
            elseif selector == "jitter_min" and #atoms >= 2 then
                self.jitter_min = math.max(0, atoms[2])
            elseif selector == "jitter_max" and #atoms >= 2 then
                self.jitter_max = math.max(0, atoms[2])
            elseif selector == "openness" and #atoms >= 2 then
                self.openness = math.max(0, math.min(1, atoms[2]))
            end
        elseif type(selector) == "number" then
            self:in_2_float(selector)
        end
    end
end

-- freq to midi
function sntv3:ftom(f)
    if f <= 0 then return 0 end
    return 69 + 12 * math.log(f / 440) / math.log(2)
end

function sntv3:get_active_notes()
    local notes = {}
    for t, v_list in pairs(self.voices) do
        for _, v in ipairs(v_list) do
            if v.active then
                table.insert(notes, v.last_midi_note)
            end
        end
    end
    return notes
end

function sntv3:find_voice_for_partial(partial_id)
    for t, v_list in pairs(self.voices) do
        for _, v in ipairs(v_list) do
            if v.active and v.partial_id == partial_id then
                return v
            end
        end
    end
    return nil
end

function sntv3:allocate_voice(partial_id, freq, db)
    local midi_base = self:ftom(freq)
    local active_notes = self:get_active_notes()
    local now = os.clock()
    
    local best_cand = nil
    local min_cost = math.huge
    
    local types = {"S", "A", "T", "B"}
    local candidates = {}

    for _, t in ipairs(types) do
        for _, v in ipairs(self.voices[t]) do
            -- Check if voice is inactive AND not in cooldown
            if not v.active and now >= v.cooldown_until then
                local min_r = self.ranges[t].min
                local max_r = self.ranges[t].max
                
                local k_min = math.ceil((min_r - midi_base) / 12)
                local k_max = math.floor((max_r - midi_base) / 12)
                
                for k = k_min, k_max do
                    local candidate_note = midi_base + k * 12
                    table.insert(candidates, {
                        voice = v,
                        note = candidate_note,
                        offset = k * 12
                    })
                end
            end
        end
    end
    
    if #candidates == 0 then return nil, nil, nil end
    
    for _, cand in ipairs(candidates) do
        local h_dist = math.abs(cand.note - cand.voice.last_midi_note)
        local v_dist_sum = 0
        if #active_notes > 0 then
            for _, an in ipairs(active_notes) do
                v_dist_sum = v_dist_sum + math.abs(cand.note - an)
            end
            v_dist_sum = v_dist_sum / #active_notes
        end
        
        local v_cost = (1 - 2 * self.openness) * v_dist_sum
        local total_cost = h_dist + v_cost
        
        if total_cost < min_cost then
            min_cost = total_cost
            best_cand = cand
        end
    end
    
    if best_cand then
        return best_cand.voice, best_cand.note, best_cand.offset
    else
        return nil, nil, nil
    end
end

function sntv3:in_1_list(atoms)
    if #atoms < 4 then return end
    
    -- Freeze Logic: Ignore all input if frozen
    if self.freeze then return end
    
    local bin_index = atoms[1]
    local freq = atoms[2]
    local db = atoms[3]
    local flag = atoms[4]
    
    -- Gate logic
    if db < self.min_db then
        if flag == 1 then
            return -- Ignore new notes below threshold
        elseif flag == 0 then
            flag = -1 -- Force release for updates below threshold
        end
    end
    
    if flag == 1 then
        -- New partial
        local existing = self:find_voice_for_partial(bin_index)
        if existing then
            if existing.pending_release_time then
                -- Rescue dying note!
                existing.pending_release_time = nil
                -- Treat as update
                local midi_out = self:ftom(freq) + existing.octave_offset
                existing.last_midi_note = midi_out
                self:schedule_output(existing, midi_out, freq, db, 0)
                return
            end
            
            local midi_out_old = self:ftom(freq) + existing.octave_offset
            self:schedule_output(existing, midi_out_old, freq, db, -1)
            
            existing.active = false
            existing.partial_id = -1
            existing.pending_release_time = nil
        end
        
        local voice, note, offset = self:allocate_voice(bin_index, freq, db)
        if voice then
            voice.active = true
            voice.partial_id = bin_index
            voice.octave_offset = offset
            voice.last_midi_note = note
            voice.start_time = os.clock() -- Track start time
            voice.pending_release_time = nil
            self:schedule_output(voice, note, freq, db, 1)
        end
        
    elseif flag == 0 then
        -- Continuation
        local voice = self:find_voice_for_partial(bin_index)
        if voice then
            if voice.pending_release_time then
                 voice.pending_release_time = nil -- Rescue
            end
            
            local midi_out = self:ftom(freq) + voice.octave_offset
            voice.last_midi_note = midi_out -- Update current note
            self:schedule_output(voice, midi_out, freq, db, 0)
        else
            -- ORPHAN UPDATE LOGIC:
            -- We received an update for a partial we aren't tracking.
            -- This happens if the patch started late, parameters changed,
            -- OR if the voice was force-released by max_duration logic.
            -- Treat it as a new allocation (Attack).
            
            local voice, note, offset = self:allocate_voice(bin_index, freq, db)
            if voice then
                voice.active = true
                voice.partial_id = bin_index
                voice.octave_offset = offset
                voice.last_midi_note = note
                voice.start_time = os.clock() -- Track start time
                voice.pending_release_time = nil
                -- Send as Attack (flag 1) to ensure envelope triggers
                self:schedule_output(voice, note, freq, db, 1)
            end
        end
        
    elseif flag == -1 then
        -- End
        local voice = self:find_voice_for_partial(bin_index)
        if voice then
            local duration = os.clock() - voice.start_time
            if duration < self.min_duration then
                -- Schedule delayed release
                voice.pending_release_time = voice.start_time + self.min_duration
                self:tick() -- Ensure clock is running to catch this release
            else
                -- Standard release
                local midi_out = self:ftom(freq) + voice.octave_offset
                self:schedule_output(voice, midi_out, freq, db, -1)
                
                voice.active = false
                voice.partial_id = -1
                voice.pending_release_time = nil
            end
        end
    end
end

function sntv3:schedule_output(voice, midi, freq, db, flag)
    -- Calculate jitter for this event
    local jitter = 0
    if self.jitter_max > 0 then
        -- Simple random float between min and max
        jitter = self.jitter_min + math.random() * (self.jitter_max - self.jitter_min)
    end

    local out_data = {
        voice.global_index,
        midi,
        freq,
        db,
        flag,
        self.current_time.min,
        self.current_time.sec
    }
    
    -- Push to queue
    table.insert(voice.queue, {
        arrival_time = os.clock(),
        jitter_val = jitter,
        data = out_data
    })
    
    -- Trigger processing
    self:tick()
end

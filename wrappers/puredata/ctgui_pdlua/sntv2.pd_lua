local ArgParser = require("pd_arg_parser")

local sntv2 = pd.Class:new():register("sntv2")

function sntv2:initialize(sel, atoms)
    self.creation_args = atoms
    local parser = ArgParser:new(atoms)

    self.openness = 0.5 -- Default openness (0 = cluster, 1 = spread)

    -- Defaults
    self.ranges = {
        S = {min = parser:get_float("Smin", 60), max = parser:get_float("Smax", 84)},
        A = {min = parser:get_float("Amin", 53), max = parser:get_float("Amax", 77)},
        T = {min = parser:get_float("Tmin", 48), max = parser:get_float("Tmax", 72)},
        B = {min = parser:get_float("Bmin", 40), max = parser:get_float("Bmax", 64)},
    }

    self.counts = {
        S = parser:get_float("Snum", 2),
        A = parser:get_float("Anum", 2),
        T = parser:get_float("Tnum", 2),
        B = parser:get_float("Bnum", 2),
    }    

    -- Delay parameters
    self.dephase = 0
    self.jitter_min = 0
    self.jitter_max = 0

    pd.post("sntv2: Initialized")
    pd.post("S: ".. self.counts.S .. "x | range: [" .. self.ranges.S.min .. "-" .. self.ranges.S.max .. "]")
    pd.post("A: ".. self.counts.A .. "x | range: [" .. self.ranges.A.min .. "-" .. self.ranges.A.max .. "]")
    pd.post("T: ".. self.counts.T .. "x | range: [" .. self.ranges.T.min .. "-" .. self.ranges.T.max .. "]")
    pd.post("B: ".. self.counts.B .. "x | range: [" .. self.ranges.B.min .. "-" .. self.ranges.B.max .. "]")

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
                queue = {} -- Queue for delayed events
            }
            
            table.insert(self.voices[t], v)
            global_idx_counter = global_idx_counter + 1
        end
    end

    self.current_time = {min = 0, sec = 0}

    -- Inlets
    self.inlets = 6
    -- Outlets
    self.outlets = 1
    return true
end

-- Main clock tick function
function sntv2:tick()
    local now = os.clock()
    local next_wake_time = math.huge
    
    for t, v_list in pairs(self.voices) do
        for _, v in ipairs(v_list) do
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
    end
end

-- Trigger queue processing for all voices (e.g. when params change)
function sntv2:update_all_queues()
    self:tick()
end

-- Inlet 2: Control Parameters (Openness, Ranges, Dephase, Jitter)
function sntv2:in_2_float(f)
    self.openness = math.max(0, math.min(1, f))
end

function sntv2:in_2_list(atoms)
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
            end
        elseif type(selector) == "number" then
            self:in_2_float(selector)
        end
    end
end

-- Inlet 3: Time
function sntv2:in_3_list(atoms)
    if #atoms >= 2 then
        self.current_time.min = atoms[1]
        self.current_time.sec = atoms[2]
    end
end

-- Inlet 4: Dephase
function sntv2:in_4_float(f)
    self.dephase = math.max(0, f)
    self:update_all_queues()
end

function sntv2:in_4_list(atoms)
    if type(atoms) == "table" and #atoms > 0 and type(atoms[1]) == "number" then
        self:in_4_float(atoms[1])
    elseif type(atoms) == "number" then
        self:in_4_float(atoms)
    end
end

-- Inlet 5: Jitter Min
function sntv2:in_5_float(f)
    self.jitter_min = math.max(0, f)
end

function sntv2:in_5_list(atoms)
    if type(atoms) == "table" and #atoms > 0 and type(atoms[1]) == "number" then
        self:in_5_float(atoms[1])
    elseif type(atoms) == "number" then
        self:in_5_float(atoms)
    end
end

-- Inlet 6: Jitter Max
function sntv2:in_6_float(f)
    self.jitter_max = math.max(0, f)
end

function sntv2:in_6_list(atoms)
    if type(atoms) == "table" and #atoms > 0 and type(atoms[1]) == "number" then
        self:in_6_float(atoms[1])
    elseif type(atoms) == "number" then
        self:in_6_float(atoms)
    end
end

-- freq to midi
function sntv2:ftom(f)
    if f <= 0 then return 0 end
    return 69 + 12 * math.log(f / 440) / math.log(2)
end

function sntv2:get_active_notes()
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

function sntv2:find_voice_for_partial(partial_id)
    for t, v_list in pairs(self.voices) do
        for _, v in ipairs(v_list) do
            if v.active and v.partial_id == partial_id then
                return v
            end
        end
    end
    return nil
end

function sntv2:allocate_voice(partial_id, freq, db)
    local midi_base = self:ftom(freq)
    local active_notes = self:get_active_notes()
    
    local best_cand = nil
    local min_cost = math.huge
    
    local types = {"S", "A", "T", "B"}
    local candidates = {}

    for _, t in ipairs(types) do
        for _, v in ipairs(self.voices[t]) do
            if not v.active then
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

function sntv2:in_1_list(atoms)
    if #atoms < 4 then return end
    
    local bin_index = atoms[1]
    local freq = atoms[2]
    local db = atoms[3]
    local flag = atoms[4]
    
    if flag == 1 then
        -- New partial
        local existing = self:find_voice_for_partial(bin_index)
        if existing then
            local midi_out_old = self:ftom(freq) + existing.octave_offset
            self:schedule_output(existing, midi_out_old, freq, db, -1)
            
            existing.active = false
            existing.partial_id = -1
        end
        
        local voice, note, offset = self:allocate_voice(bin_index, freq, db)
        if voice then
            voice.active = true
            voice.partial_id = bin_index
            voice.octave_offset = offset
            voice.last_midi_note = note
            self:schedule_output(voice, note, freq, db, 1)
        end
        
    elseif flag == 0 then
        -- Continuation
        local voice = self:find_voice_for_partial(bin_index)
        if voice then
            local midi_out = self:ftom(freq) + voice.octave_offset
            voice.last_midi_note = midi_out -- Update current note
            self:schedule_output(voice, midi_out, freq, db, 0)
        else
            -- ORPHAN UPDATE LOGIC:
            -- We received an update for a partial we aren't tracking.
            -- This happens if the patch started late or parameters changed.
            -- Treat it as a new allocation (Attack).
            
            local voice, note, offset = self:allocate_voice(bin_index, freq, db)
            if voice then
                voice.active = true
                voice.partial_id = bin_index
                voice.octave_offset = offset
                voice.last_midi_note = note
                -- Send as Attack (flag 1) to ensure envelope triggers
                self:schedule_output(voice, note, freq, db, 1)
            end
        end
        
    elseif flag == -1 then
        -- End
        local voice = self:find_voice_for_partial(bin_index)
        if voice then
            local midi_out = self:ftom(freq) + voice.octave_offset
            self:schedule_output(voice, midi_out, freq, db, -1)
            
            voice.active = false
            voice.partial_id = -1
        end
    end
end

function sntv2:schedule_output(voice, midi, freq, db, flag)
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
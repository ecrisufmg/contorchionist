local ArgParser = require("pd_arg_parser")

local sntv = pd.Class:new():register("sntv")

function sntv:initialize(sel, atoms)
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

    pd.post("sntv: Initialized")
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

    for _, t in ipairs(types_ordered) do
        for k = 1, self.counts[t] do
            -- Initialize last_midi_note to center of range to avoid initial bias
            local center = (self.ranges[t].min + self.ranges[t].max) / 2
            table.insert(self.voices[t], {
                active = false,
                partial_id = -1,
                octave_offset = 0, -- Stores the transposition (k * 12)
                last_midi_note = center,
                type = t,
                index = k,
                global_index = global_idx_counter
            })
            global_idx_counter = global_idx_counter + 1
        end
    end

    self.current_time = {min = 0, sec = 0}

    -- Inlets
    self.inlets = 3
    -- Outlets
    self.outlets = 1
    return true
end

-- Inlet 2: Control Parameters (Openness, Ranges)
function sntv:in_2_float(f)
    self.openness = math.max(0, math.min(1, f))
end

function sntv:in_2_list(atoms)
    if #atoms > 0 then
        local selector = atoms[1]
        if type(selector) == "string" then
            -- Handle range updates: S_range min max
            if (selector == "S_range" or selector == "S") and #atoms >= 3 then
                self.ranges.S.min = atoms[2]
                self.ranges.S.max = atoms[3]
            elseif (selector == "A_range" or selector == "A") and #atoms >= 3 then
                self.ranges.A.min = atoms[2]
                self.ranges.A.max = atoms[3]
            elseif (selector == "T_range" or selector == "T") and #atoms >= 3 then
                self.ranges.T.min = atoms[2]
                self.ranges.T.max = atoms[3]
            elseif (selector == "B_range" or selector == "B") and #atoms >= 3 then
                self.ranges.B.min = atoms[2]
                self.ranges.B.max = atoms[3]
            end
        elseif type(selector) == "number" then
            -- Treat single float as openness
            self:in_2_float(selector)
        end
    end
end

-- Inlet 3: Time
function sntv:in_3_list(atoms)
    if #atoms >= 2 then
        self.current_time.min = atoms[1]
        self.current_time.sec = atoms[2]
    end
end

-- freq to midi
function sntv:ftom(f)
    if f <= 0 then return 0 end
    return 69 + 12 * math.log(f / 440) / math.log(2)
end

function sntv:get_active_notes()
    local notes = {}
    for t, v_list in pairs(self.voices) do
        for _, v in ipairs(v_list) do
            if v.active then
                -- Calculate current note based on last known input? 
                -- Or just use the last output note?
                -- Using last_midi_note is correct as it stores the currently playing note for active voices
                table.insert(notes, v.last_midi_note)
            end
        end
    end
    return notes
end

function sntv:find_voice_for_partial(partial_id)
    for t, v_list in pairs(self.voices) do
        for _, v in ipairs(v_list) do
            if v.active and v.partial_id == partial_id then
                return v
            end
        end
    end
    return nil
end

function sntv:allocate_voice(partial_id, freq, db)
    local midi_base = self:ftom(freq)
    local active_notes = self:get_active_notes()
    
    local best_cand = nil
    local min_cost = math.huge
    
    -- Iterate over all voice types and indices
    local types = {"S", "A", "T", "B"}
    
    -- 4.2 Projection and Candidate Generation
    local candidates = {}

    for _, t in ipairs(types) do
        for _, v in ipairs(self.voices[t]) do
            if not v.active then -- 4.1 Voice State Filtering (Idle only)
                local min_r = self.ranges[t].min
                local max_r = self.ranges[t].max
                
                -- Calculate valid octaves
                -- midi_base + k*12 in [min_r, max_r]
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
    
    if #candidates == 0 then
        return nil, nil, nil -- No voice available or no range fit
    end
    
    -- 4.3 Decision Heuristics
    for _, cand in ipairs(candidates) do
        -- 3.1 Horizontal Proximity (Voice Leading)
        -- Minimize distance to last note executed by THIS voice
        local h_dist = math.abs(cand.note - cand.voice.last_midi_note)
        
        -- 3.2 Vertical Distribution (Voicing/Openness)
        local v_dist_sum = 0
        if #active_notes > 0 then
            for _, an in ipairs(active_notes) do
                v_dist_sum = v_dist_sum + math.abs(cand.note - an)
            end
            -- Normalize by number of active notes to keep scale consistent?
            -- Or just sum. Sum encourages spreading more as more notes are added.
            -- Let's use average to keep it balanced with h_dist
            v_dist_sum = v_dist_sum / #active_notes
        end
        
        -- Cost Function
        -- Openness 0 (Cluster): Minimize v_dist_sum -> Cost += v_dist_sum
        -- Openness 1 (Spread): Maximize v_dist_sum -> Cost -= v_dist_sum
        -- Formula: (1 - 2*openness) * v_dist_sum
        -- If openness=0.5, vertical distance doesn't matter.
        
        local v_cost = (1 - 2 * self.openness) * v_dist_sum
        
        -- Total Cost
        -- We weight horizontal distance slightly higher to ensure smooth voice leading
        -- unless openness is extreme.
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

function sntv:in_1_list(atoms)
    -- New format: <bin_index> <freq> <mag> <state> <rank>
    if #atoms < 4 then return end
    
    local bin_index = atoms[1]  -- Stable ID (FFT bin)
    local freq = atoms[2]
    local db = atoms[3]
    local flag = atoms[4]
    -- local rank = atoms[5] -- Optional, not used for tracking
    
    if flag == 1 then
        -- New partial
        local existing = self:find_voice_for_partial(bin_index)
        if existing then
            -- Partial ID reused or re-triggered without release
            -- Send release for the old note first
            local midi_out_old = self:ftom(freq) + existing.octave_offset
            self:output_voice(existing, midi_out_old, freq, db, -1)
            
            -- Now reallocate as a new note
            existing.active = false
            existing.partial_id = -1
        end
        
        -- Allocate voice
        local voice, note, offset = self:allocate_voice(bin_index, freq, db)
        if voice then
            voice.active = true
            voice.partial_id = bin_index
            voice.octave_offset = offset
            voice.last_midi_note = note
            self:output_voice(voice, note, freq, db, 1)
        else
            -- Allocation failed (full or out of range)
        end
        
    elseif flag == 0 then
        -- Continuation
        local voice = self:find_voice_for_partial(bin_index)
        if voice then
            -- 4.1 ...strictly preserving the pitch offset
            local midi_out = self:ftom(freq) + voice.octave_offset
            voice.last_midi_note = midi_out -- Update current note
            self:output_voice(voice, midi_out, freq, db, 0)
        end
        
    elseif flag == -1 then
        -- End
        local voice = self:find_voice_for_partial(bin_index)
        if voice then
            -- Output release
            local midi_out = self:ftom(freq) + voice.octave_offset
            self:output_voice(voice, midi_out, freq, db, -1)
            
            voice.active = false
            voice.partial_id = -1
            -- voice.last_midi_note is PRESERVED for voice leading logic
        end
    end
end

function sntv:output_voice(voice, midi, freq, db, flag)
    -- Output format: list <global_index> <midi> <freq> <db> <flag> <min> <sec>
    local out = {
        voice.global_index,
        midi,
        freq,
        db,
        flag,
        self.current_time.min,
        self.current_time.sec
    }
    self:outlet(1, "list", out)
end

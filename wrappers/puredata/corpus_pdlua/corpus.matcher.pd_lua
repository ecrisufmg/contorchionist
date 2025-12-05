-- corpus.matcher.pd_lua
-- Matches spectral analysis data to a corpus of audio segments.

local json = require("json")
local ArgParser = require("pd_arg_parser")

local CorpusMatcher = pd.Class:new():register("corpus.matcher")

function CorpusMatcher:initialize(sel, atoms)
    self.inlets = 4
    self.outlets = 4

    -- State variables
    self.corpus = {}
    self.filtered_corpus = {} -- Subset with crest > 100, etc.
    self.last_trigger_time = 0
    
    -- Parameters (defaults)
    self.threshold_db = -40
    self.rate_hz = 4
    self.smoothing = 0.5 -- 0 to 1 (1 = no smoothing, instant update)
    
    -- Parse arguments
    local parser = ArgParser:new(atoms)
    
    -- Update parameters from flags
    self.threshold_db = tonumber(parser:get_value("thresh")) or self.threshold_db
    self.rate_hz = tonumber(parser:get_value("rate")) or self.rate_hz
    self.smoothing = tonumber(parser:get_value("smooth")) or self.smoothing
    
    -- Azimuth smoothing state (Unit vector components)
    self.az_x = 0
    self.az_y = 0

    -- Load corpus automatically if possible
    local file_path = parser:get_value("file")
    if file_path then
        self:in_1_read(file_path)
    end
    
    return true
end

function CorpusMatcher:in_2_float(f)
    self.threshold_db = f
end

function CorpusMatcher:in_3_float(f)
    self.rate_hz = f
    if self.rate_hz <= 0 then self.rate_hz = 0.1 end
end

function CorpusMatcher:in_4_float(f)
    self.smoothing = math.max(0, math.min(1, f))
end

function CorpusMatcher:in_1_read(filename)
    local f = io.open(filename, "r")
    if not f then
        pd.post("corpus.matcher: Could not open file " .. filename)
        return
    end
    local content = f:read("*all")
    f:close()
    
    local status, data = pcall(json.decode, content)
    if not status then
        pd.post("corpus.matcher: JSON decode error: " .. data)
        return
    end
    
    self.corpus = data
    self:filter_corpus()
    pd.post("corpus.matcher: Loaded " .. #self.corpus .. " segments. Filtered down to " .. #self.filtered_corpus)
end

function CorpusMatcher:filter_corpus()
    self.filtered_corpus = {}
    -- Filter criteria: Crest > 100 AND Tonality > 0.8 (arbitrary default, maybe expose later)
    -- The design doc mentions "Crest > 100 & High Tonality"
    local tonality_thresh = 0.6 
    
    for _, segment in ipairs(self.corpus) do
        -- Ensure keys exist and are numbers
        local crest = tonumber(segment.crest) or 0
        local tonality = tonumber(segment.tonality) or 0
        
        if crest > 100 and tonality > tonality_thresh then
            table.insert(self.filtered_corpus, segment)
        end
    end
    
    -- Sort by centroid frequency for binary search (optional, but good for performance)
    table.sort(self.filtered_corpus, function(a, b)
        return (tonumber(a.centroid) or 0) < (tonumber(b.centroid) or 0)
    end)
end

function CorpusMatcher:find_closest_sample(freq)
    if #self.filtered_corpus == 0 then return nil end
    
    -- Linear search for now (simple and robust enough for small-ish corpora < 1000 items)
    -- If corpus is large, implement binary search.
    
    local closest = nil
    local min_diff = math.huge
    
    for _, segment in ipairs(self.filtered_corpus) do
        local cent = tonumber(segment.centroid) or 0
        local diff = math.abs(cent - freq)
        if diff < min_diff then
            min_diff = diff
            closest = segment
        end
    end
    
    return closest
end

function CorpusMatcher:in_1_list(atoms)
    -- Expected input: <band_index> <azimuth> <strength> <level_db> <frequency>
    if #atoms < 5 then return end
    
    local band_idx = atoms[1]
    local azimuth = atoms[2]
    local strength = atoms[3]
    local level_db = atoms[4]
    local freq = atoms[5]
    
    -- 1. Threshold Check
    if level_db < self.threshold_db then return end
    
    -- 2. Rate Limiting
    local now = pd.sys_time() -- Returns time in seconds? No, usually not available in standard pd-lua.
    -- pd-lua doesn't have a built-in high-res timer easily accessible without external libs or clock objects.
    -- However, we can use os.clock() which is usually available.
    local now_ms = os.clock() * 1000
    local min_interval = 1000 / self.rate_hz
    
    if (now_ms - self.last_trigger_time) < min_interval then
        return
    end
    
    self.last_trigger_time = now_ms
    
    -- 3. Find Match
    local match = self:find_closest_sample(freq)
    if not match then return end
    
    -- 4. Calculate Outputs
    
    -- Azimuth Smoothing (Circular)
    -- Convert degrees to radians
    local rad = math.rad(azimuth)
    local x = math.cos(rad)
    local y = math.sin(rad)
    
    -- Leaky integrator
    local alpha = self.smoothing
    -- If alpha is 1, we use new value entirely. If 0, we never update (bad).
    -- Let's interpret smoothing: 0 = instant (no smooth), 1 = frozen.
    -- Actually, standard is: out = out * smooth + in * (1-smooth)
    -- So if input smoothing is 0.9, it's very smooth.
    
    -- Re-interpreting the input parameter "smoothing" from design:
    -- Let's assume the user gives a factor 0..1 where 0 is no smoothing.
    local s_factor = self.smoothing
    
    self.az_x = self.az_x * s_factor + x * (1 - s_factor)
    self.az_y = self.az_y * s_factor + y * (1 - s_factor)
    
    local smooth_az_rad = math.atan2(self.az_y, self.az_x)
    local smooth_az_deg = math.deg(smooth_az_rad)
    
    -- Rho Calculation
    -- Design: rho = strength ^ 0.5
    local rho = math.pow(strength, 0.5)
    
    -- Gain Calculation
    -- Map dB to linear: 10 ^ (db / 20)
    local gain = math.pow(10, level_db / 20)
    
    -- 5. Output
    -- Outlet 4: Gain
    self:outlet(4, "float", gain)
    
    -- Outlet 3: Rho
    self:outlet(3, "float", rho)
    
    -- Outlet 2: Azimuth
    self:outlet(2, "float", smooth_az_deg)
    
    -- Outlet 1: Player Control
    -- "open <filename>", then "1" to play
    if match.file then
        self:outlet(1, "list", { "open", match.file })
        self:outlet(1, "float", 1)
    end
end

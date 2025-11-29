local snt_timestamp = pd.Class:new():register("snt_timestamp")

function snt_timestamp:initialize(sel, atoms)
    self.last_output = ""
    self.inlets = 1
    self.outlets = 1
    pd.post("snt_timestamp: initialized")
    return true
end

function snt_timestamp:in_1_list(atoms)
    if #atoms < 2 then return end
    
    local min = tonumber(atoms[1])
    local sec = tonumber(atoms[2])
    
    if not min or not sec then return end
    
    -- Ignore inputs less than 15 seconds (start of piece)
    if min == 0 and sec < 15 then return end
    
    local r_sec = math.floor(sec + 0.5)
    
    -- Handle 60 seconds case (carry over to minute)
    if r_sec == 60 then
        r_sec = 0
        min = min + 1
    end
    
    if r_sec == 0 or r_sec == 15 or r_sec == 30 or r_sec == 45 then
        local formatted = string.format("%02d_%02d.mp3", math.floor(min), r_sec)
        
        if formatted ~= self.last_output then
            self:outlet(1, "symbol", {formatted})
            self.last_output = formatted
        end
    else
        -- Reset last_output when we are not on a target second
        -- This allows re-triggering if we seek back and pass the time again
        self.last_output = ""
    end
end

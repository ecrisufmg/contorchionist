#include "m_pd.h"
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

// Minimal JSON Parser for the specific format
// [ {"key": val, ...}, ... ]
struct Segment {
    std::string file;
    float centroid;
    float tonality;
    float crest;
    float duration;
};

class SimpleJsonParser {
public:
    static std::vector<Segment> parse(const std::string& content) {
        std::vector<Segment> segments;
        size_t pos = 0;
        
        // Find start of array
        size_t arrayStart = content.find('[', pos);
        if (arrayStart == std::string::npos) {
             post("corpus.matcher: JSON parse error: Could not find start of array '['");
             return segments;
        }
        pos = arrayStart + 1;

        while (true) {
            // Find start of object
            size_t objStart = content.find('{', pos);
            if (objStart == std::string::npos) break;
            
            // Parse object
            size_t endObj = content.find('}', objStart);
            if (endObj == std::string::npos) {
                post("corpus.matcher: JSON parse error: Could not find end of object '}'");
                break;
            }
            
            std::string objStr = content.substr(objStart, endObj - objStart + 1);
            segments.push_back(parseObject(objStr));
            
            pos = endObj + 1;
        }
        return segments;
    }

private:
    static Segment parseObject(const std::string& objStr) {
        Segment seg;
        seg.centroid = 0; seg.tonality = 0; seg.crest = 0; seg.duration = 0;
        
        seg.file = getString(objStr, "file");
        seg.centroid = getFloat(objStr, "centroid");
        seg.tonality = getFloat(objStr, "tonality");
        seg.crest = getFloat(objStr, "crest");
        seg.duration = getFloat(objStr, "duration");
        
        return seg;
    }

    static std::string getString(const std::string& json, const std::string& key) {
        std::string keyPattern = "\"" + key + "\"";
        size_t keyPos = json.find(keyPattern);
        if (keyPos == std::string::npos) return "";
        
        size_t colonPos = json.find(':', keyPos);
        size_t startQuote = json.find('"', colonPos);
        size_t endQuote = json.find('"', startQuote + 1);
        
        if (startQuote != std::string::npos && endQuote != std::string::npos) {
            return json.substr(startQuote + 1, endQuote - startQuote - 1);
        }
        return "";
    }

    static float getFloat(const std::string& json, const std::string& key) {
        std::string keyPattern = "\"" + key + "\"";
        size_t keyPos = json.find(keyPattern);
        if (keyPos == std::string::npos) return 0.0f;
        
        size_t colonPos = json.find(':', keyPos);
        size_t valStart = json.find_first_of("0123456789-.", colonPos);
        size_t valEnd = json.find_first_not_of("0123456789-.eE", valStart);
        
        if (valStart != std::string::npos) {
            std::string valStr = json.substr(valStart, valEnd - valStart);
            try {
                return std::stof(valStr);
            } catch (...) {
                return 0.0f;
            }
        }
        return 0.0f;
    }
};

static t_class *corpus_matcher_class;

typedef struct _corpus_matcher {
    t_object x_obj;
    
    // Outlets
    t_outlet *out_main; // list: filename gain rho azimuth
    t_outlet *out_info; // info/debug
    
    // Parameters
    float threshold_db;
    float rate_hz;
    float smoothing;
    
    // State
    double last_trigger_time;
    float az_x;
    float az_y;
    
    // Corpus
    std::vector<Segment> corpus;
    std::vector<Segment> filtered_corpus;
    
    t_canvas *x_canvas;

} t_corpus_matcher;

// Helper: Filter corpus
static void corpus_matcher_filter(t_corpus_matcher *x) {
    x->filtered_corpus.clear();
    // Relaxed thresholds for testing
    float tonality_thresh = 0.0f; // was 0.6f
    float crest_thresh = 0.0f;    // was 100.0f
    
    for (const auto& seg : x->corpus) {
        if (seg.crest > crest_thresh && seg.tonality > tonality_thresh) {
            x->filtered_corpus.push_back(seg);
        }
    }
    
    // Sort by centroid
    std::sort(x->filtered_corpus.begin(), x->filtered_corpus.end(), 
        [](const Segment& a, const Segment& b) {
            return a.centroid < b.centroid;
        });
        
    post("corpus.matcher: Loaded %lu segments. Filtered down to %lu", x->corpus.size(), x->filtered_corpus.size());
}

// Method: Read JSON
static void corpus_matcher_read(t_corpus_matcher *x, t_symbol *s) {
    std::string filename = s->s_name;
    std::ifstream file(filename);
    
    // Try relative to canvas if absolute fails
    if (!file.is_open()) {
        t_canvas *canvas = x->x_canvas;
        if (canvas) {
            const char *dir = canvas_getdir(canvas)->s_name;
            std::string path = std::string(dir) + "/" + filename;
            file.open(path);
            if (file.is_open()) {
                post("corpus.matcher: Opened %s", path.c_str());
            } else {
                 post("corpus.matcher: Failed to open %s", path.c_str());
            }
        }
    }
    
    if (!file.is_open()) {
        pd_error(x, "corpus.matcher: Could not open file %s", filename.c_str());
        return;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    
    if (content.empty()) {
        pd_error(x, "corpus.matcher: File %s is empty", filename.c_str());
        return;
    }

    // post("corpus.matcher: Read %lu bytes from file", content.size());
    
    x->corpus = SimpleJsonParser::parse(content);
    
    if (x->corpus.empty()) {
         pd_error(x, "corpus.matcher: Parsed 0 segments from JSON. Check format.");
         // Debug: print first few chars
         // post("corpus.matcher: Content start: %.100s", content.c_str());
    }

    corpus_matcher_filter(x);
}

// Method: List input
// <band_index> <azimuth> <strength> <level_db> <frequency>
static void corpus_matcher_list(t_corpus_matcher *x, t_symbol *s, int argc, t_atom *argv) {
    if (argc < 5) {
        // post("corpus.matcher: received list with %d args, expected >= 5", argc);
        return;
    }
    
    float azimuth = atom_getfloat(argv + 1);
    float strength = atom_getfloat(argv + 2);
    float level_db = atom_getfloat(argv + 3);
    float freq = atom_getfloat(argv + 4);
    
    // 1. Threshold Check
    if (level_db < x->threshold_db) {
        // post("corpus.matcher: level %.2f < thresh %.2f", level_db, x->threshold_db);
        return;
    }
    
    // 2. Rate Limiting
    double now_ms = clock_getlogicaltime(); 
    double min_interval = 1000.0 / (x->rate_hz > 0 ? x->rate_hz : 0.1f);
    double elapsed = clock_gettimesince(x->last_trigger_time);

    if (elapsed < min_interval) {
        // post("corpus.matcher: rate limited. elapsed %.2f < min %.2f", elapsed, min_interval);
        return;
    }
    x->last_trigger_time = now_ms;
    
    // 3. Find Match
    if (x->filtered_corpus.empty()) {
        post("corpus.matcher: filtered corpus is empty! (Total loaded: %lu)", x->corpus.size());
        return;
    }
    
    const Segment* closest = nullptr;
    float min_diff = 1e9f; // Large number
    
    // Linear search (optimize to binary later if needed)
    for (const auto& seg : x->filtered_corpus) {
        float diff = std::abs(seg.centroid - freq);
        if (diff < min_diff) {
            min_diff = diff;
            closest = &seg;
        }
    }
    
    if (!closest) {
        post("corpus.matcher: no closest match found?");
        return;
    }
    
    // 4. Calculate Outputs
    
    // Azimuth Smoothing
    float rad = azimuth * (M_PI / 180.0f);
    float input_x = std::cos(rad);
    float input_y = std::sin(rad);
    
    float s_factor = x->smoothing;
    x->az_x = x->az_x * s_factor + input_x * (1.0f - s_factor);
    x->az_y = x->az_y * s_factor + input_y * (1.0f - s_factor);
    
    float smooth_az_rad = std::atan2(x->az_y, x->az_x);
    float smooth_az_deg = smooth_az_rad * (180.0f / M_PI);
    
    // Rho
    float rho = std::pow(strength, 0.5f);
    
    // Gain
    float gain = std::pow(10.0f, level_db / 20.0f);
    
    // Output: filename, gain, rho, azimuth
    t_atom out_atoms[4];
    SETSYMBOL(out_atoms + 0, gensym(closest->file.c_str()));
    SETFLOAT(out_atoms + 1, gain);
    SETFLOAT(out_atoms + 2, rho);
    SETFLOAT(out_atoms + 3, smooth_az_deg);
    
    outlet_list(x->out_main, &s_list, 4, out_atoms);
    
    // Debug info to second outlet
    t_atom info[2];
    SETSYMBOL(info, gensym("match_freq"));
    SETFLOAT(info+1, closest->centroid);
    outlet_anything(x->out_info, gensym("debug"), 2, info);
}

// Parameter methods
static void corpus_matcher_thresh(t_corpus_matcher *x, t_floatarg f) { x->threshold_db = f; }
static void corpus_matcher_rate(t_corpus_matcher *x, t_floatarg f) { x->rate_hz = (f > 0 ? f : 0.1f); }
static void corpus_matcher_smooth(t_corpus_matcher *x, t_floatarg f) { x->smoothing = (f < 0 ? 0 : (f > 1 ? 1 : f)); }

// Constructor
static void *corpus_matcher_new(t_symbol *s, int argc, t_atom *argv) {
    t_corpus_matcher *x = (t_corpus_matcher *)pd_new(corpus_matcher_class);
    
    x->x_canvas = canvas_getcurrent();
    
    // Defaults
    x->threshold_db = -40;
    x->rate_hz = 4;
    x->smoothing = 0.5;
    x->last_trigger_time = 0;
    x->az_x = 0;
    x->az_y = 0;
    
    // Parse args
    // Simple arg parsing: -file <f> -thresh <t> -rate <r> -smooth <s>
    for (int i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            std::string flag = argv[i].a_w.w_symbol->s_name;
            if (flag == "-file" && i+1 < argc && argv[i+1].a_type == A_SYMBOL) {
                corpus_matcher_read(x, argv[i+1].a_w.w_symbol);
                i++;
            } else if (flag == "-thresh" && i+1 < argc && argv[i+1].a_type == A_FLOAT) {
                x->threshold_db = argv[i+1].a_w.w_float;
                i++;
            } else if (flag == "-rate" && i+1 < argc && argv[i+1].a_type == A_FLOAT) {
                x->rate_hz = argv[i+1].a_w.w_float;
                i++;
            } else if (flag == "-smooth" && i+1 < argc && argv[i+1].a_type == A_FLOAT) {
                x->smoothing = argv[i+1].a_w.w_float;
                i++;
            }
        }
    }
    
    // Inlets
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("thresh"));
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("rate"));
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("smooth"));
    
    // Outlets
    x->out_main = outlet_new(&x->x_obj, &s_list);
    x->out_info = outlet_new(&x->x_obj, &s_anything);
    
    // Initialize vectors
    new (&x->corpus) std::vector<Segment>();
    new (&x->filtered_corpus) std::vector<Segment>();
    
    return (void *)x;
}

static void corpus_matcher_free(t_corpus_matcher *x) {
    x->corpus.~vector();
    x->filtered_corpus.~vector();
}

extern "C" void setup_torch0x2ecorpus0x2ematcher_tilde(void) {
    corpus_matcher_class = class_new(gensym("torch.corpus.matcher~"),
        (t_newmethod)corpus_matcher_new,
        (t_method)corpus_matcher_free,
        sizeof(t_corpus_matcher),
        CLASS_DEFAULT,
        A_GIMME, 0);
        
    class_addlist(corpus_matcher_class, corpus_matcher_list);
    class_addmethod(corpus_matcher_class, (t_method)corpus_matcher_read, gensym("read"), A_SYMBOL, 0);
    class_addmethod(corpus_matcher_class, (t_method)corpus_matcher_thresh, gensym("thresh"), A_FLOAT, 0);
    class_addmethod(corpus_matcher_class, (t_method)corpus_matcher_rate, gensym("rate"), A_FLOAT, 0);
    class_addmethod(corpus_matcher_class, (t_method)corpus_matcher_smooth, gensym("smooth"), A_FLOAT, 0);
}

#ifndef CORE_AP_MONITOR_H
#define CORE_AP_MONITOR_H

#include <vector>
#include <chrono>
#include <string>
#include <functional>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace contorchionist {
    namespace core {
        namespace ap_monitor {

// struct to hold performance statistics
struct PerformanceStats {
    float mean_ms = 0.0f; // average processing time in milliseconds
    float min_ms = 0.0f; // minimum processing time in milliseconds
    float max_ms = 0.0f; // maximum processing time in milliseconds
    float std_dev_ms = 0.0f; // standard deviation of processing time in milliseconds
    float jitter_ms = 0.0f; // jitter (max - min) in milliseconds
    float throughput_hz = 0.0f; // throughput in inferences per second
    int total_inferences = 0; // total number of inferences
    int xrun_count = 0; // number of buffer overruns
    size_t sample_count = 0; // number of samples processed
};

/**
 * @brief Performance monitoring class for tracking and logging processing times.
 *
 * @tparam LoggerFunc Type of the logging function (default: std::function<void(const std::string&)>)
 */
template<typename LoggerFunc = std::function<void(const std::string&)>>
class PerformanceMonitor {
private:
    std::vector<float> processing_time_history;
    int max_history_size;
    int inference_count;
    int xrun_count;
    std::chrono::steady_clock::time_point last_stats_time;
    std::chrono::steady_clock::time_point start_time;
    LoggerFunc log_callback;


/**
 * @brief Constructor for PerformanceMonitor.
 *
 * @param max_history Maximum number of history entries to keep.
 */
public:
    explicit PerformanceMonitor(
        int max_history = 100,
        LoggerFunc logger = default_logger()
    ) : max_history_size(max_history),
        inference_count(0),
        xrun_count(0),
        log_callback(std::move(logger)) {
        
        processing_time_history.reserve(max_history_size);
        reset_timers();
    }

    // register processing time in milliseconds
    inline void record_processing_time(float time_ms) {
        processing_time_history.push_back(time_ms);
        inference_count++;

        // Keep history limited to max size
        if (processing_time_history.size() > static_cast<size_t>(max_history_size)) {
            processing_time_history.erase(processing_time_history.begin());
        }
    }

    // register xrun event
    inline void record_xrun() {
        xrun_count++;
    }

    // Calculate statistics
    inline PerformanceStats calculate_stats() const {
        PerformanceStats stats;
        
        if (processing_time_history.empty()) {
            return stats;
        }

        // avg, min, max
        float sum = std::accumulate(processing_time_history.begin(), processing_time_history.end(), 0.0f);
        stats.mean_ms = sum / processing_time_history.size();
        stats.min_ms = *std::min_element(processing_time_history.begin(), processing_time_history.end());
        stats.max_ms = *std::max_element(processing_time_history.begin(), processing_time_history.end());
        
        // std dev
        float sq_sum = std::inner_product(
            processing_time_history.begin(), 
            processing_time_history.end(), 
            processing_time_history.begin(), 
            0.0f
        );
        stats.std_dev_ms = std::sqrt(sq_sum / processing_time_history.size() - stats.mean_ms * stats.mean_ms);
        
        // Derived metrics
        stats.jitter_ms = stats.max_ms - stats.min_ms; // jitter
        stats.total_inferences = inference_count; // total inferences
        stats.xrun_count = xrun_count; // xrun count
        stats.sample_count = processing_time_history.size(); // sample count

        // Throughput
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_stats_time).count();
        if (duration > 0) {
            stats.throughput_hz = (processing_time_history.size() * 1000.0f) / duration;
        }
        
        return stats;
    }

    // Log current statistics
    inline void calculate_and_log_stats() {
        if (processing_time_history.empty()) {
            return;
        }

        PerformanceStats stats = calculate_stats();
        format_and_log_stats(stats);
    }

    // Reset statistics
    inline void reset() {
        processing_time_history.clear();
        inference_count = 0;
        xrun_count = 0;
        reset_timers();
    }

    // Check if we should log stats
    inline bool should_log_stats() const {
        return processing_time_history.size() >= static_cast<size_t>(max_history_size);
    }

    // Getters
    inline int get_inference_count() const { return inference_count; }
    inline int get_xrun_count() const { return xrun_count; }
    inline size_t get_history_size() const { return processing_time_history.size(); }
    inline bool has_data() const { return !processing_time_history.empty(); }
    inline int get_max_history_size() const { return max_history_size; }

    // Set log callback
    inline void set_log_callback(LoggerFunc callback) {
        log_callback = std::move(callback);
    }

// Private methods
private:
    inline void reset_timers() {
        auto now = std::chrono::steady_clock::now();
        last_stats_time = now;
        start_time = now;
    }

    // Format and log statistics
    inline void format_and_log_stats(const PerformanceStats& stats) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4);
        
        // Header
        oss << "--- Performance Stats (last " << stats.sample_count << " inferences) ---";
        log_callback(oss.str());
        oss.str(""); oss.clear();
        
        // Timing stats
        oss << "Processing Time (ms): Avg=" << stats.mean_ms 
            << ", Min=" << stats.min_ms 
            << ", Max=" << stats.max_ms 
            << ", StdDev=" << stats.std_dev_ms;
        log_callback(oss.str());
        oss.str(""); oss.clear();
        
        // Performance metrics
        oss << "Performance: Jitter=" << stats.jitter_ms 
            << "ms, Throughput=" << stats.throughput_hz << "Hz";
        log_callback(oss.str());
        oss.str(""); oss.clear();
        
        // Stability
        oss << "Stability: " << stats.xrun_count 
            << " xruns in " << stats.total_inferences << " total inferences";
        log_callback(oss.str());
        oss.str(""); oss.clear();
        
        // Footer
        log_callback("----------------------------------------------------");
    }

    // logger default (silent)
    static std::function<void(const std::string&)> default_logger() {
        return [](const std::string& msg) {
            (void)msg; // suppress warning
        };
    }
};

inline std::function<void(const std::string&)> create_stdout_logger(const std::string& prefix = "Monitor") {
    return [prefix](const std::string& msg) {
        std::cout << prefix << ": " << msg << std::endl;
    };
}

inline std::function<void(const std::string&)> create_stderr_logger(const std::string& prefix = "Monitor") {
    return [prefix](const std::string& msg) {
        std::cerr << prefix << ": " << msg << std::endl;
    };
}

inline std::function<void(const std::string&)> create_silent_logger() {
    return [](const std::string&) { /* silent */ };
}


template<typename LogFunction>
inline std::function<void(const std::string&)> create_logger(LogFunction log_func, const std::string& prefix = "Monitor") {
    return [prefix, log_func](const std::string& msg) {
        log_func(prefix + ": " + msg);
    };
}

// Convenience aliases
using StdoutMonitor = PerformanceMonitor<std::function<void(const std::string&)>>;
using PdMonitor = PerformanceMonitor<std::function<void(const std::string&)>>;

        } // namespace ap_monitor
    } // namespace core
} // namespace contorchionist

#endif // CORE_AP_MONITOR_H
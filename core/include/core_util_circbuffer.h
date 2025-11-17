#ifndef CORE_UTIL_CIRCBUFFER_H
#define CORE_UTIL_CIRCBUFFER_H


#include <vector>
#include <stdexcept>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <algorithm> // for std::min/max
#include <torch/torch.h>

// Set DEBUG_CIRCBUFFER to 1 to enable bounds checking with .at()
// (slower but safer for debugging index bugs)
#define DEBUG_CIRCBUFFER 0

namespace contorchionist {
namespace core {
namespace util_circbuffer {

/**
 * @class CircularBuffer
 * @brief Thread-safe circular buffer optimized for real-time contexts like Pure Data.
 *
 * Designed to be instantiated once and reused. Call clear() to reset state 
 * (e.g., in Pd's dsp method) instead of recreating the object.
 * Uses std::atomic<bool> for race condition safety during destruction.
 */
template <typename T>
class CircularBuffer {
public:
    /**
     * @brief Constructor that allocates the total buffer capacity.
     * @param capacity Maximum number of items the buffer can contain.
     */
    explicit CircularBuffer(size_t capacity)
        : buffer_(capacity), capacity_(capacity), head_(0), tail_(0), count_(0), is_valid_(true) {
        if (capacity == 0) {
            throw std::invalid_argument("CircularBuffer capacity cannot be zero.");
        }
    }

    /**
     * @brief Destructor.
     * Invalidates buffer to prevent concurrent operations and notifies
     * all waiting threads so they can unblock and terminate.
     */
    ~CircularBuffer() {
        is_valid_.store(false, std::memory_order_release);
        data_available_cv_.notify_all();
        space_available_cv_.notify_all();
    }

    // Disable copy and move to ensure single ownership and avoid lifecycle issues.
    CircularBuffer(const CircularBuffer&) = delete;
    CircularBuffer& operator=(const CircularBuffer&) = delete;
    CircularBuffer(CircularBuffer&&) = delete;
    CircularBuffer& operator=(CircularBuffer&&) = delete;
    
    // ===================================================================
    // MAIN OPERATION METHODS
    // ===================================================================

    /**
     * @brief Writes data to buffer, overwriting old data if buffer is full.
     * Ideal for real-time audio where latency is critical and new data is more important.
     * This operation is lock-free in that it doesn't block waiting for space.
     * @param data Pointer to data to be written.
     * @param num_samples Number of samples to write.
     * @return true if write was successful, false if num_samples > capacity.
     */
    bool write_overwrite(const T* data, size_t num_samples) {
        if (!is_valid_.load(std::memory_order_acquire)) return false;

        std::lock_guard<std::mutex> lock(mutex_);
        
        if (num_samples > capacity_) {
            return false; // Cannot write more than total capacity.
        }
        
        // If not enough space, advance read pointer to make room.
        size_t available_space = capacity_ - count_;
        if (num_samples > available_space) {
            size_t samples_to_discard = num_samples - available_space;
            head_ = (head_ + samples_to_discard) % capacity_;
            count_ -= samples_to_discard;
        }
        
        // Write the data
        for (size_t i = 0; i < num_samples; ++i) {
            #if DEBUG_CIRCBUFFER
                buffer_.at((tail_ + i) % capacity_) = data[i];
            #else
                buffer_[(tail_ + i) % capacity_] = data[i];
            #endif
        }
        tail_ = (tail_ + num_samples) % capacity_;
        count_ += num_samples;
        
        // Notify a waiting reader, if any (for blocking operations).
        data_available_cv_.notify_one();
        return true;
    }

    /**
     * @brief Non-blocking read operation (peek).
     * Reads up to num_samples_to_peek from the buffer into out_data without removing them.
     * @param out_data Pointer to the buffer where the peeked data will be stored.
     * @param num_samples_to_peek The maximum number of samples to peek.
     * @return The number of samples actually peeked.
     */
    size_t peek(T* out_data, size_t num_samples_to_peek) const {
        if (!is_valid_.load(std::memory_order_acquire)) return 0;
        std::lock_guard<std::mutex> lock(mutex_);
        size_t samples_to_copy = std::min(num_samples_to_peek, count_);
        for (size_t i = 0; i < samples_to_copy; ++i) {
            #if DEBUG_CIRCBUFFER
                out_data[i] = buffer_.at((head_ + i) % capacity_);
            #else
                out_data[i] = buffer_[(head_ + i) % capacity_];
            #endif
        }
        return samples_to_copy;
    }

    /**
     * @brief Non-blocking read (consumes data).
     * Reads up to num_samples_to_read from the buffer into out_data, removing them.
     * @param out_data Pointer to the buffer where the read data will be stored.
     * @param num_samples_to_read The maximum number of samples to read.
     * @return The number of samples actually read.
     */
    size_t try_read(T* out_data, size_t num_samples_to_read) {
        if (!is_valid_.load(std::memory_order_acquire)) return 0;
        std::lock_guard<std::mutex> lock(mutex_);
        size_t samples_to_read = std::min(num_samples_to_read, count_);
        for (size_t i = 0; i < samples_to_read; ++i) {
            #if DEBUG_CIRCBUFFER
                out_data[i] = buffer_.at((head_ + i) % capacity_);
            #else
                out_data[i] = buffer_[(head_ + i) % capacity_];
            #endif
        }
        head_ = (head_ + samples_to_read) % capacity_;
        count_ -= samples_to_read;
        space_available_cv_.notify_one(); // Notify a waiting writer
        return samples_to_read;
    }

    /**
     * @brief Reads data from the past (delay) without consuming it.
     * Fills with zeros if requested amount of data is not available at delay position.
     * Optimized for audio applications (e.g., delay lines).
     * @param out_data Pointer to output buffer.
     * @param num_samples Number of samples to read.
     * @param delay_samples Delay in samples from the most recent sample.
     * @return Number of actual (non-zero) samples copied to out_data.
     */
    size_t peek_with_delay_and_fill(T* out_data, size_t num_samples, size_t delay_samples) const {
        // Fill output buffer with zeros as default.
        std::fill(out_data, out_data + num_samples, T(0));
        
        if (!is_valid_.load(std::memory_order_acquire)) return 0;

        std::lock_guard<std::mutex> lock(mutex_);

        // If requested delay is greater than available data, nothing to read.
        if (delay_samples >= count_) {
            return 0;
        }

        size_t readable_samples = count_ - delay_samples;
        size_t samples_to_copy = std::min(num_samples, readable_samples);

        if (samples_to_copy == 0) return 0;
        
        // tail_ points to the *next* write location.
        // The most recent sample is at (tail_ - 1).
        // Starting read position is (tail_ - 1 - delay_samples).
        size_t read_head = (tail_ + capacity_ - 1 - delay_samples) % capacity_;

        for (size_t i = 0; i < samples_to_copy; ++i) {
            // Reading moves backward from read_head.
            size_t read_idx = (read_head + capacity_ - i) % capacity_;
            // Output buffer is filled front to back.
            size_t out_idx = samples_to_copy - 1 - i;
            #if DEBUG_CIRCBUFFER
                out_data[out_idx] = buffer_.at(read_idx);
            #else
                out_data[out_idx] = buffer_[read_idx];
            #endif
        }

        return samples_to_copy;
    }

    // ===================================================================
    // CONTROL AND STATE METHODS
    // ===================================================================

    /**
     * @brief Clears the buffer, resetting its state to empty.
     * Essential to call in Pd's _dsp to ensure clean initial state.
     * Does not deallocate memory, only resets internal pointers.
     */
    void clear() {
        if (!is_valid_.load(std::memory_order_acquire)) return;
        std::lock_guard<std::mutex> lock(mutex_);
        head_ = 0;
        tail_ = 0;
        count_ = 0;
    }

    /**
     * @brief Returns the number of samples currently available for reading.
     */
    size_t getSamplesAvailable() const {
        if (!is_valid_.load(std::memory_order_acquire)) return 0;
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }

    /**
     * @brief Returns the maximum buffer capacity.
     */
    size_t getCapacity() const {
        return capacity_;
    }

    /**
     * @brief Discards (removes) the oldest samples from the buffer.
     * @param num_samples Number of samples to discard from the beginning.
     * @return Number of samples actually discarded.
     */
    size_t discard(size_t num_samples) {
        if (!is_valid_.load(std::memory_order_acquire)) return 0;
        std::lock_guard<std::mutex> lock(mutex_);
        
        size_t samples_to_discard = std::min(num_samples, count_);
        head_ = (head_ + samples_to_discard) % capacity_;
        count_ -= samples_to_discard;
        
        // Notify a waiting writer, if any (for blocking operations).
        space_available_cv_.notify_one();
        return samples_to_discard;
    }

private:
    std::vector<T> buffer_;     // Underlying data storage.
    size_t capacity_;           // Maximum buffer capacity.
    size_t head_;               // Read pointer (index of oldest sample).
    size_t tail_;               // Write pointer (index of next write location).
    size_t count_;              // Number of samples currently in buffer.

    // Synchronization primitives
    mutable std::mutex mutex_;  // mutable allows locking in const methods.
    std::condition_variable data_available_cv_;
    std::condition_variable space_available_cv_;
    
    // Lifecycle safety flag
    std::atomic<bool> is_valid_;
};

// ... keep your Tensor interface methods here if needed ...
// They should use the base methods (write_overwrite, etc.) to operate.
// For example:
template <typename T>
bool write_tensor_overwrite(CircularBuffer<T>& buf, const torch::Tensor& tensor) {
    if (tensor.dim() != 1) {
        // Simplified to 1D for clarity
        throw std::invalid_argument("Tensor must be 1D for this example function.");
    }
    auto cpu_tensor = tensor.to(torch::kCPU).contiguous();
    size_t num_samples = cpu_tensor.size(0);
    const T* data_ptr = cpu_tensor.template data_ptr<T>();
    return buf.write_overwrite(data_ptr, num_samples);
}


} // namespace util_circbuffer
} // namespace core
} // namespace contorchionist

#endif // CORE_UTIL_CIRCBUFFER_H
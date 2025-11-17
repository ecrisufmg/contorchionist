#ifndef WINDOWING_H
#define WINDOWING_H

#include <vector>
#include <string>
#include <cmath> // For M_PI, std::pow, std::sin, std::cos, std::abs
#include "audio_utils.h" // For M_PI definition if not in cmath

namespace Windowing {

// Define M_PI if not already defined (common for <cmath>)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum class Type {
    RECTANGULAR,
    HANNING,
    TRIANGULAR,
    HAMMING,
    BLACKMAN,
    COSINE
};

enum class Alignment {
    LEFT,
    CENTER,
    RIGHT
};

/**
 * @brief Generates window samples into the provided buffer.
 * @param windowBuffer The vector to fill with window samples. Will be resized to windowSize.
 * @param windowSize The desired size of the window.
 * @param type The type of window to generate.
 * @param zeroPaddingSamples The number of samples to zero-pad within the window.
 * @param alignment The alignment of the non-zero part of the window when zero-padding is applied.
 */
void generateWindow(
    std::vector<float>& windowBuffer,
    int windowSize,
    Type type,
    int zeroPaddingSamples,
    Alignment alignment
);

// Helper functions for string to enum conversion
Type stringToWindowType(const std::string& strType);
Alignment stringToWindowAlignment(const std::string& strAlign);

// Helper functions for enum to string conversion (for dumping parameters)
std::string toString(Type type);
std::string toString(Alignment alignment);

// Basic logging/post mimic - can be expanded or made more sophisticated
#ifdef DEBUG_WINDOWING
#include <cstdio> // For fprintf, stderr
#define WINDOWING_POST(...) fprintf(stderr, "[Windowing] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n")
#else
#define WINDOWING_POST(...)
#endif

} // namespace Windowing

#endif // WINDOWING_H

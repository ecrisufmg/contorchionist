#include "windowing.h"
#include <algorithm> // For std::fill
#include <cmath>     // For std::pow, std::sin, std::cos, std::abs

// Define M_PI if not already defined (e.g. on Windows with MSVC)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Windowing {

Type stringToWindowType(const std::string& strType) {
    if (strType == "rect" || strType == "rectangular" || strType == "0") return Type::RECTANGULAR;
    if (strType == "hann" || strType == "hanning" || strType == "1") return Type::HANNING;
    if (strType == "tri" || strType == "triangular" || strType == "2") return Type::TRIANGULAR;
    if (strType == "hamm" || strType == "hamming" || strType == "3") return Type::HAMMING;
    if (strType == "black" || strType == "blackman" || strType == "4") return Type::BLACKMAN;
    if (strType == "cos" || strType == "cosine" || strType == "5") return Type::COSINE;
    WINDOWING_POST("Unknown window type string '%s', defaulting to RECTANGULAR", strType.c_str());
    return Type::RECTANGULAR;
}

Alignment stringToWindowAlignment(const std::string& strAlign) {
    if (strAlign == "l" || strAlign == "left" || strAlign == "0") return Alignment::LEFT;
    if (strAlign == "c" || strAlign == "center" || strAlign == "1") return Alignment::CENTER;
    if (strAlign == "r" || strAlign == "right" || strAlign == "2") return Alignment::RIGHT;
    WINDOWING_POST("Unknown window alignment string '%s', defaulting to LEFT", strAlign.c_str());
    return Alignment::LEFT;
}

void generateWindow(
    std::vector<float>& windowBuffer,
    int windowSize,
    Type type,
    int zeroPaddingSamples,
    Alignment alignment
) {
    if (windowSize <= 0) {
        windowBuffer.assign(windowSize, 0.0f); // or clear()
        WINDOWING_POST("Window size is %d, cannot generate window.", windowSize);
        return;
    }
    
    windowBuffer.assign(windowSize, 0.0f); // Initialize with zeros, also resizes

    int actualSignalSamples = windowSize - zeroPaddingSamples;
    int firstSampleIndex = 0;

    if (zeroPaddingSamples > 0 && zeroPaddingSamples < windowSize) {
        switch (alignment) {
            case Alignment::LEFT:
                firstSampleIndex = 0;
                // Zeros are implicitly at the end by initializing then filling start
                break;
            case Alignment::CENTER:
                firstSampleIndex = zeroPaddingSamples / 2;
                // Zeros are at start and end
                break;
            case Alignment::RIGHT:
                firstSampleIndex = zeroPaddingSamples;
                // Zeros are at the start
                break;
        }
    } else if (zeroPaddingSamples >= windowSize) { // All zeros
        WINDOWING_POST("Window is fully zero-padded (zp %d, size %d).", zeroPaddingSamples, windowSize);
        return; // Buffer is already all zeros
    } else { // No zero padding or invalid zeroPaddingSamples < 0
        actualSignalSamples = windowSize;
        firstSampleIndex = 0;
    }

    if (actualSignalSamples <= 0) {
        WINDOWING_POST("Effective window signal part is zero or negative samples (%d). Window will be all zeros.", actualSignalSamples);
        return; // Buffer is already all zeros
    }

    int lastSampleIndex = firstSampleIndex + actualSignalSamples;

    for (int i = firstSampleIndex; i < lastSampleIndex; ++i) {
        // k is the index within the 'active' part of the window (0 to actualSignalSamples-1)
        int k = i - firstSampleIndex; 
        
        // Prevent division by zero if actualSignalSamples is 1 (for some window types)
        float denominator = (actualSignalSamples == 1) ? 1.0f : static_cast<float>(actualSignalSamples - 1);
        if (denominator == 0) denominator = 1.0f; // Should be caught by actualSignalSamples == 1

        switch (type) {
            case Type::RECTANGULAR:
                windowBuffer[i] = 1.0f;
                break;
            case Type::HANNING:
                // (0.5 * (1 - cos(2*PI*k / (N-1)))) is equivalent to sin^2(PI*k / (N-1))
                windowBuffer[i] = std::pow(std::sin((M_PI * k) / denominator), 2.0f);
                break;
            case Type::TRIANGULAR: // Bartlett window
                 // Corrected triangular window: 1 - | (k - (N-1)/2) / ((N-1)/2) |
                windowBuffer[i] = 1.0f - std::abs((k - denominator / 2.0f) / (denominator / 2.0f));
                break;
            case Type::HAMMING:
                windowBuffer[i] = 0.54f - 0.46f * std::cos((2.0f * M_PI * k) / denominator);
                break;
            case Type::BLACKMAN:
                 windowBuffer[i] = 0.42f 
                                 - 0.5f * std::cos((2.0f * M_PI * k) / denominator) 
                                 + 0.08f * std::cos((4.0f * M_PI * k) / denominator);
                break;
            case Type::COSINE: // Sine window (half sine)
                windowBuffer[i] = std::sin((M_PI * k) / denominator);
                break;
        }
    }
    WINDOWING_POST("Window generated. Type: %d, Size: %d, ZP: %d, Align: %d", static_cast<int>(type), windowSize, zeroPaddingSamples, static_cast<int>(alignment));
}

std::string toString(Type type) {
    switch (type) {
        case Type::RECTANGULAR: return "rectangular";
        case Type::HANNING:     return "hanning";
        case Type::TRIANGULAR:  return "triangular";
        case Type::HAMMING:     return "hamming";
        case Type::BLACKMAN:    return "blackman";
        case Type::COSINE:      return "cosine";
        default:                return "unknown_window_type";
    }
}

std::string toString(Alignment alignment) {
    switch (alignment) {
        case Alignment::LEFT:   return "left";
        case Alignment::CENTER: return "center";
        case Alignment::RIGHT:  return "right";
        default:                return "unknown_alignment";
    }
}

} // namespace Windowing

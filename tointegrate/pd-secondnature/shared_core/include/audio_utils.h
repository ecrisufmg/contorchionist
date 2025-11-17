\
// filepath: /home/padovani/data/gdrive/meus_dev/audio_multi_sys/shared_core/include/audio_utils.h
#ifndef AUDIO_UTILS_H
#define AUDIO_UTILS_H

#include <string>
#include <algorithm> // For std::max

namespace AudioUtils {

/**
 * @brief Calculates the next power of two greater than or equal to n.
 * @param n The input number.
 * @param minVal The minimum value to return.
 * @return The next power of two, or minVal if the calculated power is less than minVal.
 */
int nextPowerOfTwo(int n, int minVal = 1);

// Define M_PI if not already defined (common for <cmath>)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Basic logging/post mimic - can be expanded or made more sophisticated
#ifdef DEBUG_AUDIO_UTILS
#include <cstdio> // For fprintf, stderr
#define AUDIO_UTILS_POST(...) fprintf(stderr, "[AudioUtils] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\\n")
#else
#define AUDIO_UTILS_POST(...)
#endif

} // namespace AudioUtils

#endif // AUDIO_UTILS_H

\
// filepath: /home/padovani/data/gdrive/meus_dev/audio_multi_sys/shared_core/src/audio_utils.cpp
#include "audio_utils.h"

namespace AudioUtils {

int nextPowerOfTwo(int n, int minVal) {
    if (n <= 0) return std::max(1, minVal); // Ensure at least 1 or minVal
    int power = 1;
    while (power < n) {
        power *= 2;
        if (power <= 0) { // Overflow protection
            return std::max(n, minVal); // Should not happen with typical int sizes for audio
        }
    }
    return std::max(power, minVal);
}

} // namespace AudioUtils

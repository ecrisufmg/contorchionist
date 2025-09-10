#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath> // For std::fabs

#include "unit_conversions.h" // The header with hzToMel and melToHz

// Helper function to check float/double equality with tolerance
template <typename T>
bool areAlmostEqual(T val1, T val2, T epsilon = static_cast<T>(1e-5)) {
    return std::fabs(val1 - val2) < epsilon;
}

void test_hz_to_mel_conversions() {
    std::cout << "Testing hzToMel conversions..." << std::endl;

    // Test with float
    float freq_f1 = 1000.0f;
    float mel_f1 = contorchionist::core::unit_conversions::hzToMel(freq_f1);
    // Expected value for 1000 Hz is approx 1000 mels (using the formula 2595 * log10(1 + 1000/700) )
    // 2595 * log10(1 + 1.42857) = 2595 * log10(2.42857) = 2595 * 0.38535 = 1000.00825
    assert(areAlmostEqual(mel_f1, 1000.00825f));
    std::cout << "  ✓ hzToMel(float): " << freq_f1 << " Hz -> " << mel_f1 << " Mels" << std::endl;

    float freq_f2 = 0.0f;
    float mel_f2 = contorchionist::core::unit_conversions::hzToMel(freq_f2);
    assert(areAlmostEqual(mel_f2, 0.0f));
    std::cout << "  ✓ hzToMel(float): " << freq_f2 << " Hz -> " << mel_f2 << " Mels" << std::endl;

    // Test with double
    double freq_d1 = 1000.0;
    double mel_d1 = contorchionist::core::unit_conversions::hzToMel(freq_d1);
    assert(areAlmostEqual(mel_d1, 1000.0082516228007));
    std::cout << "  ✓ hzToMel(double): " << freq_d1 << " Hz -> " << mel_d1 << " Mels" << std::endl;

    double freq_d2 = 250.0;
    double mel_d2 = contorchionist::core::unit_conversions::hzToMel(freq_d2);
    // 2595 * log10(1 + 250/700) = 2595 * log10(1 + 0.35714) = 2595 * log10(1.35714) = 2595 * 0.1326 = 344.109
    assert(areAlmostEqual(mel_d2, 344.109085));
    std::cout << "  ✓ hzToMel(double): " << freq_d2 << " Hz -> " << mel_d2 << " Mels" << std::endl;

    std::cout << "hzToMel conversions tests passed." << std::endl;
}

void test_mel_to_hz_conversions() {
    std::cout << "Testing melToHz conversions..." << std::endl;

    // Test with float
    float mel_f1 = 1000.00825f;
    float freq_f1 = contorchionist::core::unit_conversions::melToHz(mel_f1);
    assert(areAlmostEqual(freq_f1, 1000.0f));
    std::cout << "  ✓ melToHz(float): " << mel_f1 << " Mels -> " << freq_f1 << " Hz" << std::endl;

    float mel_f2 = 0.0f;
    float freq_f2 = contorchionist::core::unit_conversions::melToHz(mel_f2);
    assert(areAlmostEqual(freq_f2, 0.0f));
    std::cout << "  ✓ melToHz(float): " << mel_f2 << " Mels -> " << freq_f2 << " Hz" << std::endl;

    // Test with double
    double mel_d1 = 1000.0082516228007;
    double freq_d1 = contorchionist::core::unit_conversions::melToHz(mel_d1);
    assert(areAlmostEqual(freq_d1, 1000.0));
    std::cout << "  ✓ melToHz(double): " << mel_d1 << " Mels -> " << freq_d1 << " Hz" << std::endl;

    double mel_d2 = 344.109085;
    double freq_d2 = contorchionist::core::unit_conversions::melToHz(mel_d2);
    assert(areAlmostEqual(freq_d2, 250.0));
    std::cout << "  ✓ melToHz(double): " << mel_d2 << " Mels -> " << freq_d2 << " Hz" << std::endl;

    std::cout << "melToHz conversions tests passed." << std::endl;
}

void test_round_trip_conversions() {
    std::cout << "Testing round-trip conversions (Hz -> Mel -> Hz)..." << std::endl;

    // Test with float
    float original_freq_f = 440.0f;
    float mel_f = contorchionist::core::unit_conversions::hzToMel(original_freq_f);
    float round_trip_freq_f = contorchionist::core::unit_conversions::melToHz(mel_f);
    assert(areAlmostEqual(original_freq_f, round_trip_freq_f, 0.01f)); // Looser tolerance for round trip
    std::cout << "  ✓ Round-trip (float): " << original_freq_f << " Hz -> " << mel_f << " Mels -> " << round_trip_freq_f << " Hz" << std::endl;

    // Test with double
    double original_freq_d = 880.0;
    double mel_d = contorchionist::core::unit_conversions::hzToMel(original_freq_d);
    double round_trip_freq_d = contorchionist::core::unit_conversions::melToHz(mel_d);
    assert(areAlmostEqual(original_freq_d, round_trip_freq_d, 0.0001)); // Looser tolerance for round trip
    std::cout << "  ✓ Round-trip (double): " << original_freq_d << " Hz -> " << mel_d << " Mels -> " << round_trip_freq_d << " Hz" << std::endl;
    
    std::vector<double> test_freqs_d = {0.0, 100.0, 1000.0, 4000.0, 8000.0, 15000.0};
    for (double freq : test_freqs_d) {
        double m = contorchionist::core::unit_conversions::hzToMel(freq);
        double f_rt = contorchionist::core::unit_conversions::melToHz(m);
        assert(areAlmostEqual(freq, f_rt, 0.0001));
        std::cout << "  ✓ Round-trip (double): " << freq << " Hz -> " << m << " Mels -> " << f_rt << " Hz" << std::endl;
    }


    std::cout << "Round-trip conversions tests passed." << std::endl;
}


int main() {
    std::cout << "=== Contorchionist Core - Audio Unit Conversions Test ===" << std::endl;

    test_hz_to_mel_conversions();
    std::cout << std::endl;
    test_mel_to_hz_conversions();
    std::cout << std::endl;
    test_round_trip_conversions();
    std::cout << std::endl;

    std::cout << "All audio unit conversion tests completed successfully!" << std::endl;
    return 0;
}

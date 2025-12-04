# torch.mic_array_spectral_analyzer~

Microphone array spectral analyzer for spatial sound source localization using LibTorch.

## Description

`torch.mic_array_spectral_analyzer~` analyzes signals from a microphone array to determine the direction of arrival (DOA) and strength of sound sources in specified frequency bands. It uses a vector intensity-like approach combined with spectral masking to isolate and track sources.

It is designed to work with frequency domain signals (magnitude) provided by upstream RFFT objects (e.g., `torch.rfft~`).

## Creation

```pd
[torch.mic_array_spectral_analyzer~ -mics 4 -device mps ...]
```

### Flags and Arguments

*   **`-mics <int>`**, **`-m`**: Number of microphones in the array. Default: 4.
*   **`-device <string>`**, **`-d`**: Torch device to use (`cpu`, `cuda`, `mps`). Default: `cpu`.
*   **`-verbose`**, **`-v`**: Enable verbose logging.
*   **`-mag`**: Set strength calculation mode to Magnitude (Linear).
*   **`-pow`**: Set strength calculation mode to Power (Squared Magnitude). Default.
*   **`-level <string>`**, **`-l`**, **`-rms`**: Set overall level calculation mode. Options: `mag` (Linear), `pow` (Power), `db` (Decibels).

## Inlets and Outlets

### Inlets

1.  **Inlet 0 (Signal/Control)**: Magnitude input for Microphone 1. Also accepts control messages.
2.  **Inlet 1...N-1 (Signal)**: Magnitude inputs for Microphones 2 to N.

### Outlets

1.  **Outlet 0 (Control)**: Outputs a list of analysis results for each band.

## Output Format

The object outputs a list of 5 float values for each analyzed band:

```
<band_index> <angle_deg> <strength> <overall_level> <dominant_freq_hz>
```

*   **band_index**: Index of the frequency band (0-based).
*   **angle_deg**: Estimated direction of arrival in degrees (0-360).
*   **strength**: Strength of the directional component (0.0 - 1.0+). Depends on `strength_mode`.
*   **overall_level**: Overall signal level in the band (average across mics). Depends on `level_mode`.
*   **dominant_freq_hz**: Frequency of the dominant peak within the band.

## Methods

*   **`mic <id> <angle> <dist>`**: Set the geometry for a specific microphone.
    *   `id`: Microphone index (0 to N-1).
    *   `angle`: Physical angle of the microphone capsule in degrees.
    *   `dist`: Distance from center (currently unused/reserved).
*   **`speaker <angle>`**: Register a speaker position (in degrees) for auto-calibration.
*   **`calibrate`**: Perform calibration. Snaps microphone "tails" (opposite of capsule) to the nearest registered speaker position to determine precise pickup angles.
*   **`bands <min1> <max1> <min2> <max2> ...`**: Define frequency bands for analysis. Requires pairs of frequencies (min Hz, max Hz).
    *   Example: `bands 100 500 1000 2000` defines two bands: 100-500Hz and 1000-2000Hz.
*   **`overlap <factor>`**, **`of <factor>`**: Set the overlap factor used in the upstream FFT (e.g., 4). Used to normalize input magnitudes if they are summed/overlapped.
*   **`strength_mode <mode>`**: Set the mode for the "strength" output.
    *   `mag` or `magnitude`: Linear magnitude.
    *   `pow` or `power`: Squared magnitude (Power).
*   **`level_mode <mode>`**: Set the mode for the "overall_level" output.
    *   `mag` or `magnitude`: Linear magnitude (RMS).
    *   `pow` or `power`: Mean Square (Power).
    *   `db`: Decibels.

## Default Geometry (4 Mics)

If created with 4 microphones, the object defaults to an "Inverted Star" configuration:
*   Mic 0: -135° (Back Left)
*   Mic 1: +135° (Back Right)
*   Mic 2: -45° (Front Left)
*   Mic 3: +45° (Front Right)

This assumes microphones are pointing *outwards* or are placed in a circle. The calibration logic assumes "Null Point Steering" where the rear of the mic (tail) points to a speaker.

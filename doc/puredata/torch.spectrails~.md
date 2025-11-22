# torch.spectrails~

Spectral trails processor based on LibTorch (Mono version).

## Description

`torch.spectrails~` is a spectral processor that detects peaks in the spectrum and creates "trails" (sustained spectral components) based on an envelope follower. It performs peak detection, parabolic interpolation, and manages the lifecycle of spectral peaks (attack, sustain, decay). It also provides control data output for detected peaks.

This is the mono version of the processor. For Ambisonic signals, use `torch.amb.spectrails~`.

## Creation

```pd
[torch.spectrails~ -overlap 4 ...]
```

### Flags and Arguments

*   **`-overlap <int>`**, **`-of`**: Overlap factor. Default: 4.
*   **`-threshold <float>`**, **`-thresh`**: Linear amplitude threshold for peak detection. Default: 0.01.
*   **`-attack <float>`**, **`-att`**: Envelope attack rate (0.0 - 1.0). Default: 0.7.
*   **`-attackms <float>`**, **`-attms`**: Attack time in milliseconds.
*   **`-attacks <float>`**, **`-atts`**: Attack time in seconds.
*   **`-decay <float>`**, **`-dec`**: Envelope decay rate (0.0 - >1.0). Values > 1.0 create positive feedback. Default: 0.999.
*   **`-decayms <float>`**: Decay time in milliseconds to reach `floordb` (or -60dB if floor is unset).
*   **`-decays <float>`**: Decay time in seconds to reach `floordb`.
*   **`-decay6dbms <float>`**: Decay time in milliseconds to drop 6dB.
*   **`-decay6dbs <float>`**: Decay time in seconds to drop 6dB.
*   **`-min_peak_distance <float>`**, **`-mindist`**, **`-mpd`**: Minimum distance between peaks in Hz. Default: 100.0.
*   **`-pregaindb <float>`**, **`-pregain`**: Input gain in dB applied before processing. Default: 0.0.
*   **`-gaindb <float>`**, **`-gain`**, **`-g`**: Output gain in dB. Default: 0.0.
*   **`-mode <int>`**: Detection mode. `0` = Slope-based (default), `1` = Prominence-based.
*   **`-prominence <float>`**, **`-prom`**: Prominence threshold (0.0 - 1.0) for mode 1. Default: 0.6.
*   **`-max_peaks <int>`**, **`-maxpeaks`**, **`-mp`**: Maximum number of simultaneous peaks allowed. 0 = unlimited. Default: 0.
*   **`-floordb <float>`**, **`-floor`**: Noise floor in dB for release gate. Peaks below this level are forced to decay. Default: -150.0 (disabled).
*   **`-limiter <float>`**, **`-lim`**, **`-l`**: Enable limiter with specific threshold in dB. Default: Enabled at 0dB if not specified.
*   **`-nolimiter`**, **`-nolim`**: Disable the limiter at startup.
*   **`-midi`**, **`-m`**: Output frequency as MIDI note numbers in the control outlet.
*   **`-velocity`**, **`-vel`**, **`-v`**: Output magnitude as MIDI velocity (0-127). Can be `log` (default), `linear`, or `off`.
*   **`-device <string>`**, **`-d`**: Torch device to use (`cpu`, `cuda`, `mps`). Default: `cpu`.
*   **`-verbose`**, **`-v`**: Enable verbose logging.

## Inlets and Outlets

### Inlets

1.  **Inlet 0 (Signal)**: Magnitude input.
2.  **Inlet 1 (Signal)**: Phase input.

### Outlets

1.  **Outlet 0 (Signal)**: Processed Magnitude.
2.  **Outlet 1 (Signal)**: Processed Phase.
3.  **Outlet 2 (Control)**: Outputs a list for each active peak in the format:
    `list <rank> <freq> <mag> <state>`

    *   **rank**: Index of the peak (sorted by magnitude).
    *   **freq**: Frequency in Hz (or MIDI note if `-midi` is active).
    *   **mag**: Magnitude in dB (or Velocity 0-127 if `-velocity` is active).
    *   **state**: `1` (new), `0` (sustained), `-1` (decayed).

## Messages

*   **`threshold <float>`**, **`thresh`**: Set detection threshold (linear).
*   **`thresholddb <float>`**, **`threshdb`**: Set detection threshold in dB.
*   **`attack <float>`**, **`att`**: Set envelope attack rate.
*   **`attackms <float>`**, **`attms`**: Set attack time in milliseconds.
*   **`attacks <float>`**, **`atts`**: Set attack time in seconds.
*   **`decay <float>`**, **`dec`**: Set envelope decay rate.
*   **`decayms <float>`**: Set decay time in milliseconds to reach `floordb`.
*   **`decays <float>`**: Set decay time in seconds to reach `floordb`.
*   **`decay6dbms <float>`**: Set decay time in milliseconds to drop 6dB.
*   **`decay6dbs <float>`**: Set decay time in seconds to drop 6dB.
*   **`overlap <int>`**: Set overlap factor.
*   **`min_peak_distance <float>`**: Set minimum peak distance in Hz.
*   **`pregaindb <float>`**, **`pregain`**: Set input gain in dB.
*   **`gaindb <float>`**, **`gain`**: Set output gain in dB.
*   **`mode <int>`**: Set detection mode (0 or 1).
*   **`prominence <float>`**: Set prominence threshold.
*   **`max_peaks <int>`**, **`maxpeaks`**: Set maximum number of peaks.
*   **`floordb <float>`**, **`floor`**: Set noise floor in dB.
*   **`limiter <arg>`**: Configure limiter.
    *   `limiter off`, `limiter false`: Disable limiter.
    *   `limiter <float>`: Enable limiter with threshold in dB (e.g., `limiter -3`).
*   **`nolimiter`**, **`nolim`**: Disable limiter.
*   **`midi <0/1>`**: Enable/disable MIDI note output format.
*   **`velocity <arg>`**, **`vel`**: Configure velocity output format.
    *   `velocity log`: Logarithmic mapping (dB to 0-127).
    *   `velocity linear`: Linear mapping.
    *   `velocity off`: Output raw dB.
*   **`reset`**: Clear internal memory buffers.

## Details

### Limiter and Feedback
The object features a built-in limiter that is enabled by default (0dB threshold). This is crucial when using `decay` values > 1.0, which create positive feedback loops. The limiter clamps both the output signal and the internal state to prevent infinite growth (NaN/Inf) and allows the system to remain stable even under heavy feedback.

### Detection Modes
*   **Slope-based (0)**: Detects peaks based on the change in slope of the spectral magnitude. Good for general usage.
*   **Prominence (1)**: Uses a prominence threshold to identify significant peaks relative to their surroundings. Better for noisy signals or complex spectra.

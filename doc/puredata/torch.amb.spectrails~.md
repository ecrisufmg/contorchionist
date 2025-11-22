# torch.amb.spectrails~

Ambisonic spectral trails processor based on LibTorch.

## Description

`torch.amb.spectrails~` is a spectral processor that detects peaks in the spectrum and creates "trails" (sustained spectral components) based on an envelope follower. It is designed for Ambisonic signals, maintaining phase coherence across channels by synchronizing the envelope detection on the omnidirectional (W) channel and applying it to all other channels.

It performs peak detection, parabolic interpolation, and manages the lifecycle of spectral peaks (attack, sustain, decay). It also provides control data output for detected peaks.

## Creation

```pd
[torch.amb.spectrails~ -order 1 -fftsize 1024 ...]
```

### Flags and Arguments

*   **`-order <int>`**, **`-ord`**, **`-o`**: Ambisonic order. Determines the number of channels $B = (N+1)^2$. Default: 1 (4 channels).
*   **`-fftsize <int>`**, **`-fft`**, **`-n`**: FFT size. Default: 1024.
*   **`-overlap <int>`**, **`-of`**: Overlap factor. Default: 4.
*   **`-threshold <float>`**, **`-thresh`**: Linear amplitude threshold for peak detection. Default: 0.01.
*   **`-attack <float>`**, **`-att`**: Envelope attack rate (0.0 - 1.0). Default: 0.7.
*   **`-decay <float>`**, **`-dec`**: Envelope decay rate (0.0 - >1.0). Values > 1.0 create positive feedback. Default: 0.999.
*   **`-decaytime <float>`**, **`-decayt`**, **`-dtime`**: Decay time in seconds (alternative to setting decay factor directly).
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

The object creates dynamic inlets based on the Ambisonic order.
Total inlets = `(order + 1)^2 * 2`.

1.  **Inlet 0 (Signal)**: Channel 0 (W) Magnitude.
2.  **Inlet 1 (Signal)**: Channel 0 (W) Phase.
3.  **Inlet 2 (Signal)**: Channel 1 (Y) Magnitude.
4.  **Inlet 3 (Signal)**: Channel 1 (Y) Phase.
... and so on for all channels.

### Outlets

The object creates dynamic signal outlets followed by one control outlet.

1.  **Signal Outlets**: Pairs of Magnitude and Phase for each channel, processed with the spectral trails effect.
    *   Out 0: Ch 0 Mag
    *   Out 1: Ch 0 Phase
    *   ...
2.  **Control Outlet (Rightmost)**: Outputs a list for each active peak in the format:
    `list <rank> <freq> <mag> <state>`

    *   **rank**: Index of the peak (sorted by magnitude).
    *   **freq**: Frequency in Hz (or MIDI note if `-midi` is active).
    *   **mag**: Magnitude in dB (or Velocity 0-127 if `-velocity` is active).
    *   **state**: `1` (new), `0` (sustained), `-1` (decayed).

## Messages

*   **`threshold <float>`**, **`thresh`**: Set detection threshold (linear).
*   **`thresholddb <float>`**, **`threshdb`**: Set detection threshold in dB.
*   **`attack <float>`**, **`att`**: Set envelope attack rate.
*   **`decay <float>`**, **`dec`**: Set envelope decay rate.
*   **`decaytime <float>`**, **`decayt`**, **`dtime`**, **`dt`**: Set decay time in seconds.
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

### Ambisonic Sync
The spectral envelope is detected solely on the first channel (W - Omnidirectional). This envelope is then applied to all other channels (X, Y, Z, etc.) to preserve the spatial image and phase relationships of the Ambisonic signal.

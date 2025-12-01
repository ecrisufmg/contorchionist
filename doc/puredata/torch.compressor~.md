# torch.compressor~

Dynamic range compressor with lookahead and auto-makeup gain, powered by LibTorch.

## Description

`torch.compressor~` is a high-quality dynamic range compressor designed for Pure Data. It utilizes a C++ core with LibTorch tensors for RMS detection and signal processing. It features standard compression controls (Threshold, Ratio, Attack, Release) as well as advanced features like variable Knee/Curve, Lookahead for transient preservation, and Automatic Makeup Gain.

## Class
`torch.compressor~`

## Creation Arguments

You can initialize the object with flags to set initial values.

*   `@thresh <float>` or `@threshold <float>`: Threshold in dB (default: -60.0).
*   `@ratio <float>`: Compression ratio (default: 2.0).
*   `@curve <float>` or `@mode <float>`: Knee curve characteristic (default: 0.0).
*   `@attack <float>` or `@att <float>`: Attack time in milliseconds (default: 10.0).
*   `@release <float>` or `@rel <float>`: Release time in milliseconds (default: 100.0).
*   `@window <int>` or `@win <int>`: RMS window size in samples (default: 1024).
*   `@lookahead <float>` or `@look <float>`: Lookahead time in milliseconds (default: 0.0).
*   `@makeup <float>` or `@gain <float>`: Manual makeup gain in dB (default: 0.0).
*   `@auto <0/1>` or `@automakeup <0/1>`: Enable automatic makeup gain (default: 0).

**Example:**
```pd
[torch.compressor~ @thresh -20 @ratio 4 @attack 5 @lookahead 2 @auto 1]
```

## Inlets

*   **Left Inlet**: Audio signal input.

## Outlets

*   **Left Outlet**: Compressed audio signal output.

## Methods

### thresh / threshold
Sets the threshold level in decibels (dB). Signals above this level will be compressed.
*   **Usage**: `[thresh -20(`

### ratio
Sets the compression ratio (e.g., 4.0 means 4:1).
*   **Usage**: `[ratio 4(`

### attack / att
Sets the attack time in milliseconds. Determines how fast the compressor reacts to signals exceeding the threshold.
*   **Usage**: `[attack 10(`

### release / rel
Sets the release time in milliseconds. Determines how fast the compressor returns to unity gain after the signal falls below the threshold.
*   **Usage**: `[release 100(`

### curve / mode
Adjusts the knee characteristic of the compression curve.
*   `0`: Hard knee (Linear).
*   `> 0`: Ease In (Exponential).
*   `< 0`: Ease Out (Inverse).
*   **Usage**: `[curve 0.5(`

### window / win
Sets the size of the RMS detection window in samples. Larger windows result in smoother detection but slower response.
*   **Usage**: `[window 1024(`

### lookahead / look
Sets the lookahead time in milliseconds. This delays the audio signal relative to the sidechain, allowing the compressor to react to transients before they occur at the output. Useful for catching fast percussive attacks.
*   **Usage**: `[lookahead 5(`

### makeup / gain
Sets a manual makeup gain in decibels (dB) applied to the output signal.
*   **Usage**: `[makeup 6(`

### auto / automakeup
Enables or disables Automatic Makeup Gain. When enabled, the compressor calculates the theoretical gain reduction at 0dB input based on the current Threshold and Ratio, and applies the inverse gain to compensate.
*   `0`: Disable.
*   `1`: Enable.
*   **Usage**: `[auto 1(`

## DSP Details

*   **RMS Detection**: Uses a sliding window average of squared values (calculated via LibTorch tensors on CPU) to determine signal level.
*   **Gain Computer**: Standard downward compression logic.
*   **Lookahead**: Implemented via a circular buffer delay line on the audio path. The sidechain path is not delayed.

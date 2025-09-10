# Pure Data Externals for Contorchionist

This directory contains Pure Data externals that provide audio processing functionality using the contorchionist library with PyTorch backend.

## Available Externals

### torch.melspectrogram~

Computes mel-frequency spectrograms from audio input using PyTorch backend.

#### Usage
```
torch.melspectrogram~ [arguments]
```

#### Arguments (all optional)

**FFT Parameters:**
- `-n_fft <int>`: FFT size (default: 2048)
- `-hop_length <int>`: Hop size in samples (default: 512)
- `-win_length <int>`: Window length in samples (default: n_fft)
- `-window <string>`: Window type (default: hann)
  - Available: `hann`, `hamming`, `blackman`, `bartlett`, `triangular`, `rectangular`

**Mel Parameters:**
- `-n_mels <int>`: Number of mel frequency bands (default: 128)
- `-fmin <float>`: Minimum frequency in Hz (default: 0)
- `-fmax <float>`: Maximum frequency in Hz (default: sample_rate/2)

**Mel Scale Options:**
- `-htk <0|1>`: Use HTK mel scale instead of Slaney (default: 0)
- `-flucoma <0|1>`: Use calc2 mel scale instead of Slaney (default: 0)
- `-mel_norm <string>`: Mel filterbank normalization method (default: slaney)

**RFFT Normalization:**
- `-rfft_norm <string>`: RFFT normalization mode (default: power_sum_squares)
  - Available modes:
    - `none`: No normalization
    - `backward`: Backward normalization (1/n)
    - `forward`: Forward normalization (1/sqrt(n))
    - `ortho`: Orthogonal normalization (1/sqrt(n) for forward, 1/sqrt(n) for backward)
    - `window`: Window-based normalization
    - `hann`, `hamming`, `blackman`, `bartlett`, `triangular`, `rectangular`: Window-specific normalizations
    - `power_sum_squares`: Power spectrogram with window sum of squares normalization
    - `psd_sum_squares`: Power spectral density with window sum of squares normalization

**Device:**
- `-device <string>`: Compute device (default: cpu)
  - Available: `cpu`, `cuda` (if available)

#### Output
The external outputs a list containing the mel spectrogram values for each frame. The list length is `n_mels` × number_of_frames.

#### Examples
```
# Basic usage with default parameters
torch.melspectrogram~

# Custom FFT size and mel bands
torch.melspectrogram~ -n_fft 4096 -n_mels 80

# Use HTK mel scale with specific frequency range
torch.melspectrogram~ -htk 1 -fmin 80 -fmax 8000

# Use calc2 mel scale with custom normalization
torch.melspectrogram~ -flucoma 1 -rfft_norm ortho

# GPU computation (if CUDA available)
torch.melspectrogram~ -device cuda
```

## Building

The Pure Data externals are built as part of the main contorchionist project. See the main README for build instructions.

## Notes

- All externals require PyTorch (libtorch) to be available
- CUDA support depends on PyTorch CUDA availability
- The mel scale options (HTK, calc2, Slaney) provide different frequency-to-mel conversions for compatibility with various audio processing libraries
- RFFT normalization modes affect the scaling of the frequency domain representation and should be chosen based on your specific analysis needs

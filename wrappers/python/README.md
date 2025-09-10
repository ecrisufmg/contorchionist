# pycontorchionist

Python bindings for the Contorchionist Audio Processing Library.

## Features

- High-performance mel-spectrogram computation with LibTorch backend
- Real-time audio processing capabilities
- Multiple normalization modes and mel-scale formulas
- Seamless integration with PyTorch ecosystem

## Installation

```bash
pip install pycontorchionist
```

For development installation:
```bash
pip install -e .
```

## Quick Start

```python
import pycontorchionist as cc
import numpy as np
import torch

# Create a mel-spectrogram processor
processor = cc.MelSpectrogramProcessor()

# Configure parameters
processor.set_sample_rate(44100.0)
processor.set_n_mels(128)
processor.set_n_fft(2048)
processor.set_hop_length(512)

# Process audio data
audio_data = np.random.randn(1024).astype(np.float32)
mel_output = processor.process(audio_data)

if mel_output is not None:
    print(f"Mel spectrogram shape: {mel_output.shape}")
```

## Dependencies

- torch >= 2.0.0
- torchaudio >= 2.0.0  
- numpy >= 1.20.0

## License

MIT License

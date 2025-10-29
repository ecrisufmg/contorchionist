# Waveshaper Effect - Neural Audio Processing Example

This example demonstrates how to train a neural network to emulate an audio effect that processes **64-sample blocks** at a time. This is a realistic audio processing scenario similar to real-time audio applications.

## Overview

The example implements a **waveshaper effect** with three controllable parameters:
- **Drive**: Distortion amount (0 = clean, 1 = heavy saturation)
- **Tone**: Brightness control (0 = dark, 0.5 = flat, 1 = bright)
- **Mix**: Dry/wet blend (0 = bypass, 1 = fully processed)

### Effect Pipeline

```
Input Audio Block (64 samples)
    ↓
1. Drive: gain = 1 + drive × 9  (amplify 1x to 10x)
    ↓
2. Waveshaping: tanh(signal)  (soft saturation)
    ↓
3. Tone Control: signal + tone × diff(signal)  (treble boost/cut)
    ↓
4. Mix: wet × processed + (1 - wet) × dry
    ↓
Output Audio Block (64 samples)
```

## Files

### Training and Testing Scripts

1. **`01_Waveshaper_train_model.py`** - Main training script
   - Trains a neural network to emulate the target waveshaper function
   - Generates 5000 synthetic training examples (50% isolated blocks, 50% consecutive blocks)
   - Saves the trained model as `waveshaper.ts` (TorchScript format)
   - **Runtime**: ~5-10 minutes on CPU

2. **`02_Waveshaper_simpletest_model.py`** - Unit tests
   - Tests the model with 9 different scenarios
   - Compares model output vs. mathematical target
   - Reports MSE, MAE, and max error for each test case

3. **`03_Waveshaper_audiotest_model.py`** - Audio quality test
   - Generates 5-second WAV files for comparison
   - Processes continuous audio block-by-block
   - Creates three files:
     - `input_audio.wav` - Original signal
     - `target_waveshaper.wav` - Mathematical target output
     - `model_waveshaper.wav` - Neural network output

4. **`04_Waveshaper_debug_blocks.py`** - Debugging tool
   - Analyzes errors across consecutive blocks
   - Useful for diagnosing block-boundary issues

### Model File

- **`waveshaper.ts`** - Trained TorchScript model
  - Input: 67 values `[sample_0, ..., sample_63, drive, tone, mix]`
  - Output: 64 values `[processed_0, ..., processed_63]`
  - Can be loaded in PureData, Max/MSP, or other environments

## Usage

### 1. Training the Model

```bash
conda activate pytorch313  # or your PyTorch environment
python 01_Waveshaper_train_model.py
```

**Expected output:**
```
Training Waveshaper Effect (64-sample blocks)
======================================================================
1. Generating synthetic data...
   (50% isolated blocks, 50% consecutive blocks from longer signals)
   ... 5000/5000 examples generated
   
2. Training...
   Epoch [500/500], Loss: 0.001745
   
3. Training completed! Best loss: 0.001668

4. Testing with a 440Hz sine wave block
   Mean Absolute Error: 0.035330
   
5. Saving model as TorchScript...
   ✓ Model saved as: waveshaper.ts
```

### 2. Testing the Model

**Unit tests (quick):**
```bash
python 02_Waveshaper_simpletest_model.py
```

**Audio quality test (generates WAV files):**
```bash
python 03_Waveshaper_audiotest_model.py
```

**Expected audio test results:**
- **MAE**: ~0.04 (very good accuracy)
- **Max Error**: ~0.7
- **Quality**: ✅ Very good - subtle difference between target and model

### 3. Using the Model in Code

```python
import torch
import numpy as np

# Load the trained model
model = torch.jit.load('waveshaper.ts')
model.eval()

# Prepare input: 64 audio samples + 3 parameters
audio_block = np.random.randn(64) * 0.5  # 64 samples
drive = 0.7  # Distortion amount
tone = 0.6   # Brightness
mix = 0.9    # Dry/wet blend

# Concatenate into input vector
input_vec = np.concatenate([audio_block, [drive, tone, mix]])
input_tensor = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0)

# Process through model
with torch.no_grad():
    output = model(input_tensor)[0].numpy()  # 64 processed samples

print(f"Input range: [{audio_block.min():.3f}, {audio_block.max():.3f}]")
print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
```

## Technical Details

### Model Architecture

```
Input: (batch, 67)
   ↓
Linear(67 → 256) + SiLU
   ↓
Linear(256 → 256) + SiLU
   ↓
Linear(256 → 128) + SiLU
   ↓
Linear(128 → 64)
   ↓
tanh(·) × 1.2  (output bounding)
   ↓
Output: (batch, 64)
```

- **Parameters**: ~117K
- **Activation**: SiLU (Swish)
- **Output**: Bounded by tanh to prevent extreme values

### Training Strategy

**Key insight**: The model must handle **continuous audio**, not just isolated blocks!

- **Training data**: 5000 examples
  - 50% isolated random blocks (for diversity)
  - 50% consecutive blocks extracted from longer signals (for continuity)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-5)
- **Batch size**: 64
- **Epochs**: Up to 500 (with early stopping, patience=50)
- **Loss**: MSE (Mean Squared Error)

### Why This Strategy Works

**Problem**: Early versions trained only on isolated random blocks achieved low error on individual blocks (MAE ~0.02) but **terrible error on continuous audio** (MAE ~0.24).

**Root cause**: In continuous audio, consecutive blocks have correlated values. Training only on independent blocks doesn't teach the model to handle this continuity.

**Solution**: Include blocks extracted from longer signals in training data. This teaches the model realistic audio patterns.

**Results**:
- Isolated blocks test: MAE ~0.035 ✅
- Continuous audio test: MAE ~0.041 ✅
- **6x improvement** in continuous audio quality!

### Stateless Design

The tone control function is **completely stateless** within each 64-sample block:

```python
def apply_tone_control(signal, tone):
    # First-order difference (high-frequency emphasis)
    diff = np.zeros_like(signal)
    diff[1:] = signal[1:] - signal[:-1]
    diff[0] = 0  # No dependency on previous block
    
    # Mix in the difference signal
    tone_mapped = (tone - 0.5) * 2.0  # Map 0..1 to -1..1
    output = signal + tone_mapped * 0.3 * diff
    
    return output
```

**Why stateless?** IIR filters (with memory) create discontinuities at block boundaries when blocks are processed independently. Using only operations within the current block ensures smooth continuous processing.

## Performance Metrics

### Single Block Tests (02_Waveshaper_simpletest_model.py)

| Test Scenario | MAE | Max Error | Quality |
|--------------|-----|-----------|---------|
| 440Hz sine, light drive | 0.028 | 0.086 | ✅ Excellent |
| 1kHz sine, heavy drive | 0.050 | 0.223 | ✅ Very Good |
| 220Hz sine, bright | 0.029 | 0.161 | ✅ Excellent |
| 2kHz sine, 50% mix | 0.037 | 0.151 | ✅ Very Good |
| Multi-tone complex | 0.057 | 0.222 | ✅ Very Good |

### Continuous Audio Test (03_Waveshaper_audiotest_model.py)

- **Duration**: 5 seconds @ 48kHz (240,000 samples)
- **MSE**: 0.003
- **MAE**: 0.041
- **Max Error**: 0.72
- **Spectral Difference**: 8.5 dB
- **Quality**: ✅ Very good - subtle difference

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- SciPy (for audio file I/O)

```bash
conda create -n pytorch313 python=3.11
conda activate pytorch313
conda install pytorch torchvision torchaudio -c pytorch
pip install scipy
```

## Integration with PureData/Max/MSP

The trained model (`waveshaper.ts`) can be loaded in external audio environments:

1. **PureData**: Use `torch.ts~` object (from contorchionist)
2. **Max/MSP**: Use `torch.ts~` external
3. **SuperCollider**: Use TorchUGen

**Expected latency**: 64 samples @ 48kHz = **1.33ms**

## Troubleshooting

### Model outputs extreme values

- Check that input audio is normalized (typical range: -1 to 1)
- Verify parameters are in correct range (drive, tone, mix: 0 to 1)
- The model uses `tanh` output activation to bound values to ~[-1.2, 1.2]

### High error on continuous audio

- Retrain the model to ensure it includes consecutive blocks in training data
- Check that the tone control function is truly stateless (no IIR filters)
- Use debug script `04_Waveshaper_debug_blocks.py` to analyze block-by-block errors

### Training is slow

- Reduce `num_examples` from 5000 to 2000 (faster but slightly less accurate)
- Use GPU if available (modify `train_model()` to use `.cuda()`)
- Reduce `num_epochs` to 200 (model often converges early)

## Key Learnings

1. **Block-based processing is realistic** for real-time audio applications
2. **Training data must match test conditions**: include consecutive blocks for continuous audio
3. **Stateless operations** (within blocks) avoid discontinuities at boundaries
4. **Simple architectures** (fully connected) can work well for audio effects
5. **Output activation** (tanh) is important to prevent unbounded values

## Future Improvements

- [ ] Add more effect types (reverb, delay, compression)
- [ ] Implement true IIR filter state handling across blocks
- [ ] Add time-varying parameters (modulation)
- [ ] Optimize model size for embedded systems
- [ ] Create PureData/Max patch examples

## References

- Original example: `example_nonlinosc_model` (single-sample processing)
- TorchScript documentation: https://pytorch.org/docs/stable/jit.html
- Neural audio effects: https://github.com/csteinmetz1/micro-tcn

## License

This example is part of the **contorchionist** project.

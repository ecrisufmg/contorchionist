# PureData Patch Structure for Waveshaper Effect

## Overview

The `waveshaper_effect.pd` patch demonstrates how to use the trained neural waveshaper model with the `torch.ts~` object in PureData.

## Patch Components

### 1. Audio Input/Output Chain

```
[adc~]                         # Audio input (stereo)
   ↓
[*~ 0.5]                       # Attenuate input
   ↓
[snake~ in 64]                 # Convert stream to 64-sample blocks
   ↓
[torch.ts~ waveshaper.ts -m forward]   # Neural model processing
   ↓
[snake~ out 64]                # Convert blocks back to stream
   ↓
[*~ 0.8]                       # Output gain control
   ↓
[dac~]                         # Audio output (stereo)
```

### 2. The torch.ts~ Object

**Format:**
```
[torch.ts~ MODEL_FILE -m METHOD]
```

**For our example:**
```
[torch.ts~ waveshaper.ts -m forward]
```

- **First argument**: Path to the `.ts` model file
- **`-m` flag**: Specifies the model method to call (default: `forward`)
- **`-mc` flag**: Multi-channel mode (not used here)
- **Input conversion**: Use `[snake~ in BLOCKSIZE]` to convert audio stream to blocks
- **Output conversion**: Use `[snake~ out BLOCKSIZE]` to convert blocks back to audio
- **Inlets**: 
  - Inlet 0: Audio input blocks (from `snake~ in 64`)
  - Inlet 1: Drive parameter (0-1)
  - Inlet 2: Tone parameter (0-1)
  - Inlet 3: Mix parameter (0-1)
- **Outlet**: Processed audio blocks (to `snake~ out 64`)

### 3. Parameter Controls

Three horizontal sliders control the effect parameters:

#### Drive (Inlet 1)
```
[hsl] → [/ 127] → [s drive_param] → [r drive_param] → [torch.ts~ ...]
```
- Range: 0-1 (0 = clean, 1 = heavy distortion)
- Maps slider (0-127) to float (0-1)

#### Tone (Inlet 2)
```
[hsl] → [/ 127] → [s tone_param] → [r tone_param] → [torch.ts~ ...]
```
- Range: 0-1 (0 = dark, 0.5 = flat, 1 = bright)
- Controls high-frequency content

#### Mix (Inlet 3)
```
[hsl] → [/ 127] → [s mix_param] → [r mix_param] → [torch.ts~ ...]
```
- Range: 0-1 (0 = dry/bypass, 1 = fully wet)
- Blends processed and original signal

### 4. DSP Control

```
[loadbang] → [; pd dsp 1(
```
- Automatically starts DSP when patch loads

## Signal Flow Diagram

```
┌─────────┐
│  [adc~] │ Stereo input (L+R)
└────┬────┘
     │
     ▼
  [*~ 0.5] Attenuate input
     │
     ▼
[snake~ in 64] Convert audio stream to 64-sample blocks
     │
     ▼
┌─────────────────────────────────────┐
│ [torch.ts~ waveshaper.ts -m forward]│
│                                      │
│  Inlet 0: Audio blocks (64 samples) │◄─── [snake~ in 64]
│  Inlet 1: Drive (0-1)               │◄─── [r drive_param]
│  Inlet 2: Tone (0-1)                │◄─── [r tone_param]
│  Inlet 3: Mix (0-1)                 │◄─── [r mix_param]
│                                      │
│  Outlet: Processed blocks           │
└────────────────┬────────────────────┘
                 │
                 ▼
           [snake~ out 64] Convert blocks back to stream
                 │
                 ▼
              [*~ 0.8] Output gain
                 │
                 ▼
             ┌───┴───┐
             │ [dac~]│ Stereo output
             └───────┘
```

## Model Input/Output Specification

### Input Format (67 values)
```
[sample_0, sample_1, ..., sample_63, drive, tone, mix]
│                                      │      │     │
└─ 64 audio samples ──────────────────┘      │     │
                                              │     │
└─ Drive parameter (0-1) ────────────────────┘     │
                                                    │
└─ Tone parameter (0-1) ────────────────────────────┘
                                                    
└─ Mix parameter (0-1) ─────────────────────────────┘
```

### Output Format (64 values)
```
[processed_0, processed_1, ..., processed_63]
│
└─ 64 processed audio samples
```

## Important Notes

### snake~ Objects
The `snake~` objects are **essential** for block-based processing:
- **`[snake~ in BLOCKSIZE]`**: Converts continuous audio stream into discrete blocks
- **`[snake~ out BLOCKSIZE]`**: Converts discrete blocks back into continuous stream
- Without these, the model won't receive/produce data in the correct format

### Block Size
- **Must be 64 samples** (as specified during training)
- Set in `snake~` objects: `[snake~ in 64]` and `[snake~ out 64]`
- This creates a latency of **64/48000 = 1.33ms** at 48kHz
- Optional: Use `[block~ 64]` to force global 64-sample processing

### Sample Rate
- Model was trained at **48kHz**
- Use PureData at 48kHz for best results
- Other rates may work but could have quality degradation

### Parameter Ranges
All parameters must be in range **0.0 to 1.0**:
- Values outside this range may produce unexpected results
- Use `[clip 0 1]` if needed to ensure valid ranges

### Signal Levels
- Input should be normalized (typically -1 to +1)
- Output may exceed ±1.0 slightly (model uses tanh × 1.2)
- Use output gain ([*~ 0.8]) to prevent clipping

## Alternative Patch Structures

### Simple Version (No GUI)
```pd
[adc~]
   ↓
[snake~ in 64]
   ↓
[torch.ts~ waveshaper.ts -m forward]
   ↑     ↑     ↑
   │     │     │
 [sig~ 0.7] [sig~ 0.5] [sig~ 1.0]
 (drive)     (tone)     (mix)
   ↓
[snake~ out 64]
   ↓
[dac~]
```

### With Explicit Block Size Control
```pd
[block~ 64]  ← Optional: force 64-sample blocks globally
[adc~]
   ↓
[snake~ in 64]
   ↓
[torch.ts~ waveshaper.ts -m forward]
   ↓
[snake~ out 64]
   ↓
[dac~]
```

### Mono Processing
```pd
[adc~ 1]  ← Single channel
   ↓
[snake~ in 64]
   ↓
[torch.ts~ waveshaper.ts -m forward]
   ↓
[snake~ out 64]
   ↓
[dac~ 1]  ← Single channel out
```

### Stereo Processing (Independent L/R)
```pd
[adc~]
  ↓   ↓
 L│   │R
  ↓   ↓
[snake~ in 64] [snake~ in 64]
  ↓              ↓
[torch.ts~ ...] [torch.ts~ ...]
  ↓              ↓
[snake~ out 64] [snake~ out 64]
  ↓              ↓
[dac~]
```

## Testing the Patch

1. **Load the patch**: Open `waveshaper_effect.pd` in PureData
2. **Check model path**: Ensure `waveshaper.ts` is in the same directory
3. **Start audio**: DSP should start automatically via `[loadbang]`
4. **Adjust parameters**:
   - Start with Drive=0.5, Tone=0.5, Mix=0.8
   - Gradually increase Drive to hear distortion
   - Adjust Tone to change brightness
   - Use Mix to blend dry/wet signals
5. **Monitor levels**: Watch for clipping (red indicators)

## Troubleshooting

### "Can't find waveshaper.ts"
- Model file must be in same directory as patch
- Or use absolute path: `[torch.ts~ /full/path/to/waveshaper.ts 64]`

### "torch.ts~: No such object"
- Install contorchionist externals
- Check PureData preferences → Path includes external folder

### Distorted/Clipped Output
- Reduce input gain (`[*~ 0.5]`)
- Reduce output gain (`[*~ 0.8]`)
- Lower Drive parameter value

### High CPU Usage
- Block size is 64 samples = ~15 blocks per ms at 48kHz
- Normal for neural audio processing
- Consider using `[switch~]` to enable/bypass effect

### Latency Issues
- Inherent latency: 64 samples (1.33ms @ 48kHz)
- Add `[block~ 64]` to ensure consistent processing
- Cannot be reduced (model requires 64-sample blocks)

## Performance Tips

### CPU Optimization
```pd
[switch~]  ← Add bypass control
   ↓
[torch.ts~ waveshaper.ts 64]
```

### Parameter Smoothing
```pd
[hsl] → [line~] → [snapshot~] → [torch.ts~ ...]
         ↑
       [20(  ← Ramp time (ms)
```

### Metering
```pd
[torch.ts~ ...]
   ↓
[env~ 1024] → [vu 15 120]  ← VU meter
   ↓
[dac~]
```

## Files Required

1. **waveshaper.ts** - The trained neural model (required)
2. **waveshaper_effect.pd** - The PureData patch (this file)
3. **torch.ts~** - The external object (from contorchionist)

## Next Steps

- Try processing different sound sources (synth, vocals, drums)
- Create presets for different parameter combinations
- Add MIDI control for parameters
- Chain multiple effects together
- Record output and compare with target function

## References

- torch.ts~ documentation: See contorchionist repository
- Model training: See `01_Waveshaper_train_model.py`
- Model testing: See `02_Waveshaper_simpletest_model.py`

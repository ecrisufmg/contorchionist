#!/usr/bin/env python3
"""
03_Waveshaper_audiotest_model.py
=================================
Generates 5-second audio files to compare:
  1. target_waveshaper.wav - processed by the target function
  2. model_waveshaper.wav - processed by the neural model waveshaper.ts

The test processes audio in 64-sample blocks continuously.

Usage:
  conda run -n pytorch313 python 03_Waveshaper_audiotest_model.py

Output:
  - input_audio.wav (5s @ 48kHz) - original signal
  - target_waveshaper.wav (5s @ 48kHz) - target function output
  - model_waveshaper.wav (5s @ 48kHz) - neural model output
  - Comparison metrics (MSE, MAE, spectral difference)
"""
import numpy as np
import torch
import scipy.io.wavfile as wavfile

FS = 48000.0
BLOCK_SIZE = 64
DURATION = 5.0


def apply_tone_control(signal, tone):
    """Simple stateless tone control using first-order difference."""
    diff = np.zeros_like(signal)
    diff[1:] = signal[1:] - signal[:-1]
    diff[0] = 0
    tone_mapped = (tone - 0.5) * 2.0
    output = signal + tone_mapped * 0.3 * diff
    return output.astype(np.float32)


def process_block_target(input_block, drive, tone, mix):
    """Target waveshaper function (STATELESS)."""
    gain = 1.0 + drive * 9.0
    driven = input_block * gain
    shaped = np.tanh(driven)
    toned = apply_tone_control(shaped, tone)
    output = mix * toned + (1.0 - mix) * input_block
    return output.astype(np.float32)


def generate_input_audio(duration_sec, fs):
    """
    Generates a test audio signal with multiple frequency components.
    This simulates a rich musical signal for testing the effect.
    """
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs
    
    # Base tone: 110Hz (A2)
    signal = 0.4 * np.sin(2 * np.pi * 110 * t)
    
    # Add harmonics and other tones
    signal += 0.2 * np.sin(2 * np.pi * 220 * t)  # Octave
    signal += 0.15 * np.sin(2 * np.pi * 330 * t)  # Fifth
    signal += 0.1 * np.sin(2 * np.pi * 440 * t)  # Major third
    
    # Add some modulation (vibrato-like effect)
    lfo_freq = 4.0  # 4 Hz modulation
    modulation = 0.15 * np.sin(2 * np.pi * lfo_freq * t)
    signal = signal * (1.0 + modulation)
    
    # Add a bit of noise for realism
    signal += 0.02 * np.random.randn(n_samples)
    
    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.7
    
    return signal.astype(np.float32)


def process_audio_target(input_audio, drive, tone, mix, block_size):
    """Process entire audio with target function, block by block."""
    n_samples = len(input_audio)
    n_blocks = n_samples // block_size
    output = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        block_in = input_audio[start:end]
        block_out = process_block_target(block_in, drive, tone, mix)
        output[start:end] = block_out
    
    # Process remaining samples if any
    remainder = n_samples % block_size
    if remainder > 0:
        start = n_blocks * block_size
        block_in = input_audio[start:]
        # Pad to block size
        padded = np.pad(block_in, (0, block_size - remainder), mode='constant')
        block_out = process_block_target(padded, drive, tone, mix)
        output[start:] = block_out[:remainder]
    
    return output


def process_audio_model(model_path, input_audio, drive, tone, mix, block_size):
    """Process entire audio with neural model, block by block."""
    model = torch.jit.load(model_path)
    model.eval()
    
    n_samples = len(input_audio)
    n_blocks = n_samples // block_size
    output = np.zeros(n_samples, dtype=np.float32)
    
    # Prepare parameter tensors (constant for all blocks)
    drive_tensor = torch.tensor([[drive]], dtype=torch.float32)  # (1, 1)
    tone_tensor = torch.tensor([[tone]], dtype=torch.float32)  # (1, 1)
    mix_tensor = torch.tensor([[mix]], dtype=torch.float32)  # (1, 1)
    
    with torch.no_grad():
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            block_in = input_audio[start:end]
            
            # Create audio tensor
            audio_tensor = torch.tensor(block_in, dtype=torch.float32).unsqueeze(0)  # (1, 64)
            
            # Process with separate inputs
            block_out = model(audio_tensor, drive_tensor, tone_tensor, mix_tensor)[0].numpy()
            output[start:end] = block_out
        
        # Process remaining samples if any
        remainder = n_samples % block_size
        if remainder > 0:
            start = n_blocks * block_size
            block_in = input_audio[start:]
            padded = np.pad(block_in, (0, block_size - remainder), mode='constant')
            
            audio_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0)  # (1, 64)
            
            block_out = model(audio_tensor, drive_tensor, tone_tensor, mix_tensor)[0].numpy()
            output[start:] = block_out[:remainder]
    
    return output


def compute_metrics(target, model_output):
    """Computes comparison metrics."""
    mse = np.mean((target - model_output) ** 2)
    mae = np.mean(np.abs(target - model_output))
    max_error = np.max(np.abs(target - model_output))
    
    # Spectral difference
    target_fft = np.abs(np.fft.rfft(target))
    model_fft = np.abs(np.fft.rfft(model_output))
    spectral_diff = np.mean(np.abs(target_fft - model_fft))
    
    return {
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'spectral_diff': spectral_diff
    }


def main():
    print("=" * 70)
    print("Audio Test: Waveshaper Effect")
    print("=" * 70)
    
    # Effect parameters
    drive = 0.7  # Medium-high distortion
    tone = 0.6   # Slightly bright
    mix = 0.9    # Mostly wet
    
    print(f"\nParameters:")
    print(f"  Drive: {drive} (distortion amount)")
    print(f"  Tone: {tone} (brightness)")
    print(f"  Mix: {mix} (dry/wet blend)")
    print(f"  Duration: {DURATION}s @ {FS:.0f} Hz")
    print(f"  Block size: {BLOCK_SIZE} samples")
    print(f"  Total samples: {int(DURATION * FS)}")
    
    # Generate input audio
    print("\n1. Generating input audio signal...")
    input_audio = generate_input_audio(DURATION, FS)
    print(f"   ✓ Generated: {len(input_audio)} samples")
    print(f"   Range: [{input_audio.min():.4f}, {input_audio.max():.4f}]")
    
    # Process with target function
    print("\n2. Processing with target function...")
    target_audio = process_audio_target(input_audio, drive, tone, mix, BLOCK_SIZE)
    print(f"   ✓ Processed: {len(target_audio)} samples")
    print(f"   Range: [{target_audio.min():.4f}, {target_audio.max():.4f}]")
    
    # Process with neural model
    print("\n3. Processing with neural model...")
    model_audio = process_audio_model('waveshaper.ts', input_audio, drive, tone, mix, BLOCK_SIZE)
    print(f"   ✓ Processed: {len(model_audio)} samples")
    print(f"   Range: [{model_audio.min():.4f}, {model_audio.max():.4f}]")
    
    # Compute metrics
    print("\n4. Comparing outputs...")
    metrics = compute_metrics(target_audio, model_audio)
    print(f"   MSE:             {metrics['mse']:.8f}")
    print(f"   MAE:             {metrics['mae']:.6f}")
    print(f"   Max error:       {metrics['max_error']:.6f}")
    print(f"   Spectral diff:   {metrics['spectral_diff']:.6f}")
    
    # Save WAV files
    print("\n5. Saving WAV files...")
    
    def to_int16(audio):
        audio_norm = audio / (np.max(np.abs(audio)) + 1e-8)
        return (audio_norm * 32767).astype(np.int16)
    
    input_int16 = to_int16(input_audio)
    target_int16 = to_int16(target_audio)
    model_int16 = to_int16(model_audio)
    
    wavfile.write('input_audio.wav', int(FS), input_int16)
    wavfile.write('target_waveshaper.wav', int(FS), target_int16)
    wavfile.write('model_waveshaper.wav', int(FS), model_int16)
    
    print(f"   ✓ Saved: input_audio.wav")
    print(f"   ✓ Saved: target_waveshaper.wav")
    print(f"   ✓ Saved: model_waveshaper.wav")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS:")
    if metrics['mae'] < 0.01:
        print("  ✅ Excellent: Average error < 0.01 (almost imperceptible)")
    elif metrics['mae'] < 0.05:
        print("  ✅ Very good: Average error < 0.05 (subtle difference)")
    elif metrics['mae'] < 0.1:
        print("  ⚠️  Fair: Average error < 0.1 (perceptible difference)")
    else:
        print("  ❌ High error: Significant difference between target and model")
    
    print("\nRecommendation:")
    print("  Listen to the three files to compare:")
    print("    - input_audio.wav (original signal)")
    print("    - target_waveshaper.wav (mathematical target)")
    print("    - model_waveshaper.wav (neural network)")
    print("=" * 70)


if __name__ == "__main__":
    main()

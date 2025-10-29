#!/usr/bin/env python3
"""
04_Waveshaper_debug_blocks.py
==============================
Debug script to analyze where errors occur in block-by-block processing.
Processes a few consecutive blocks and shows detailed sample-by-sample comparison.

Includes spectral brightness analysis using librosa's spectral centroid.
"""
import numpy as np
import torch
import librosa

FS = 48000.0
BLOCK_SIZE = 64


def compute_brightness(signal, sr):
    """
    Compute spectral brightness (spectral centroid) of a signal.
    Higher values indicate brighter/more high-frequency content.
    
    Returns:
        float: Spectral centroid in Hz
    """
    # Compute spectral centroid using librosa
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=min(512, len(signal)))[0]
    return np.mean(centroid)


def apply_tone_control(signal, tone):
    """Simple stateless tone control using first-order difference."""
    diff = np.zeros_like(signal)
    diff[1:] = signal[1:] - signal[:-1]
    diff[0] = 0
    tone_mapped = (tone - 0.5) * 2.0
    output = signal + tone_mapped * 0.3 * diff
    return output.astype(np.float32)


def process_block_target(input_block, drive, tone, mix):
    """Target waveshaper function."""
    gain = 1.0 + drive * 9.0
    driven = input_block * gain
    shaped = np.tanh(driven)
    toned = apply_tone_control(shaped, tone)
    output = mix * toned + (1.0 - mix) * input_block
    return output.astype(np.float32)


def main():
    model = torch.jit.load('waveshaper.ts')
    model.eval()
    
    # Parameters
    drive, tone, mix = 0.7, 0.6, 0.9
    
    # Generate continuous audio (5 blocks = 320 samples)
    n_blocks = 5
    n_samples = n_blocks * BLOCK_SIZE
    t = np.arange(n_samples) / FS
    
    # Simple 440Hz sine
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    print("=" * 80)
    print("Debug: Processing 5 consecutive blocks (320 samples total)")
    print("=" * 80)
    print(f"Parameters: drive={drive}, tone={tone}, mix={mix}")
    print(f"Input: 440Hz sine wave, amplitude=0.5\n")
    
    # Process block by block
    for block_idx in range(n_blocks):
        start = block_idx * BLOCK_SIZE
        end = start + BLOCK_SIZE
        input_block = audio[start:end]
        
        # Target
        target_output = process_block_target(input_block, drive, tone, mix)
        
        # Model - prepare separate tensors
        audio_tensor = torch.tensor(input_block, dtype=torch.float32).unsqueeze(0)  # (1, 64)
        drive_tensor = torch.tensor([[drive]], dtype=torch.float32)  # (1, 1)
        tone_tensor = torch.tensor([[tone]], dtype=torch.float32)  # (1, 1)
        mix_tensor = torch.tensor([[mix]], dtype=torch.float32)  # (1, 1)
        
        with torch.no_grad():
            model_output = model(audio_tensor, drive_tensor, tone_tensor, mix_tensor)[0].numpy()
        
        # Analyze
        error = np.abs(model_output - target_output)
        mae = np.mean(error)
        max_err = np.max(error)
        
        # Compute brightness (spectral centroid)
        brightness_input = compute_brightness(input_block, FS)
        brightness_target = compute_brightness(target_output, FS)
        brightness_model = compute_brightness(model_output, FS)
        
        print(f"Block {block_idx} (samples {start}-{end-1})")
        print(f"  MAE: {mae:.6f}, Max Error: {max_err:.6f}")
        print(f"  Input range: [{np.min(input_block):.4f}, {np.max(input_block):.4f}]")
        print(f"  Target range: [{np.min(target_output):.4f}, {np.max(target_output):.4f}]")
        print(f"  Model range: [{np.min(model_output):.4f}, {np.max(model_output):.4f}]")
        print(f"  Brightness (Hz):")
        print(f"    Input: {brightness_input:.1f} Hz")
        print(f"    Target: {brightness_target:.1f} Hz (tone={tone})")
        print(f"    Model: {brightness_model:.1f} Hz")
        print(f"    Δ Target-Model: {abs(brightness_target - brightness_model):.1f} Hz")
        
        # Show first/last samples of each block
        print(f"  First 3 samples:")
        for i in range(3):
            print(f"    [{i}] input={input_block[i]:7.4f}, target={target_output[i]:7.4f}, "
                  f"model={model_output[i]:7.4f}, err={error[i]:.4f}")
        
        print(f"  Last 3 samples:")
        for i in range(61, 64):
            print(f"    [{i}] input={input_block[i]:7.4f}, target={target_output[i]:7.4f}, "
                  f"model={model_output[i]:7.4f}, err={error[i]:.4f}")
        
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
02_Waveshaper_simpletest_model.py
==================================
Tests the TorchScript waveshaper.ts model by processing multiple audio blocks
with different parameters and evaluating error against the target function.

Usage:
  conda run -n pytorch313 python 02_Waveshaper_simpletest_model.py
"""
import numpy as np
import torch

FS = 48000.0
BLOCK_SIZE = 64


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


def main():
    model = torch.jit.load('waveshaper.ts')
    model.eval()
    
    print("=" * 70)
    print("Testing Waveshaper Model")
    print("=" * 70)
    
    # Test scenarios with different input signals and parameters
    tests = [
        {
            "name": "440Hz sine, light drive",
            "freq": 440.0, "amp": 0.5,
            "drive": 0.2, "tone": 0.7, "mix": 1.0
        },
        {
            "name": "1kHz sine, heavy drive, dark tone",
            "freq": 1000.0, "amp": 0.6,
            "drive": 0.9, "tone": 0.2, "mix": 1.0
        },
        {
            "name": "220Hz sine, medium drive, bright",
            "freq": 220.0, "amp": 0.7,
            "drive": 0.5, "tone": 0.9, "mix": 1.0
        },
        {
            "name": "2kHz sine, heavy drive, 50% mix",
            "freq": 2000.0, "amp": 0.5,
            "drive": 0.8, "tone": 0.5, "mix": 0.5
        },
        {
            "name": "Multi-tone complex signal",
            "freq": None, "amp": None,  # Will generate complex signal
            "drive": 0.6, "tone": 0.6, "mix": 0.8
        },
    ]
    
    for test_cfg in tests:
        print(f"\n{test_cfg['name']}")
        print("-" * 70)
        
        # Generate input signal
        t = np.arange(BLOCK_SIZE) / FS
        
        if test_cfg['freq'] is not None:
            # Simple sine wave
            signal = test_cfg['amp'] * np.sin(2 * np.pi * test_cfg['freq'] * t)
        else:
            # Complex multi-tone signal
            signal = (
                0.3 * np.sin(2 * np.pi * 300 * t) +
                0.2 * np.sin(2 * np.pi * 800 * t) +
                0.15 * np.sin(2 * np.pi * 1500 * t) +
                0.05 * np.random.randn(BLOCK_SIZE)
            )
        
        drive = test_cfg['drive']
        tone = test_cfg['tone']
        mix = test_cfg['mix']
        
        # Process with target function
        target_output = process_block_target(signal, drive, tone, mix)
        
        # Process with model (separate inputs - torch.ts~ compatible)
        with torch.no_grad():
            audio_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # (1, 64)
            drive_tensor = torch.tensor([[drive]], dtype=torch.float32)  # (1, 1)
            tone_tensor = torch.tensor([[tone]], dtype=torch.float32)  # (1, 1)
            mix_tensor = torch.tensor([[mix]], dtype=torch.float32)  # (1, 1)
            
            model_output = model(audio_tensor, drive_tensor, tone_tensor, mix_tensor)[0].numpy()
        
        # Compute metrics
        mse = np.mean((model_output - target_output) ** 2)
        mae = np.mean(np.abs(model_output - target_output))
        max_error = np.max(np.abs(model_output - target_output))
        
        print(f"Parameters: drive={drive:.2f}, tone={tone:.2f}, mix={mix:.2f}")
        print(f"MSE: {mse:.8f} | MAE: {mae:.6f} | Max Error: {max_error:.6f}")
        
        # Show some sample comparisons
        print("\nSample comparisons:")
        for i in [0, 16, 32, 48, 63]:
            err = abs(model_output[i] - target_output[i])
            print(f"  [{i:2d}] model={model_output[i]:7.4f}, target={target_output[i]:7.4f}, err={err:.5f}")
    
    print("\n" + "=" * 70)
    print("Testing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

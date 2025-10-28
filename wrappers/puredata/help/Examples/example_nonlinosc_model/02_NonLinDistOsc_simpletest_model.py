#!/usr/bin/env python3
"""
Tests the TorchScript model oscillator_v2.ts by generating a continuous
waveform and evaluating the average error against the target function.

Usage:
  conda run -n pytorch313 python test_oscillator_v2.py
"""
import math
import numpy as np
import torch

FS = 48000.0


def generate_sample(freq_hz, amp, distortion_0_100, bias, phase_norm):
    phi = 2 * math.pi * phase_norm
    gain = 1 + 9 * (distortion_0_100 / 100.0)
    return amp * math.tanh(gain * math.sin(phi)) + bias


def main():
    model = torch.jit.load('oscillator_v2.ts')
    model.eval()

    # Test scenarios
    tests = [
        {"freq": 110.0, "amp": 1.0, "dist": 0.0, "bias": 0.0, "n": 4096},
        {"freq": 440.0, "amp": 0.8, "dist": 50.0, "bias": 0.0, "n": 4096},
        {"freq": 5000.0, "amp": 0.7, "dist": 90.0, "bias": 0.1, "n": 4096},
        {"freq": 12000.0, "amp": 0.9, "dist": 30.0, "bias": -0.1, "n": 4096},
        {"freq": 20000.0, "amp": 0.5, "dist": 75.0, "bias": 0.0, "n": 4096},
    ]

    for cfg in tests:
        freq_hz = cfg["freq"]
        amp = cfg["amp"]
        dist = cfg["dist"]
        bias = cfg["bias"]
        n = cfg["n"]

        phase = 0.0
        inc = freq_hz / FS

        model_vals = []
        target_vals = []

        with torch.no_grad():
            for _ in range(n):
                phase_norm = phase % 1.0
                inp = torch.tensor([[freq_hz, amp, dist, bias, float(phase_norm)]], dtype=torch.float32)
                y = model(inp)[0, 0].item()
                t = generate_sample(freq_hz, amp, dist, bias, phase_norm)
                model_vals.append(y)
                target_vals.append(t)
                phase += inc

        model_arr = np.array(model_vals, dtype=np.float64)
        target_arr = np.array(target_vals, dtype=np.float64)

        mse = np.mean((model_arr - target_arr) ** 2)
        mae = np.mean(np.abs(model_arr - target_arr))
        print(f"freq={freq_hz:7.1f}Hz | amp={amp:.2f} | dist%={dist:5.1f} | bias={bias:+.2f} -> MSE={mse:.6f}, MAE={mae:.4f}")

        # Show some reference points
        for i in [0, n//8, n//4, 3*n//8, n//2]:
            print(f"  i={i:4d}, phase={((i*inc)%1.0):.4f}, model={model_arr[i]: .4f}, target={target_arr[i]: .4f}")
        print("-")


if __name__ == "__main__":
    main()

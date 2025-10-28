#!/usr/bin/env python3
"""
train_oscillator_model.py (v2)
==============================
Trains a model that generates 1 sample at a time from a non-linear oscillator.

Important changes in this version:
- Now accepts frequency in Hz (0 to 24000, fs=48000) with internal normalization.
- Phase is NORMALIZED in the interval [0, 1) relative to the period (independent of absolute index),
  which ensures stability even for very high phase values.
- Distortion is specified as 0..100 and normalized internally to 0..1 (with mapping to gain 1..10).
- Model improvements: periodic features sin/cos(2π·phase), larger MLP and SiLU activation.

Input interface (maintained with 5 elements):
    [freq_hz, amp, distortion_0_100, bias, phase_norm]
  - freq_hz: frequency in Hz (0 to 24000) — will be normalized by fs=48000 internally
  - amp: amplitude (0 to 1)
  - distortion_0_100: intensity 0..100 — will be normalized to 0..1 and mapped to gain 1..10 in target function
  - bias: DC offset (-0.5 to 0.5)
  - phase_norm: phase normalized by period in [0, 1) — 0 at cycle start, 0.5 at π rad, 1 wraps to 0

Output:
    [sample_value] - single sample

Target function (for generating supervision):
    phi = 2π * phase_norm
    gain = 1 + 9 * (distortion_0_100 / 100)
    output = amp * tanh(gain * sin(phi)) + bias

Note:
- Frequency in Hz doesn't enter directly into the target function at the current instant (depends only on current phase),
  but is maintained as input for compatibility and future extensions. In practical use, frequency controls
  the phase increment between samples: phase_norm += freq_hz / fs; phase_norm %= 1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

FS = 48000.0  # Sampling rate used for frequency normalization

class OscillatorModel(nn.Module):
    def __init__(self):
        super(OscillatorModel, self).__init__()
        # We use 6 internal features: [freq_norm, amp, dist_norm, bias, sinphi, cosphi]
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        x: tensor (N, 5) with [freq_hz, amp, distortion_0_100, bias, phase_norm]
        """
        # Separate inputs
        freq_hz = x[:, 0]
        amp = x[:, 1]
        dist_pct = x[:, 2]
        bias = x[:, 3]
        phase_norm = x[:, 4]

        # Normalizations and periodic features
        freq_norm = freq_hz / FS
        dist_norm = dist_pct / 100.0
        phi = 2 * torch.pi * phase_norm
        # Periodic features directly with torch
        sinphi = torch.sin(phi)
        cosphi = torch.cos(phi)

        features = torch.stack([freq_norm, amp, dist_norm, bias, sinphi, cosphi], dim=1)

        h = self.act(self.fc1(features))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))
        y = self.out(h)
        return y

def generate_sample(freq_hz, amp, distortion_0_100, bias, phase_norm):
    """Generates ONE sample from the non-linear oscillator with normalized phase [0,1)."""
    phi = 2 * np.pi * phase_norm
    gain = 1 + 9 * (distortion_0_100 / 100.0)
    sample = amp * np.tanh(gain * np.sin(phi)) + bias
    return float(sample)

def create_synthetic_data(num_examples=20000):
    """
    Creates synthetic data. Each example is a sample with its parameters.

    Inputs (per example):
      - freq_hz ~ U[0, 24000]
      - amp ~ U[0.05, 1.0]
      - distortion_0_100 ~ U[0, 100]
      - bias ~ U[-0.5, 0.5]
      - phase_norm ~ U[0, 1)
    """
    inputs = []
    targets = []

    for _ in range(num_examples):
        freq_hz = np.random.uniform(0.0, 24000.0)
        amp = np.random.uniform(0.05, 1.0)
        distortion_0_100 = np.random.uniform(0.0, 100.0)
        bias = np.random.uniform(-0.5, 0.5)
        phase_norm = np.random.uniform(0.0, 1.0)

        sample = generate_sample(freq_hz, amp, distortion_0_100, bias, phase_norm)

        inputs.append([freq_hz, amp, distortion_0_100, bias, phase_norm])
        targets.append([sample])

    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

def train_model():
    print("=" * 70)
    print("Training Non-Linear Oscillator (1 sample at a time)")
    print("=" * 70)
    print("Input:  [freq_hz, amp, distortion(0..100), bias, phase_norm]")
    print("Output: [sample_value]")
    print("Call multiple times to build the waveform!")
    print(f"fs={FS:.0f} Hz; internally we normalize freq_hz/fs and use sin/cos(2π·phase_norm)")
    print("=" * 70)
    
    model = OscillatorModel()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("\n1. Generating synthetic data...")
    train_inputs, train_targets = create_synthetic_data(20000)
    print(f"   - {len(train_inputs)} training samples created")
    
    print("\n2. Training...")
    num_epochs = 1500
    model.train()

    # Train with mini-batches for better stability
    batch_size = 1024
    n = train_inputs.shape[0]

    for epoch in range(num_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            batch_x = train_inputs[idx]
            batch_y = train_targets[idx]

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= n
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"   Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
    
    print(f"\n3. Training completed! Final loss: {loss.item():.6f}")
    
    # Test: generate a complete waveform
    print("\n4. Testing: generating 64 consecutive samples")
    print("-" * 70)
    model.eval()
    
    freq_hz, amp, dist_pct, bias = 440.0, 0.9, 50.0, 0.0
    print(f"Parameters: freq_hz={freq_hz}, amp={amp}, dist%={dist_pct}, bias={bias}\n")

    # Generate normalized phase sequence for 64 samples
    phase = 0.0
    phase_inc = freq_hz / FS

    with torch.no_grad():
        print("Idx | Phase  | Model   | Expected | Error")
        print("----|--------|---------|----------|--------")

        for idx in [0, 16, 32, 48, 64, 80, 96, 112]:
            phase_norm = phase % 1.0
            input_tensor = torch.tensor([[freq_hz, amp, dist_pct, bias, float(phase_norm)]])
            model_output = model(input_tensor)[0, 0].item()
            expected = generate_sample(freq_hz, amp, dist_pct, bias, phase_norm)
            error = abs(model_output - expected)

            print(f" {idx:3d} | {phase_norm:1.4f} | {model_output:7.4f} | {expected:8.4f} | {error:.4f}")
            phase += phase_inc * (16 if idx != 0 else 0)
    
    print("\n" + "-" * 70)
    
    # Save the model
    print("\n5. Saving model as TorchScript...")
    model.eval()
    example_input = torch.randn(1, 5)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save("nonlinosc.ts")
    print("   ✓ Model saved as: nonlinosc.ts")
    
    print("\n" + "=" * 70)
    print("HOW TO USE:")
    print("  - Send [freq_hz amp dist(0..100) bias phase_norm] to the model")
    print("  - Update phase with phase_norm = (phase_norm + freq_hz/48000) % 1.0 at each sample")
    print("  - Collect outputs in a list to build the waveform!")
    print("=" * 70)

if __name__ == "__main__":
    train_model()

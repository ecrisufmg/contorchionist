#!/usr/bin/env python3
"""
01_Waveshaper_train_model.py
=============================
Trains a model that processes audio blocks (64 samples) with a waveshaper effect.

The effect combines:
  1. Drive (pre-gain): amplifies the signal before distortion
  2. Waveshaping: tanh saturation for warm distortion
  3. Tone control: simple high-shelf (boosts/cuts high frequencies)
  4. Dry/wet mix: blends processed and original signal

Input interface (TORCH.TS~ COMPATIBLE):
    - audio: (N, 64) tensor with audio samples
    - drive: (N, 1) tensor with distortion amount 0..1 (maps to gain 1..10)
    - tone: (N, 1) tensor with brightness 0..1 (0=cut highs, 0.5=flat, 1=boost highs)
    - mix: (N, 1) tensor with dry/wet blend 0..1 (0=dry, 1=fully wet)

Output:
    - processed: (N, 64) tensor with processed audio samples
    
This architecture is compatible with torch.ts~ which expects:
    - Inlet 0: audio channel (64 samples per block)
    - Inlet 1: drive parameter (scalar)
    - Inlet 2: tone parameter (scalar)
    - Inlet 3: mix parameter (scalar)

Target function (STATELESS - uses only current and previous sample within block):
    1. driven = input * (1 + drive * 9)
    2. shaped = tanh(driven)
    3. toned = shaped + tone * first_difference(shaped)  # Simple treble boost/cut
    4. output = mix * toned + (1 - mix) * input
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

FS = 48000.0  # Sampling rate
BLOCK_SIZE = 64  # Audio block size


class WaveshaperModel(nn.Module):
    def __init__(self):
        super(WaveshaperModel, self).__init__()
        # Architecture compatible with torch.ts~
        # torch.ts~ calls: forward(ch0, ch1, ch2, ch3) - 4 SEPARATE tensors
        # Each tensor has shape (N, 64)
        
        self.fc1 = nn.Linear(67, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.act = nn.SiLU()
        
    def forward(self, ch0, ch1, ch2, ch3):
        """
        Forward pass with 4 SEPARATE channel tensors (torch.ts~ compatible).
        
        Args:
            ch0: (N, 64) - audio channel
            ch1: (N, 64) - drive parameter as DC signal
            ch2: (N, 64) - tone parameter as DC signal
            ch3: (N, 64) - mix parameter as DC signal
            
        Returns:
            (N, 64) - processed audio (single channel output)
        """
        # Extract DC values from parameter channels (take first sample)
        audio = ch0  # (N, 64)
        drive = ch1[:, 0:1]  # (N, 1) - first sample of DC signal
        tone = ch2[:, 0:1]   # (N, 1) - first sample of DC signal
        mix = ch3[:, 0:1]    # (N, 1) - first sample of DC signal
        
        # Concatenate audio with parameters: (N, 64) + (N, 3) = (N, 67)
        features = torch.cat([audio, drive, tone, mix], dim=1)
        
        # Process through network
        h = self.act(self.fc1(features))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))
        output = self.fc4(h)
        
        # Apply tanh to keep output in reasonable range [-1, 1]
        output = torch.tanh(output) * 1.2
        
        return output  # (N, 64) - single channel


def apply_tone_control(signal, tone):
    """
    Simple stateless tone control using first-order difference.
    - tone < 0: cut high frequencies (darker)
    - tone = 0: flat response
    - tone > 0: boost high frequencies (brighter)
    
    Uses first-order difference as a crude high-frequency emphasis:
      diff[n] = signal[n] - signal[n-1]
    Then: output = signal + tone * diff
    
    This is completely stateless within each block.
    """
    # First-order difference (high-pass characteristic)
    diff = np.zeros_like(signal)
    diff[1:] = signal[1:] - signal[:-1]
    diff[0] = 0  # No previous sample in block
    
    # Apply tone control: mix in the difference signal
    # tone in range [-1, 1], so we map from 0..1 to -1..1
    tone_mapped = (tone - 0.5) * 2.0  # Maps 0..1 to -1..1
    output = signal + tone_mapped * 0.3 * diff  # Scale diff by 0.3 to keep it subtle
    
    return output.astype(np.float32)


def process_block_target(input_block, drive, tone, mix):
    """
    Target function: processes one 64-sample block with waveshaper effect.
    
    Args:
        input_block: numpy array (64,) with input samples
        drive: float 0..1 (distortion amount)
        tone: float 0..1 (brightness)
        mix: float 0..1 (dry/wet)
    
    Returns:
        numpy array (64,) with processed samples
    """
    # 1. Apply drive (pre-gain)
    gain = 1.0 + drive * 9.0  # Maps 0..1 to 1..10
    driven = input_block * gain
    
    # 2. Waveshaping (saturation)
    shaped = np.tanh(driven)
    
    # 3. Tone control (simple treble boost/cut)
    toned = apply_tone_control(shaped, tone)
    
    # 4. Mix dry/wet
    output = mix * toned + (1.0 - mix) * input_block
    
    return output.astype(np.float32)


def create_synthetic_data(num_examples=5000):
    """
    Creates synthetic training data.
    Each example is a 64-sample block with random parameters.
    
    IMPORTANT: Generates both isolated blocks AND consecutive blocks
    from longer signals to help the model generalize to continuous audio.
    EMPHASIS on consecutive blocks to learn smooth transitions.
    
    Returns:
        Tuple of tensors for 4 separate channels + targets:
        - ch0: (N, 64) - audio channel
        - ch1: (N, 64) - drive as DC signal
        - ch2: (N, 64) - tone as DC signal
        - ch3: (N, 64) - mix as DC signal
        - targets: (N, 64) - target output
    """
    print(f"   Generating {num_examples} training examples...")
    print(f"   (10% isolated, 90% consecutive blocks - HEAVY continuity focus to eliminate 750Hz)")
    
    ch0_list = []  # audio
    ch1_list = []  # drive DC
    ch2_list = []  # tone DC
    ch3_list = []  # mix DC
    targets = []
    
    for idx in range(num_examples):
        if (idx + 1) % 1000 == 0:
            print(f"   ... {idx + 1}/{num_examples} examples generated")
        
        # 10% isolated blocks, 90% consecutive blocks (HEAVY continuity focus)
        if idx % 10 < 1:  # Only 10% isolated
            # Generate a single random 64-sample block
            t = np.arange(BLOCK_SIZE) / FS
            
            freq1 = np.random.uniform(100, 5000)
            freq2 = np.random.uniform(100, 5000)
            amp1 = np.random.uniform(0.1, 0.8)
            amp2 = np.random.uniform(0.1, 0.4)
            
            signal = amp1 * np.sin(2 * np.pi * freq1 * t) + amp2 * np.sin(2 * np.pi * freq2 * t)
            
            noise_amp = np.random.uniform(0.0, 0.1)
            signal += noise_amp * np.random.randn(BLOCK_SIZE)
            
            signal = signal / (np.max(np.abs(signal)) + 1e-8) * np.random.uniform(0.3, 0.9)
        
        else:  # 90% consecutive - learn continuous signals!
            # Generate a longer signal and extract a random block
            # This creates blocks that have realistic continuity
            long_length = BLOCK_SIZE * np.random.randint(7, 16)  # 7-15 blocks long (MUCH longer sequences)
            t_long = np.arange(long_length) / FS
            
            freq1 = np.random.uniform(100, 5000)
            freq2 = np.random.uniform(100, 5000)
            amp1 = np.random.uniform(0.1, 0.8)
            amp2 = np.random.uniform(0.1, 0.4)
            
            long_signal = amp1 * np.sin(2 * np.pi * freq1 * t_long) + amp2 * np.sin(2 * np.pi * freq2 * t_long)
            
            noise_amp = np.random.uniform(0.0, 0.1)
            long_signal += noise_amp * np.random.randn(long_length)
            
            long_signal = long_signal / (np.max(np.abs(long_signal)) + 1e-8) * np.random.uniform(0.3, 0.9)
            
            # Extract a random 64-sample block (not necessarily aligned)
            max_start = long_length - BLOCK_SIZE
            start_idx = np.random.randint(0, max_start)
            signal = long_signal[start_idx:start_idx + BLOCK_SIZE]
        
        # Random parameters
        drive = np.random.uniform(0.0, 1.0)
        tone = np.random.uniform(0.0, 1.0)
        mix = np.random.uniform(0.0, 1.0)
        
        # Process with target function
        output_block = process_block_target(signal, drive, tone, mix)
        
        # Create DC channels (constant values repeated 64 times)
        drive_dc = np.full(BLOCK_SIZE, drive, dtype=np.float32)
        tone_dc = np.full(BLOCK_SIZE, tone, dtype=np.float32)
        mix_dc = np.full(BLOCK_SIZE, mix, dtype=np.float32)
        
        # Store as separate channels
        ch0_list.append(signal)
        ch1_list.append(drive_dc)
        ch2_list.append(tone_dc)
        ch3_list.append(mix_dc)
        targets.append(output_block)
    
    print(f"   ✓ All {num_examples} examples generated!")
    
    # Convert to tensors
    ch0 = torch.from_numpy(np.array(ch0_list, dtype=np.float32))  # (N, 64)
    ch1 = torch.from_numpy(np.array(ch1_list, dtype=np.float32))  # (N, 64)
    ch2 = torch.from_numpy(np.array(ch2_list, dtype=np.float32))  # (N, 64)
    ch3 = torch.from_numpy(np.array(ch3_list, dtype=np.float32))  # (N, 64)
    targets_tensor = torch.from_numpy(np.array(targets, dtype=np.float32))  # (N, 64)
    
    return ch0, ch1, ch2, ch3, targets_tensor


def continuity_loss(outputs, targets):
    """
    Penalizes discontinuities at block boundaries.
    Uses multiple approaches:
    1. Direct jump (last sample vs first sample of next block)
    2. Derivative continuity (slope continuity)
    3. Weighted more heavily to really enforce smooth transitions
    """
    if outputs.shape[0] < 2:
        return torch.tensor(0.0)
    
    # Approach 1: Direct sample discontinuity
    last_samples = outputs[:-1, -1]    # (N-1,)
    first_samples = outputs[1:, 0]     # (N-1,)
    target_last = targets[:-1, -1]
    target_first = targets[1:, 0]
    
    # Absolute jump between blocks
    output_jump = torch.abs(first_samples - last_samples)
    target_jump = torch.abs(target_first - target_last)
    jump_loss = torch.mean((output_jump - target_jump) ** 2)
    
    # Approach 2: Derivative continuity (check slope)
    # Last 2 samples of each block
    output_slope_before = outputs[:-1, -1] - outputs[:-1, -2]
    target_slope_before = targets[:-1, -1] - targets[:-1, -2]
    
    # First 2 samples of next block
    output_slope_after = outputs[1:, 1] - outputs[1:, 0]
    target_slope_after = targets[1:, 1] - targets[1:, 0]
    
    # Penalize slope mismatch
    slope_diff_output = torch.abs(output_slope_after - output_slope_before)
    slope_diff_target = torch.abs(target_slope_after - target_slope_before)
    slope_loss = torch.mean((slope_diff_output - slope_diff_target) ** 2)
    
    # Combine both losses
    return jump_loss + 0.5 * slope_loss


def train_model():
    print("=" * 70)
    print("Training Waveshaper Effect (64-sample blocks)")
    print("=" * 70)
    print("Input:  4 separate channel tensors (audio, drive_dc, tone_dc, mix_dc)")
    print("Output: 1 channel tensor (processed audio)")
    print(f"Block size: {BLOCK_SIZE}, Sampling rate: {FS:.0f} Hz")
    print("With CONTINUITY LOSS to prevent block boundary artifacts")
    print("=" * 70)
    
    model = WaveshaperModel()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    print("\n1. Generating synthetic data...")
    import time
    start_time = time.time()
    train_ch0, train_ch1, train_ch2, train_ch3, train_targets = create_synthetic_data(5000)
    elapsed = time.time() - start_time
    print(f"   - Training data ready: {len(train_ch0)} blocks (took {elapsed:.1f}s)")
    print(f"   - CH0 (audio) shape: {train_ch0.shape}")
    print(f"   - CH1 (drive) shape: {train_ch1.shape}")
    print(f"   - CH2 (tone) shape: {train_ch2.shape}")
    print(f"   - CH3 (mix) shape: {train_ch3.shape}")
    print(f"   - Target shape: {train_targets.shape}")
    
    print("\n2. Training...")
    num_epochs = 2000 
    model.train()
    
    # Train with mini-batches
    batch_size = 64
    n = train_ch0.shape[0]
    
    print(f"   Using mini-batches of size {batch_size}")
    print(f"   Total batches per epoch: {n // batch_size}")
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 50  # Early stopping
    
    # Continuity loss weight - VERY HIGH to eliminate 750Hz artifact
    continuity_weight = 20.0  # Increased from 5.0 - aggressive continuity enforcement
    print(f"   Continuity weight: {continuity_weight} (VERY HIGH to eliminate 750Hz artifact)")
    
    for epoch in range(num_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        epoch_cont_loss = 0.0
        
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            batch_ch0 = train_ch0[idx]  # (batch, 64)
            batch_ch1 = train_ch1[idx]  # (batch, 64)
            batch_ch2 = train_ch2[idx]  # (batch, 64)
            batch_ch3 = train_ch3[idx]  # (batch, 64)
            batch_y = train_targets[idx]  # (batch, 64)
            
            outputs = model(batch_ch0, batch_ch1, batch_ch2, batch_ch3)  # (batch, 64)
            
            # Main reconstruction loss
            mse_loss = criterion(outputs, batch_y)
            
            # Continuity loss (penalize discontinuities between consecutive blocks)
            cont_loss = continuity_loss(outputs, batch_y)
            
            # Combined loss
            loss = mse_loss + continuity_weight * cont_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += mse_loss.item() * batch_ch0.size(0)
            epoch_cont_loss += cont_loss.item() * batch_ch0.size(0)
        
        epoch_loss /= n
        epoch_cont_loss /= n
        
        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"   Epoch [{epoch+1}/{num_epochs}], MSE: {epoch_loss:.6f}, Continuity: {epoch_cont_loss:.6f}")
        elif (epoch + 1) % 10 == 0:
            # More frequent but less verbose feedback
            print(f"   ... epoch {epoch+1}/{num_epochs} (mse: {epoch_loss:.6f}, cont: {epoch_cont_loss:.6f})")
    
    print(f"\n3. Training completed! Final MSE: {epoch_loss:.6f}, Best MSE: {best_loss:.6f}")
    print(f"   Final Continuity Loss: {epoch_cont_loss:.6f}")
    
    # Test: process a simple sine wave block
    print("\n4. Testing with a 440Hz sine wave block")
    print("-" * 70)
    model.eval()
    
    # Generate test signal
    t = np.arange(BLOCK_SIZE) / FS
    test_signal = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    
    drive, tone, mix = 0.7, 0.5, 1.0
    print(f"Parameters: drive={drive}, tone={tone}, mix={mix}\n")
    
    with torch.no_grad():
        # Target
        target_output = process_block_target(test_signal, drive, tone, mix)
        
        # Model - prepare 4 separate channel tensors
        drive_dc = np.full(BLOCK_SIZE, drive, dtype=np.float32)
        tone_dc = np.full(BLOCK_SIZE, tone, dtype=np.float32)
        mix_dc = np.full(BLOCK_SIZE, mix, dtype=np.float32)
        
        ch0 = torch.tensor(test_signal, dtype=torch.float32).unsqueeze(0)  # (1, 64)
        ch1 = torch.tensor(drive_dc, dtype=torch.float32).unsqueeze(0)     # (1, 64)
        ch2 = torch.tensor(tone_dc, dtype=torch.float32).unsqueeze(0)      # (1, 64)
        ch3 = torch.tensor(mix_dc, dtype=torch.float32).unsqueeze(0)       # (1, 64)
        
        model_output = model(ch0, ch1, ch2, ch3)[0].numpy()  # Extract (64,) from (1, 64)
        
        # Compare
        error = np.abs(model_output - target_output)
        print(f"Sample | Model    | Target   | Error")
        print(f"-------|----------|----------|----------")
        for i in [0, 16, 32, 48, 63]:
            print(f"  {i:2d}   | {model_output[i]:8.5f} | {target_output[i]:8.5f} | {error[i]:.5f}")
        
        print(f"\nMean Absolute Error: {np.mean(error):.6f}")
        print(f"Max Error: {np.max(error):.6f}")
    
    print("\n" + "-" * 70)
    
    # Save model
    print("\n5. Saving model as TorchScript...")
    model.eval()
    
    # Create 4 example inputs - one per channel (each 64 samples)
    ex_ch0 = torch.randn(1, 64)  # audio channel
    ex_ch1 = torch.randn(1, 64)  # drive DC
    ex_ch2 = torch.randn(1, 64)  # tone DC
    ex_ch3 = torch.randn(1, 64)  # mix DC
    
    traced_model = torch.jit.trace(model, (ex_ch0, ex_ch1, ex_ch2, ex_ch3))
    
    # Add attributes required by torch.ts~ using _c (C++ interface)
    # These attributes tell torch.ts~ how many channels and what buffer size to use
    traced_model._c._register_attribute('forward_in_ch', torch.IntType.get(), 4)
    traced_model._c._register_attribute('forward_out_ch', torch.IntType.get(), 1)
    traced_model._c._register_attribute('m_buffer_size', torch.IntType.get(), 64)
    
    traced_model.save("waveshaper.ts")
    print("   ✓ Model saved as: waveshaper.ts")
    print("   ✓ Added torch.ts~ attributes:")
    print("       - forward_in_ch = 4 (audio + 3 DC parameters)")
    print("       - forward_out_ch = 1 (processed audio)")
    print("       - m_buffer_size = 64 (block size)")
    
    print("\n" + "=" * 70)
    print("HOW TO USE IN PUREDATA:")
    print("  - Model expects 4 audio input channels:")
    print("    Channel 0: audio signal")
    print("    Channel 1: drive parameter as DC signal (0-1)")
    print("    Channel 2: tone parameter as DC signal (0-1)")
    print("    Channel 3: mix parameter as DC signal (0-1)")
    print("  - Model outputs 1 audio channel (processed)")
    print("\n  PUREDATA PATCH:")
    print("    [osc~ 440] → inlet 0 (audio)")
    print("    [sig~ $1] ← drive value (0-1) → inlet 1")
    print("    [sig~ $1] ← tone value (0-1) → inlet 2")
    print("    [sig~ $1] ← mix value (0-1) → inlet 3")
    print("    ↓")
    print("    [torch.ts~ waveshaper.ts]")
    print("    ↓")
    print("    [dac~]")
    print("\n  Use [sig~] to convert numbers to DC audio signals!")
    print("=" * 70)


if __name__ == "__main__":
    train_model()


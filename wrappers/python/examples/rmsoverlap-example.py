import numpy as np
import matplotlib.pyplot as plt
import pycontorchionist as cc
import torch

def main():
    # --- Configuration ---
    sample_rate = 48000
    duration_s = 4.0
    num_samples = int(sample_rate * duration_s)

    # LFO for amplitude modulation
    lfo_freq = 0.5  # 0.5 Hz
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    lfo = (np.sin(2 * np.pi * lfo_freq * t) + 1) / 2  # Normalize to 0-1 range

    # White noise signal
    noise = np.random.normal(0, 0.5, num_samples)

    # Modulated signal
    input_signal = (noise * lfo).astype(np.float32)

    print(f"Generated a {duration_s}s signal at {sample_rate}Hz.")

    # --- RMS Processor Setup ---
    window_size = 1024
    hop_size = 512
    block_size = 64 # Simulation block size, can be different from hop_size

    print(f"RMS Processor: Window={window_size}, Hop={hop_size}, Block={block_size}")

    # Instantiate the RMSOverlap processor
    rms_processor = cc.RMSOverlap(
        initialWindowSize=window_size,
        initialHopSize=hop_size,
        initialBlockSize=block_size,
        initialWinType=cc.WindowType.HANN,
        verbose=False
    )

    # --- Processing ---
    output_signal = np.array([], dtype=np.float32)

    # Process the signal in blocks
    for i in range(0, num_samples, block_size):
        # Get a chunk of the input signal
        chunk = input_signal[i:i+block_size]

        # Post the data to the internal circular buffer of the processor
        rms_processor.post_input_data(chunk)

        # Process the data in the buffer and get an output block
        # The C++ process method expects a block size, and returns an array of that size
        output_chunk = rms_processor.process(block_size)

        # Append the processed chunk to our output signal
        output_signal = np.append(output_signal, output_chunk)

    print("Processing complete.")

    # --- Plotting ---
    # Ensure output signal has the same length as input for plotting
    if len(output_signal) > num_samples:
        output_signal = output_signal[:num_samples]
    elif len(output_signal) < num_samples:
        padding = np.zeros(num_samples - len(output_signal))
        output_signal = np.append(output_signal, padding)

    plt.figure(figsize=(12, 6))
    plt.title("RMS-processed Signal vs. Original Modulated Noise")

    # Plot original signal (and its envelope)
    plt.plot(t, input_signal, label='Original Signal', alpha=0.6)
    plt.plot(t, lfo, label='LFO Envelope (Ground Truth)', linestyle='--', color='red')

    # Plot RMS output
    plt.plot(t, output_signal, label='RMS Output', linewidth=2)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.savefig("rmsoverlap_test.png")
    print("Plot saved to rmsoverlap_test.png")
    # plt.show()

if __name__ == "__main__":
    main()

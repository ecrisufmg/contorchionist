#!/usr/bin/env python3
"""Quick diagnostics for torch.spectrail artefacts.

Loads `input.wav` and `output.wav` from the current folder, compares energy per hop,
print frames where the residual (output - input) spikes, and measures spectral
differences using the FFT settings from the Pd patch (n_fft=8192, hop_length=2048).
"""
from __future__ import annotations

import pathlib
import sys
from typing import Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
INPUT_PATH = ROOT / "input.wav"
OUTPUT_PATH = ROOT / "output.wav"
PLOTS_PATH = ROOT / "diagnostics.png"
RESIDUAL_SPEC_PATH = ROOT / "diagnostics_residual.png"

N_FFT = 8192
HOP_LENGTH = 2048
TOP_K = 15


def load(path: pathlib.Path) -> Tuple[np.ndarray, int]:
    if not path.exists():
        raise FileNotFoundError(path)
    audio, sr = librosa.load(path.as_posix(), sr=None, mono=True)
    return audio, sr


def frame_rms(signal: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Compute frame-wise RMS with same geometry as STFT."""
    if signal.size < frame_len:
        pad = frame_len - signal.size
        signal = np.pad(signal, (0, pad))
    framed = librosa.util.frame(signal, frame_length=frame_len, hop_length=hop)
    rms = np.sqrt(np.mean(framed ** 2, axis=0))
    return rms


def describe(name: str, data: np.ndarray, sr: int) -> None:
    duration = data.size / float(sr)
    print(f"{name}: sr={sr} Hz, {data.size} samples, {duration:.3f}s")
    print(f"  peak amplitude: {np.max(np.abs(data)):.6f}")
    print(f"  RMS: {np.sqrt(np.mean(data ** 2)):.6f}")


def report_spikes(residual_rms: np.ndarray, sr: int) -> None:
    times = librosa.frames_to_time(np.arange(residual_rms.size), sr=sr, hop_length=HOP_LENGTH)
    sorted_idx = np.argsort(residual_rms)[::-1]
    print(f"Top {min(TOP_K, residual_rms.size)} residual spikes (RMS, time):")
    for idx in sorted_idx[:TOP_K]:
        print(f"  frame {idx:5d} | t={times[idx]:7.4f}s | rms={residual_rms[idx]:.6f}")

    if residual_rms.size > 1:
        diffs = np.diff(sorted_idx[:TOP_K])
        if diffs.size:
            hop_time = HOP_LENGTH / float(sr)
            print("Frame distance between strongest spikes:")
            print("  ".join(f"{d} (~{d * hop_time:.4f}s)" for d in diffs))


def main() -> int:
    if not INPUT_PATH.exists() or not OUTPUT_PATH.exists():
        print("input.wav or output.wav missing", file=sys.stderr)
        return 1

    x, sr_in = load(INPUT_PATH)
    y, sr_out = load(OUTPUT_PATH)

    if sr_in != sr_out:
        print(f"Sample rates differ: input {sr_in}, output {sr_out}", file=sys.stderr)
        return 2

    sr = sr_in

    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    describe("input", x, sr)
    describe("output", y, sr)

    residual = y - x
    describe("residual", residual, sr)

    in_rms = frame_rms(x, N_FFT, HOP_LENGTH)
    out_rms = frame_rms(y, N_FFT, HOP_LENGTH)
    residual_rms = frame_rms(residual, N_FFT, HOP_LENGTH)

    print("\nMean frame RMS (input/output/residual):")
    print(f"  input   : {np.mean(in_rms):.6f}")
    print(f"  output  : {np.mean(out_rms):.6f}")
    print(f"  residual: {np.mean(residual_rms):.6f}")

    report_spikes(residual_rms, sr)

    delta = residual_rms / (in_rms + 1e-9)
    idx = np.argmax(delta)
    time = librosa.frames_to_time(idx, sr=sr, hop_length=HOP_LENGTH)
    print("\nFrame with worst residual/input ratio:")
    print(f"  frame {idx} at t={time:.4f}s | ratio={delta[idx]:.3f}")

    # Plot waveforms with hop grid
    times = np.arange(min_len) / float(sr)
    hop_times = np.arange(0, len(in_rms) + 1) * HOP_LENGTH / float(sr)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    data_list = [(x, "Input"), (y, "Output"), (residual, "Residual (output - input)")]
    for ax, (sig, title) in zip(axes, data_list):
        ax.plot(times, sig, linewidth=0.6)
        ax.set_ylabel(title)
        for t in hop_times:
            ax.axvline(t, color="gray", linewidth=0.3, alpha=0.4)
        ax.grid(True, which="both", axis="x", linestyle="--", linewidth=0.2, alpha=0.5)
    axes[-1].set_xlabel("Tempo (s)")
    fig.suptitle("torch.spectrail Diagnostics — Waveforms")
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(PLOTS_PATH, dpi=200)
    plt.close(fig)

    # Residual spectrogram
    S_residual = np.abs(librosa.stft(residual, n_fft=N_FFT, hop_length=HOP_LENGTH))
    S_residual_db = librosa.amplitude_to_db(S_residual, ref=np.max)

    fig_spec, ax_spec = plt.subplots(figsize=(12, 6))
    img = librosa.display.specshow(
        S_residual_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="linear",
        ax=ax_spec,
    )
    ax_spec.set_title("Residual Spectrogram (linear freq scale)")
    ax_spec.set_xlabel("Tempo (s)")
    ax_spec.set_ylabel("Frequência (Hz)")
    fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB")
    fig_spec.tight_layout()
    fig_spec.savefig(RESIDUAL_SPEC_PATH, dpi=200)
    plt.close(fig_spec)

    print(f"Waveform plot saved to {PLOTS_PATH}")
    print(f"Residual spectrogram saved to {RESIDUAL_SPEC_PATH}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

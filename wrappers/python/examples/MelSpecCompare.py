import sys
import os

# Adiciona o diretório do script ao sys.path para encontrar o módulo
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import pycontorchionist as cc
except ImportError:
    print("Erro: Não foi possível importar o módulo pycontorchionist.")
    print("Verifique se o arquivo do módulo (pycontorchionist...so ou .pyd) está no mesmo diretório que este script.")
    sys.exit(1)

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr



# ====== CONFIGURAÇÕES ======
import os
AUDIO_PATH = os.path.join(os.path.dirname(__file__), "audio", "Vn-ord_crush-A#3-mf-4c-R200d.wav")
SAMPLE_RATE = 44100
N_MELS = 64
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 0.0
FMAX = SAMPLE_RATE // 2
MEL_NORM = "slaney"
MEL_MODE = cc.MelNormMode.ENERGY_POWER
print(f"Parâmetros: {SAMPLE_RATE}, {N_MELS}, {N_FFT}, {HOP_LENGTH}, {FMIN}, {FMAX}, {MEL_NORM}, {MEL_MODE} ({cc.mel_norm_mode_to_string(MEL_MODE)})")

# # ====== CARREGAR ÁUDIO ======
waveform, sr = torchaudio.load(AUDIO_PATH)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)  # Mono
waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
audio = waveform.squeeze().numpy()

print(f"Áudio carregado: {audio.shape}, SR={SAMPLE_RATE}")

# ====== MEL SPECTROGRAM PYTORCH ======
mel_torch = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=FMIN,
    f_max=FMAX,
    norm=MEL_NORM,
    power=2.0,
)(torch.tensor(audio)).numpy()

print(f"MelSpectrogram PyTorch: {mel_torch.shape}")

# ====== MEL SPECTROGRAM CONTORCHIONIST ======
processor = cc.MelSpectrogramProcessor()
processor.set_sample_rate(float(SAMPLE_RATE))
processor.set_n_fft(N_FFT)
processor.set_hop_length(HOP_LENGTH)
processor.set_n_mels(N_MELS)
processor.set_fmin_mel(FMIN)
processor.set_fmax_mel(FMAX)
processor.set_filterbank_norm(MEL_NORM)
processor.set_mel_norm_mode(MEL_MODE)

# print(processor.process)
# help(processor.process)
print("Processor carregado")

## Processar em blocos
frames = []
block_size = HOP_LENGTH
for i in range(0, len(audio), block_size):
    chunk = audio[i:i+block_size].astype(np.float32)
    mel = processor.process(chunk)
    if mel is not None:
        frames.append(np.array(mel))
mel_contorch = np.stack(frames, axis=1) if frames else np.zeros((N_MELS, 1))
print(f"MelSpectrogram Contorchionist: {mel_contorch.shape}")

# ====== AJUSTAR FORMATO ======
# Ambos: (n_mels, n_frames)
min_frames = min(mel_torch.shape[1], mel_contorch.shape[1])
mel_torch = mel_torch[:, :min_frames]
mel_contorch = mel_contorch[:, :min_frames]

# ====== MÉTRICAS ======
mse = np.mean((mel_torch - mel_contorch) ** 2)
mae = np.mean(np.abs(mel_torch - mel_contorch))
corr = np.mean([
    pearsonr(mel_torch[i], mel_contorch[i])[0]
    for i in range(N_MELS)
    if np.std(mel_torch[i]) > 0 and np.std(mel_contorch[i]) > 0
])

print(f"\nMétricas comparativas:")
print(f"  MSE: {mse:.6f}")
print(f"  MAE: {mae:.6f}")
print(f"  Correlação média: {corr:.4f}")

# ====== PLOT ======
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(10 * np.log10(mel_torch + 1e-8), aspect='auto', origin='lower')
plt.title("PyTorch/Torchaudio")
plt.xlabel("Frames")
plt.ylabel("Mel bins")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(10 * np.log10(mel_contorch + 1e-8), aspect='auto', origin='lower')
plt.title("Contorchionist Core")
plt.xlabel("Frames")
plt.ylabel("Mel bins")
plt.colorbar()

plt.suptitle("Comparação MelSpectrogram")
plt.tight_layout()
plt.show()

import torch
import pycontorchionist as cc
import numpy as np

# Configuração
sample_rate = 48000
n_fft = 2048
hop_length = 512
n_mels = 140

processor = cc.MelSpectrogramProcessor(n_fft, hop_length, -1, cc.WindowType.HANN, cc.NormalizationType.POWER, cc.SpectrumDataFormat.POWERPHASE, sample_rate, n_mels, 0.0, -1.0, cc.MelFormulaType.CALC2, 'slaney', cc.MelNormMode.ENERGY_POWER, torch.device('cpu'), False)

print(f'⚙️  Configuração: SR={processor.get_sample_rate()}, n_fft={processor.get_n_fft()}, hop={processor.get_hop_length()}, n_mels={processor.get_n_mels()}')

# Gerar sinal sintético (tom puro 440Hz)
duration = 0.5  # 0.5 segundos
t = np.linspace(0, duration, int(sample_rate * duration), False)
frequency = 440  # Hz
audio_data = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

print(f'🎵 Sinal gerado: {len(audio_data)} amostras, {duration}s, {frequency}Hz')

## block processing
total_frames = 0
for i in range(0, len(audio_data), hop_length):
    chunk = audio_data[i:i+hop_length]
    result = processor.process(chunk)
    print(result)
    if result is not None:
        total_frames += 1
        if total_frames <= 3:  # Mostrar apenas os primeiros 3 frames
            energy = np.sum(result)
            peak_mel = np.argmax(result)
            print(f'📊 Frame {total_frames}: energia={energy:.2f}, peak_mel={peak_mel}')

print(f'✅ Processamento concluído! {total_frames} frames extraídos')

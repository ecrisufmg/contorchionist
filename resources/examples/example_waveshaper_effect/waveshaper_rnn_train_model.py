#!/usr/bin/env python3
"""
01_Waveshaper_train_model_RNN.py
================================
Trains a model that processes audio blocks (64 samples) with a waveshaper effect.

Esta versão usa uma arquitetura RNN (GRU) para aprender dependências temporais
e garantir a continuidade entre as amostras e blocos.

Input interface (TORCH.TS~ COMPATIBLE):
    - audio: (N, 64) tensor with audio samples
    - drive: (N, 1) tensor with distortion amount 0..1 (maps to gain 1..10)
    - tone: (N, 1) tensor with brightness 0..1 (0=cut highs, 0.5=flat, 1=boost highs)
    - mix: (N, 1) tensor with dry/wet blend 0..1 (0=dry, 1=fully wet)

Output:
    - processed: (N, 64) tensor with processed audio samples
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

FS = 48000.0  # Sampling rate
BLOCK_SIZE = 64  # Audio block size


def _param_to_control(x: torch.Tensor, block_size: int) -> torch.Tensor:
    # se for um único valor: reshape [1,1]
    if x.dim() == 0:
        return x.view(1, 1)
    
    # se for 1D [block_size] ou [1]: reshape para [1, block_size] ou [1,1]
    if x.dim() == 1:
        if x.size(0) == block_size:
            x = x.view(1, block_size)
            # pega o primeiro sample como controle
            return x[:, 0:1]
        # caso [1]: reshape para [1,1]
        x = x.unsqueeze(1)

    # se for 2D [1, block_size]: retorna como está
    if x.size(1) == block_size:
        # bloco DC -> pega o primeiro sample como controle
        return x[:, 0:1]
    if x.size(1) == 1:
        return x
    raise RuntimeError("Parameter tensor has invalid shape (expected scalar, (N,1) or (N,block_size))")

class WaveshaperModelRNN(nn.Module):
    def __init__(self, hidden_size=32):
        super(WaveshaperModelRNN, self).__init__()
        
        # 4 features por amostra: (audio, drive, tone, mix)
        input_size = 4
        
        # A GRU processa a sequência. 
        # batch_first=True -> espera (N, Seq, Features) = (N, 64, 4)
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=2,       # Duas camadas para mais profundidade
            batch_first=True,   # IMPORTANTE
            bidirectional=False # Não podemos olhar para o futuro em tempo real
        )
        
        # Camada de saída: mapeia o estado oculto da GRU para 1 amostra de áudio
        self.fc_out = nn.Linear(hidden_size, 1)
        
        # Ativação
        self.act = nn.SiLU()

    def forward(self, ch0: torch.Tensor, ch1: torch.Tensor, ch2: torch.Tensor, ch3: torch.Tensor) -> torch.Tensor:
        """
        Forward pass com arquitetura Recorrente (GRU).
        """
        if ch0.dim() == 1:
            ch0 = ch0.unsqueeze(0)
        
        N = ch0.size(0) # Batch size
        bs = ch0.size(1) # Block size (64)

        # 1. Preparar parâmetros (como você já fazia)
        # _param_to_control retorna [N, 1]
        drive = _param_to_control(ch1, bs).expand(N, 1)
        tone = _param_to_control(ch2, bs).expand(N, 1)
        mix = _param_to_control(ch3, bs).expand(N, 1)

        # 2. Preparar a sequência de entrada para a GRU
        
        # Áudio: [N, 64] -> [N, 64, 1]
        audio_seq = ch0.unsqueeze(2) 
        
        # Parâmetros: [N, 1] -> [N, 64, 1] (repetir 64x)
        drive_seq = drive.unsqueeze(1).expand(N, bs, 1)
        tone_seq = tone.unsqueeze(1).expand(N, bs, 1)
        mix_seq = mix.unsqueeze(1).expand(N, bs, 1)
        
        # Concatenar características ao longo da última dimensão
        # [N, 64, 1] + 3x[N, 64, 1] -> [N, 64, 4]
        features = torch.cat([audio_seq, drive_seq, tone_seq, mix_seq], dim=2)
        
        # 3. Passar pela GRU
        # A GRU processa a sequência de 64 amostras.
        # Nós não passamos o estado oculto (h_0) explicitamente, 
        # então ele começa em zero para cada bloco (que é o comportamento 
        # esperado pelo torch.ts~, que é stateless).
        rnn_out, _ = self.gru(features)
        
        # Aplicar ativação na saída da GRU
        rnn_out = self.act(rnn_out)
        
        # 4. Camada de saída
        # Mapeia de [N, 64, hidden_size] -> [N, 64, 1]
        output_seq = self.fc_out(rnn_out)
        
        # 5. Formatar saída final
        # [N, 64, 1] -> [N, 64]
        output = output_seq.squeeze(2)
        
        # Aplicar tanh final para garantir o range
        output = torch.tanh(output)
        
        return output


def process_long_signal_target_stateful(long_signal: np.ndarray, drive: float, tone: float, mix: float) -> np.ndarray:
    """
    Função alvo STATEFUL. Processa um sinal longo de uma vez, mantendo
    a continuidade do estado do filtro de 'tone'.
    """
    # 1. Drive
    gain = 1.0 + drive * 9.0
    driven = long_signal * gain
    
    # 2. Waveshaping
    shaped = np.tanh(driven)
    
    # 3. Tone control (STATEFUL)
    # diff[n] = shaped[n] - shaped[n-1]
    shaped_prev = np.roll(shaped, 1)
    shaped_prev[0] = 0.0  # Única descontinuidade é no início do sinal longo
    
    diff = shaped - shaped_prev
    
    tone_mapped = (tone - 0.5) * 2.0 # 0..1 -> -1..1
    toned = shaped + tone_mapped * 0.3 * diff
    
    # 4. Mix
    output = mix * toned + (1.0 - mix) * long_signal
    
    return output.astype(np.float32)


def create_synthetic_data(num_examples=5000):
    """
    Cria dados sintéticos 100% contínuos.
    Gera sinais longos, processa-os de forma stateful, e DEPOIS
    fatia em blocos de 64 amostras.
    """
    print(f"   Gerando {num_examples} exemplos de treino (100% blocos consecutivos)...")
    
    ch0_list = []  # audio
    ch1_list = []  # drive DC
    ch2_list = []  # tone DC
    ch3_list = []  # mix DC
    targets = []
    
    # Contador para quantos blocos já geramos
    generated_count = 0
    
    while generated_count < num_examples:
        
        # 1. Gerar um sinal longo
        long_length = BLOCK_SIZE * np.random.randint(20, 50) # Sinais bem longos
        t_long = np.arange(long_length) / FS
        
        freq1 = np.random.uniform(100, 5000)
        freq2 = np.random.uniform(100, 5000)
        amp1 = np.random.uniform(0.1, 0.8)
        amp2 = np.random.uniform(0.1, 0.4)
        
        long_signal = amp1 * np.sin(2 * np.pi * freq1 * t_long) + amp2 * np.sin(2 * np.pi * freq2 * t_long)
        
        noise_amp = np.random.uniform(0.0, 0.1)
        long_signal += noise_amp * np.random.randn(long_length)
        
        long_signal = long_signal / (np.max(np.abs(long_signal)) + 1e-8) * np.random.uniform(0.3, 0.9)

        # 2. Gerar parâmetros (constantes para todo o sinal longo)
        drive = np.random.uniform(0.0, 1.0)
        tone = np.random.uniform(0.0, 1.0)
        mix = np.random.uniform(0.0, 1.0)

        # 3. Processar o sinal longo INTEIRO statefully
        long_target = process_long_signal_target_stateful(long_signal, drive, tone, mix)
        
        # 4. Fatiar o sinal longo e o alvo longo em blocos alinhados
        for i in range(0, long_length - BLOCK_SIZE, BLOCK_SIZE):
            if generated_count >= num_examples:
                break
                
            signal_block = long_signal[i : i + BLOCK_SIZE]
            target_block = long_target[i : i + BLOCK_SIZE]
            
            # Criar canais DC
            drive_dc = np.full(BLOCK_SIZE, drive, dtype=np.float32)
            tone_dc = np.full(BLOCK_SIZE, tone, dtype=np.float32)
            mix_dc = np.full(BLOCK_SIZE, mix, dtype=np.float32)
            
            # Armazenar
            ch0_list.append(signal_block)
            ch1_list.append(drive_dc)
            ch2_list.append(tone_dc)
            ch3_list.append(mix_dc)
            targets.append(target_block)
            
            generated_count += 1
            
        if (generated_count // 1000) > ((generated_count - (long_length // BLOCK_SIZE)) // 1000):
             print(f"   ... {generated_count}/{num_examples} exemplos gerados")

    print(f"   ✓ Todos os {num_examples} exemplos gerados!")
    
    # Convert to tensors
    ch0 = torch.from_numpy(np.array(ch0_list, dtype=np.float32))  # (N, 64)
    ch1 = torch.from_numpy(np.array(ch1_list, dtype=np.float32))  # (N, 64)
    ch2 = torch.from_numpy(np.array(ch2_list, dtype=np.float32))  # (N, 64)
    ch3 = torch.from_numpy(np.array(ch3_list, dtype=np.float32))  # (N, 64)
    targets_tensor = torch.from_numpy(np.array(targets, dtype=np.float32))  # (N, 64)
    
    return ch0, ch1, ch2, ch3, targets_tensor


def train_model():
    print("=" * 70)
    print("Training Waveshaper Effect (RNN/GRU Architecture)")
    print("=" * 70)
    print("Input:  4 separate channel tensors (audio, drive_dc, tone_dc, mix_dc)")
    print("Output: 1 channel tensor (processed audio)")
    print(f"Block size: {BLOCK_SIZE}, Sampling rate: {FS:.0f} Hz")
    print("MODELO: nn.GRU para continuidade temporal.")
    print("DADOS: 100% contínuos e stateful.")
    print("LOSS: Apenas MSE (continuidade implícita nos dados).")
    print("=" * 70)
    
    model = WaveshaperModelRNN()
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
    num_epochs = 500 
    model.train()
    
    # Train with mini-batches
    batch_size = 64
    n = train_ch0.shape[0]
    
    print(f"   Using mini-batches of size {batch_size}")
    print(f"   Total batches per epoch: {n // batch_size}")
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 70  # Early stopping
    
    for epoch in range(num_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            batch_ch0 = train_ch0[idx]  # (batch, 64)
            batch_ch1 = train_ch1[idx]  # (batch, 64)
            batch_ch2 = train_ch2[idx]  # (batch, 64)
            batch_ch3 = train_ch3[idx]  # (batch, 64)
            batch_y = train_targets[idx]  # (batch, 64)
            
            outputs = model(batch_ch0, batch_ch1, batch_ch2, batch_ch3)  # (batch, 64)
            
            # Main reconstruction loss
            # A continuidade agora é aprendida via MSE, 
            # pois os DADOS ALVO são contínuos.
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_ch0.size(0)
        
        epoch_loss /= n
        
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
            print(f"   Epoch [{epoch+1}/{num_epochs}], MSE: {epoch_loss:.6f}")
        elif (epoch + 1) % 10 == 0:
            # More frequent but less verbose feedback
            print(f"   ... epoch {epoch+1}/{num_epochs} (mse: {epoch_loss:.6f})")
    
    print(f"\n3. Training completed! Final MSE: {epoch_loss:.6f}, Best MSE: {best_loss:.6f}")
    
    # Test: process a simple sine wave block
    print("\n4. Testing with a 440Hz sine wave block")
    print("-" * 70)
    model.eval()
    
    # Generate test signal (agora vamos gerar um sinal LONGO e pegar o SEGUNDO bloco)
    # Isso simula o uso real e testa a continuidade.
    t_long = np.arange(BLOCK_SIZE * 2) / FS
    test_signal_long = 0.5 * np.sin(2 * np.pi * 440.0 * t_long)
    
    # Pegamos o segundo bloco para o teste
    test_signal_block = test_signal_long[BLOCK_SIZE:]
    
    drive, tone, mix = 0.7, 0.5, 1.0
    print(f"Parameters: drive={drive}, tone={tone}, mix={mix}\n")
    
    with torch.no_grad():
        # Target (processa o sinal longo e pega o segundo bloco)
        target_output_long = process_long_signal_target_stateful(test_signal_long, drive, tone, mix)
        target_output = target_output_long[BLOCK_SIZE:]
        
        # Model - prepara 4 canais
        drive_dc = np.full(BLOCK_SIZE, drive, dtype=np.float32)
        tone_dc = np.full(BLOCK_SIZE, tone, dtype=np.float32)
        mix_dc = np.full(BLOCK_SIZE, mix, dtype=np.float32)
        
        ch0 = torch.tensor(test_signal_block, dtype=torch.float32).unsqueeze(0)  # (1, 64)
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
    
    print("\n" + "-" * 70)
    
    # Save model WEIGHTS ONLY
    print("\n5. Saving model weights...")
    model.eval()

    # Salva apenas o state_dict (pesos)
    dir_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(dir_path, "waveshaper_rnn_weights.pt") # Novo nome para os pesos
    torch.save(model.state_dict(), save_path)
    
    print(f"   ✓ Model weights saved to: {save_path}")
    print("\n" + "=" * 70)
    print("Training complete. Now run the export_wrapper.py script.")
    print("=" * 70)


if __name__ == "__main__":
    train_model()
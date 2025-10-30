#!/usr/bin/env python3
"""
export_rnn_wrapper.py
======================
Carrega o modelo RNN treinado (apenas os pesos) e o envolve em um
Wrapper compatível com torch.ts~.

Este wrapper expõe métodos (@torch.jit.export) como `drive()`, `tone()`,
e `mix()`, permitindo que os parâmetros sejam controlados por mensagens
no Pure Data, em vez de sinais de áudio DC.

O método `forward` do wrapper recebe APENAS o áudio (Inlet 0) e usa
os parâmetros armazenados internamente para processar o bloco.
"""

import torch
import torch.nn as nn
from typing import List
import os

# Importa a arquitetura do modelo RNN definida no script de treinamento
try:
    from waveshaper_rnn_train_model import WaveshaperModelRNN
except ImportError:
    print("ERRO: Não foi possível encontrar o arquivo 'waveshaper_rnn_train.py'.")
    print("Certifique-se de que o script de treinamento esteja na mesma pasta.")
    exit(1)


#--- Wrapper compatível com torch.ts~ ---#
class WaveshaperWrapper(nn.Module):
    def __init__(self, waveshaper: nn.Module, drive: float, tone: float, mix: float, m_buffer_size=64) -> None:
        super().__init__()
        # Armazena o modelo RNN treinado
        self.waveshaper = waveshaper 
        
        # Parâmetros internos que serão controlados por mensagens
        self._drive = float(drive)
        self._tone = float(tone)
        self._mix = float(mix)
        
        # Metadados para torch.ts~
        self._methods = ["forward", "drive", "tone", "mix"]
        self._attributes = ["drive", "tone", "mix",
                            "forward_in_ch", "forward_out_ch",
                            "m_buffer_size", "max_buffer_size"]
        
        # Define os atributos que torch.ts~ espera
        self.forward_in_ch = 1  # Agora só 1 entrada de ÁUDIO
        self.forward_out_ch = 1
        self.m_buffer_size = m_buffer_size
        self.max_buffer_size = 8192
 
    
    @torch.jit.export
    def get_methods(self) -> List[str]:
        return self._methods

    @torch.jit.export
    def get_attributes(self) -> List[str]:
        return self._attributes
    
    
    @torch.jit.export
    def drive(self, drive: float):
        """Define o 'drive' dinamicamente (controlado por mensagem)"""
        if drive < 0 or drive > 1:
            return
        self._drive = float(drive)
        # print(f"Drive set to {self._drive}") # Opcional: bom para debug
        
    @torch.jit.export
    def tone(self, tone: float):
        """Define o 'tone' dinamicamente (controlado por mensagem)"""
        if tone < 0 or tone > 1:
            return
        self._tone = float(tone)
        # print(f"Tone set to {self._tone}") # Opcional: bom para debug

    @torch.jit.export
    def mix(self, mix: float):
        """Define o 'mix' dinamicamente (controlado por mensagem)"""
        if mix < 0 or mix > 1:
            return
        self._mix = float(mix)
        # print(f"Mix set to {self._mix}") # Opcional: bom para debug

    @torch.jit.export
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Processa um bloco de áudio de entrada"""

        # Garante que o input seja [N, 64]
        if input.dim() == 1:
            ch0 = input.unsqueeze(0)
        else:
            ch0 = input

        batch, bs = ch0.shape[0], ch0.shape[1]
        device = ch0.device
        dtype = ch0.dtype

        # Cria os tensores de controle (parâmetros) com base nos valores
        # ARMAZENADOS internamente (_drive, _tone, _mix)
        
        # O modelo RNN espera 4 tensores. O _param_to_control dentro dele
        # aceita [N, 1], então é isso que criamos.
        drive_tensor = torch.full((batch, 1), float(self._drive), device=device, dtype=dtype)
        tone_tensor = torch.full((batch, 1), float(self._tone),  device=device, dtype=dtype)
        mix_tensor = torch.full((batch, 1), float(self._mix),   device=device, dtype=dtype)

        # Chama o modelo RNN interno com os 4 tensores necessários
        output = self.waveshaper(ch0, drive_tensor, tone_tensor, mix_tensor)
        
        return output

 
# --- Script Principal: Exportar o Modelo --- #

print("=" * 70)
print("Exportando Waveshaper RNN Wrapper para torch.ts~")
print("=" * 70)

# 1. Instanciar a arquitetura do modelo RNN
model = WaveshaperModelRNN() 
model.eval()

# 2. Carregar os pesos treinados
dir_path = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(dir_path, "waveshaper_rnn_weights.pt")
    
try:
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    print(f"✓ Pesos treinados carregados de: {weights_path}")
except FileNotFoundError:
    print(f"!!! ATENÇÃO: Arquivo de pesos '{weights_path}' não encontrado.")
    print("    Continuando com pesos aleatórios (apenas para teste).")
except Exception as e:
    print(f"ERRO ao carregar pesos: {e}")
    exit(1)

# 3. "Scriptar" o modelo RNN (o modelo interno)
# Usar torch.jit.script é mais robusto para RNNs e lógica de controle
try:
    scripted_model = torch.jit.script(model)
    print("✓ Modelo RNN interno 'scriptado' com sucesso.")
except Exception as e:
    print(f"ERRO ao scriptar o modelo RNN: {e}")
    exit(1)


# 4. Criar o Wrapper com parâmetros padrão
waveshaper_wrapper = WaveshaperWrapper(
    scripted_model, 
    drive=0.5, 
    tone=0.5, 
    mix=1.0
)

# 5. "Scriptar" o Wrapper (o objeto final)
try:
    scripted_wrapper = torch.jit.script(waveshaper_wrapper)
    print("✓ Wrapper final 'scriptado' com sucesso.")
except Exception as e:
    print(f"ERRO ao scriptar o Wrapper: {e}")
    exit(1)

# 6. Salvar o Wrapper scriptado
output_filename = "waveshaper_rnn.ts"
output_path = os.path.join(dir_path, output_filename)
scripted_wrapper.save(output_path)
print(f"✓ Modelo final salvo em: {output_path}")


# 7. Testar o modelo salvo
print("\n" + "-" * 70)
print("Testando o modelo salvo...")
loaded_model = torch.jit.load(output_path)
loaded_model.eval()

# Entrada fictícia
dummy_input = torch.randn(1, 64)
flat_input = dummy_input.view(-1) # Simula entrada 1D do PD

print(f"Métodos registrados: {loaded_model.get_methods()}")
print(f"Atributos registrados: {loaded_model.get_attributes()}")

# --- Teste de chamada de métodos ---
print("\n------ Testando chamadas de método ------")
print("Definindo drive=0.8, tone=0.2, mix=1.0")
loaded_model.drive(0.8) 
loaded_model.tone(0.2)
loaded_model.mix(1.0) 

with torch.no_grad():
    output = loaded_model.forward(flat_input)

print(f"Shape do input: {flat_input.shape}")
print(f"Shape da saída: {output.shape}")
assert output.shape == (1, 64)
print("✓ Teste de forward e métodos concluído com sucesso.")

print("\n" + "=" * 70)
print("COMO USAR NO PUREDATA:")
print(f"  [torch.ts~ {output_filename}]")
print("  Inlet 0: Áudio")
print("  Inlets 1-3: Mensagens (ex: [drive 0.8(, [tone 0.2(, [mix 1.0()")
print("=" * 70)
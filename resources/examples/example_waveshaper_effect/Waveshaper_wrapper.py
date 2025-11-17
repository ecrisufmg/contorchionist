import torch
import torch.nn as nn
from typing import List
from Waveshaper_train_model import WaveshaperModel
import os


#--- waveshaper wrapper for torch.ts~ ---#
class WaveshaperWrapper(nn.Module):
    def __init__(self, waveshaper: nn.Module, drive: float, tone: float, mix: float, m_buffer_size=64) -> None:
        super().__init__()
        self.waveshaper = waveshaper
        self._drive = float(drive)
        self._tone = float(tone)
        self._mix = float(mix)
        self._methods = ["forward", "drive", "tone", "mix"]
        self._attributes = ["drive", "tone", "mix",
                            "forward_in_ch", "forward_out_ch",
                            "m_buffer_size", "max_buffer_size"]
        
        self.forward_in_ch = 1
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
        """set drive dynamically"""
        if drive < 0 or drive > 1:
            return
        self._drive = float(drive)
        print(f"Drive set to {self._drive}")
        
    @torch.jit.export
    def tone(self, tone: float):
        """set tone dynamically"""
        if tone < 0 or tone > 1:
            return
        self._tone = float(tone)
        print(f"Tone set to {self._tone}")


    @torch.jit.export
    def mix(self, mix: float):
        """set mix dynamically"""
        if mix < 0 or mix > 1:
            return
        self._mix = float(mix)
        print(f"Mix set to {self._mix}")

    @torch.jit.export
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """process input block with waveshaper effect"""

        if input.dim() == 1:
            ch0 = input.unsqueeze(0)
        else:
            ch0 = input

        batch, bs = ch0.shape[0], ch0.shape[1]
        device = ch0.device
        dtype = ch0.dtype

        # create control tensors (N,1) — model slices [:,0:1], so (N,1) é suficiente
        drive = torch.full((batch, 1), float(self._drive), device=device, dtype=dtype)
        tone = torch.full((batch, 1), float(self._tone),  device=device, dtype=dtype)
        mix = torch.full((batch, 1), float(self._mix),   device=device, dtype=dtype)

        output = self.waveshaper(input, drive, tone, mix)
        return output

 

# ---  export wavshaper model (torchscript) --- #
model = WaveshaperModel()

# load trained weights
weights_path = "model.pt"
    
try:
    model.load_state_dict(torch.load(weights_path))
    print(f"Trained weights loaded from {weights_path}")
except FileNotFoundError:
    print(f"Warning: Weights file '{weights_path}' not found.")
    print("Continuing with random weights for testing.")
except Exception as e:
    print(f"Error loading weights: {e}")

# script the model
model.eval()
scripted_model = torch.jit.script(model)

# create the wrapper with default parameters
waveshaper_wrapper = WaveshaperWrapper(scripted_model, drive=0.5, tone=0.5, mix=1.0)
scripted_wrapper = torch.jit.script(waveshaper_wrapper)

# save the scripted wrapper
dir_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(dir_path, "waveshaper.ts")
scripted_wrapper.save(output_path)
print(f"Waveshaper wrapper saved to {output_path}")


# test the torchscript model
print("\nIniciando teste de geração...")
# carrega o modelo salvo
loaded_model = torch.jit.load(output_path)
loaded_model.eval()

# entrada fictícia para teste
dummy_input = torch.randn(1, 64)
flat_input = dummy_input.view(-1)
print("Métodos registrados:", loaded_model.get_methods())
print("Atributos registrados:", loaded_model.get_attributes())

# --- Teste ---
print("\n------ Testando ------")
loaded_model.drive(0.5) # seta drive para 0.5
loaded_model.tone(0.7)  # seta tone para 0.7
loaded_model.mix(0.8)   # seta mix para 0.8
with torch.no_grad():
    out_flat_random = loaded_model.forward(flat_input)

print(f"Shape do input (features de entrada): {dummy_input.shape}")
print(f"Shape da saída (features de saída): {out_flat_random.shape}") 
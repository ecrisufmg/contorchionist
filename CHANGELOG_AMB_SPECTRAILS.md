# Changelog: torch.amb.spectrails~ - Correção de Espacialização

## Data: 21 de Novembro de 2025

## [NOVO] Adição de Suporte a GPU (21/11/2025)

O objeto `torch.amb.spectrails~` agora aceita processamento via GPU usando as flags `-d` ou `-device`, consistente com outros objetos da biblioteca.

### Uso

```
[torch.amb.spectrails~ -order 1 -device cuda]
[torch.amb.spectrails~ -order 1 -d mps]
[torch.amb.spectrails~ -order 2 -d cuda:0]
```

### Implementação

**Arquivo:** `wrappers/puredata/src/torch.amb.spectrails~.cpp`

- Adicionado include de `pd_torch_device_adapter.h`
- Adicionado membro `torch::Device device_` à estrutura do objeto
- Parsing de argumentos `-device` / `-d` no construtor
- Processadores criados com device especificado
- Mensagem de inicialização mostra device ativo

### Correção de Conflito de Aliases

**IMPORTANTE:** Removidos aliases curtos que conflitavam com `-d` (device):
- `threshold`: mantém `-threshold` e `-thresh` (removido `-t`)
- `attack`: mantém `-attack` e `-att` (removido `-a`)
- `decay`: mantém `-decay` e `-dec` (removido `-d`)
- `decaytime`: mantém `-decaytime`, `-decayt` e `-dtime` (removido `-dt`)

Agora `-d` é exclusivamente para `device`.

### Correção de Crash com MPS/CUDA

**CRÍTICO:** Adicionado `.to(torch::kCPU)` antes de `memcpy` dos tensores de saída.

**Problema:** Quando processando com MPS ou CUDA, os tensores ficam em memória GPU. Tentar fazer `memcpy` direto desses tensores para buffers CPU causava `EXC_BAD_ACCESS (SIGSEGV)`.

**Solução:** Garantir que todos os tensores de saída sejam movidos para CPU antes de copiar para os buffers de Pure Data:
```cpp
auto w_mag_out = w_outputs[0].to(torch::kCPU).contiguous();
auto w_phase_out = w_outputs[1].to(torch::kCPU).contiguous();
```

O suporte funciona tanto para CUDA quanto MPS (macOS), seguindo o mesmo padrão dos objetos `torch.rfft~`, `torch.linear~`, `torch.mha`, etc.

---

## Problema Identificado

O objeto `torch.amb.spectrails~` estava detectando picos espectrais corretamente no canal W (omnidirecional), mas a **espacialização era perdida no decoding** Ambisonics. A localização sonora ficava difusa ou incorreta.

### Causa Raiz

Cada canal (W, X, Y, Z) processava picos **independentemente** com base em seu próprio threshold de magnitude:

```cpp
bool significant_increase = curr_mag > (out_mag_acc[i] * 1.1f);
```

**Resultado:**
- W detectava e sustentava um pico no bin 150
- X não passava no teste de 10% de aumento → **não sustentava**
- Y tinha fase diferente → sustentava com valores dessincronizados
- **Espacialização quebrada**: energia em W sem correspondência correta em X,Y,Z

## Solução Implementada

### 1. Novo Método no Core: `force_write_peak()`

**Arquivo:** `core/include/core_ap_spectrails.h`

Adicionado método público que permite **forçar a escrita** de um valor magnitude/fase em um bin específico, **ignorando** o threshold de detecção:

```cpp
void force_write_peak(int bin_idx, float magnitude, float phase) {
    if (bin_idx < 0 || bin_idx >= num_bins_) return;
    
    auto out_mag_acc = out_mag_.accessor<float, 1>();
    auto out_phase_acc = out_phase_.accessor<float, 1>();
    auto env_acc = envelope_positions_.accessor<float, 1>();
    
    out_mag_acc[bin_idx] = magnitude;
    out_phase_acc[bin_idx] = phase;
    env_acc[bin_idx] = 1.0f;  // Reset envelope para novo pico
}
```

**Finalidade:** Permitir que canais direcionais (X,Y,Z) escrevam seus valores **exatamente no momento** em que W detecta um pico, preservando a coerência espacial.

### 2. Processamento Sincronizado no PureData External

**Arquivo:** `wrappers/puredata/src/torch.amb.spectrails~.cpp`

**Antes (❌ Processamento Independente):**
```cpp
// Cada canal processava sozinho
for (int ch = 0; ch < x->num_channels_; ++ch) {
    auto outputs = x->processors_[ch]->process_frame(mag_tensor, phase_tensor);
}
```

**Depois (✅ Processamento Coordenado):**
```cpp
// 1. W processa e detecta picos
auto w_outputs = x->processors_[0]->process_frame(w_mag_tensor, w_phase_tensor);
auto w_peaks = x->processors_[0]->get_detected_peaks();

// 2. Propaga envelope para sincronizar decay
auto w_envelope = x->processors_[0]->get_envelope_positions();
for (int ch = 1; ch < x->num_channels_; ++ch) {
    x->processors_[ch]->set_envelope_positions(w_envelope);
}

// 3. Para cada pico detectado em W, FORÇA escrita em X,Y,Z
auto w_peaks_acc = w_peaks.accessor<float, 1>();
for (size_t bin = 0; bin < num_bins; ++bin) {
    if (w_peaks_acc[bin] > 0.0f) {  // W detectou pico neste bin
        for (int ch = 1; ch < x->num_channels_; ++ch) {
            float mag = in_mags[ch][bin];
            float phase = in_phases[ch][bin];
            x->processors_[ch]->force_write_peak(bin, mag, phase);
        }
    }
}

// 4. Processa canais direcionais normalmente
for (int ch = 1; ch < x->num_channels_; ++ch) {
    auto outputs = x->processors_[ch]->process_frame(...);
}
```

## Resultado Esperado

### Antes
- ✅ Picos detectados em W
- ❌ Espacialização perdida/difusa
- ❌ Correlação com fonte original inconsistente

### Depois
- ✅ Picos detectados em W
- ✅ **Todos os canais (W,X,Y,Z) escrevem valores simultaneamente**
- ✅ **Fase relativa preservada** (valores exatos do momento do pico)
- ✅ **Localização espacial mantida** no decoding Ambisonics
- ✅ Correlação com fonte original estável

## Conceito Chave: Sample & Hold Complexo

A solução implementa o conceito de **Sample & Hold dos números complexos** (magnitude + fase) de todos os canais simultaneamente:

1. **Trigger:** W detecta pico no bin 150
2. **Freeze:** Todos os canais escrevem seus valores **exatos** naquele momento:
   - W: mag=0.8, phase=1.2
   - X: mag=0.3, phase=0.5 (pode ser menor, mas preserva relação!)
   - Y: mag=-0.2, phase=3.1 (pode ser negativo - fase invertida!)
   - Z: mag=0.1, phase=2.0
3. **Decay:** Envelope sincronizado reduz amplitude, mas mantém **proporções relativas**

## Arquivos Modificados

1. `core/include/core_ap_spectrails.h`
   - Adicionado: `force_write_peak(int bin_idx, float magnitude, float phase)`

2. `wrappers/puredata/src/torch.amb.spectrails~.cpp`
   - Modificado: `torch_amb_spectrails_tilde_perform()`
   - Nova lógica: processamento coordenado com força de escrita sincronizada

## Próximos Passos

1. Compilar e testar
2. Verificar se a espacialização está preservada
3. Ajustar parâmetros de threshold/attack/decay se necessário
4. Validar com diferentes ordens ambisonics (1ª, 2ª, 3ª ordem)

# Memória de Desenvolvimento: torch.amb.spectrails~

## Contexto do Projeto

Desenvolvimento de um objeto PureData para processamento espectral ambisônico com sustentação de picos.

**Objetivo:** Detectar picos espectrais em sinais Ambisonics (W,X,Y,Z...) e aplicar envelope de sustain/decay mantendo a espacialização intacta.

---

## Histórico de Problemas e Soluções

### 1. CRASH: Segmentation Fault (PC=0x40)

**Data:** 19 de Novembro de 2025

#### Sintomas
- PureData/PlugData crashava ao instanciar objeto
- Erro: `EXC_BAD_ACCESS` em endereço `0x0000000000000040`
- Crash no thread principal durante DSP setup

#### Causa Raiz
Bug crítico na função `dsp()`: estávamos incluindo o **ponteiro da função perform** no vetor de argumentos `dsp_vec`:

```cpp
// ❌ ERRADO
std::vector<t_int> dsp_vec;
dsp_vec.push_back(reinterpret_cast<t_int>(torch_amb_spectrails_tilde_perform)); // BUG!
dsp_vec.push_back(reinterpret_cast<t_int>(x));
dsp_vec.push_back(static_cast<t_int>(sp[0]->s_n));
```

**Problema:** Pure Data adiciona automaticamente a função em `w[0]`. Ao incluir manualmente, todos os índices ficavam deslocados:
- `w[1]` (deveria ser `x`) → recebia ponteiro da função
- `w[2]` (deveria ser `n`) → recebia `x`
- Resultado: `x` apontava para código executável, causando acesso inválido

#### Solução
Remover ponteiro da função do vetor:

```cpp
// ✅ CORRETO
std::vector<t_int> dsp_vec;
dsp_vec.push_back(reinterpret_cast<t_int>(x));           // w[1]
dsp_vec.push_back(static_cast<t_int>(sp[0]->s_n));      // w[2]
// PureData adiciona função em w[0] automaticamente
```

**Função perform ajustada:**
```cpp
static t_int *torch_amb_spectrails_tilde_perform(t_int *w) {
    // w[0] = função (adicionada pelo PD)
    auto *x = reinterpret_cast<t_torch_amb_spectrails_tilde *>(w[1]);
    int n = static_cast<int>(w[2]);
    // w[3+] = signal pointers
}
```

**Status:** ✅ RESOLVIDO

---

### 2. CRASH: Acesso à Memória Inválida durante Processamento

**Data:** 21 de Novembro de 2025

#### Sintomas
- Objeto funciona inicialmente
- Crash aleatório durante processamento de áudio
- Erro: `KERN_INVALID_ADDRESS at 0xa934ffbfb81643ad`
- Thread: `com.apple.audio.IOThread.client`
- Função: `torch_amb_spectrails_tilde_perform`

#### Causas Potenciais Identificadas

**A. Processamento Duplo do Canal W**
```cpp
// ❌ PROBLEMA
auto w_outputs = x->processors_[0]->process_frame(w_mag, w_phase); // 1ª vez

for (int ch = 0; ch < x->num_channels_; ++ch) {
    auto outputs = x->processors_[ch]->process_frame(...); // W processado 2ª vez!
}
```

**Risco:** Processador pode ter estado interno (smoothing, histórico) que é corrompido ao processar duas vezes.

**B. Race Condition com Tensors**
```cpp
auto w_envelope = x->processors_[0]->get_envelope_positions();
// Retorna tensor interno que pode ser modificado durante set_envelope_positions()
```

**Risco:** Se `get_envelope_positions()` retorna referência (não cópia), modificar em outro processador pode corromper dados.

**C. Clone/Move de Tensors**
```cpp
auto mag_tensor = torch::from_blob(...).clone();
// Se from_blob cria view e clone falha, pode acessar memória já liberada
```

#### Solução Implementada

**1. Processar W apenas uma vez e reusar resultado:**
```cpp
// Processa W
auto w_outputs = x->processors_[0]->process_frame(w_mag_tensor, w_phase_tensor);

// Copia envelope (assume que retorna cópia, não referência)
auto w_envelope = x->processors_[0]->get_envelope_positions();

// Sincroniza outros canais
for (int ch = 1; ch < x->num_channels_; ++ch) {
    x->processors_[ch]->set_envelope_positions(w_envelope);
}

// Escreve saída de W SEM reprocessar
std::memcpy(out_mags[0], w_outputs[0].data_ptr<float>(), num_bins * sizeof(float));
std::memcpy(out_phases[0], w_outputs[1].data_ptr<float>(), num_bins * sizeof(float));

// Processa X,Y,Z (ch=1 em diante)
for (int ch = 1; ch < x->num_channels_; ++ch) {
    auto outputs = x->processors_[ch]->process_frame(...);
}
```

**2. Usar memcpy ao invés de accessor em loop:**
```cpp
// ❌ Lento e potencial risco
for (size_t i = 0; i < num_bins; ++i) {
    out_mags[ch][i] = mag_acc[i];
}

// ✅ Rápido e seguro
std::memcpy(out_mags[ch], processed_mag.data_ptr<float>(), num_bins * sizeof(float));
```

**3. Garantir contiguidade antes de data_ptr:**
```cpp
auto processed_mag = outputs[0].contiguous().to(torch::kCPU);
auto processed_phase = outputs[1].contiguous().to(torch::kCPU);
std::memcpy(out_mags[ch], processed_mag.data_ptr<float>(), ...);
```

**Status:** ⚠️ EM TESTE (implementado mas precisa validação)

---

### 3. PROBLEMA: Espacialização Perdida no Decoding

**Data:** 21 de Novembro de 2025

#### Sintomas
- Objeto funciona sem crash
- Picos detectados corretamente em W
- Após decoding Ambisonics: **localização sonora incorreta/difusa**
- Som perde a direcionalidade original

#### Causa Raiz

Cada canal (W,X,Y,Z) decidia **independentemente** quando escrever um pico:

```cpp
// No processador (core_ap_spectrails.h linha 149):
bool significant_increase = curr_mag > (out_mag_acc[i] * 1.1f);

if (significant_increase) {
    out_mag_acc[i] = curr_mag;
    out_phase_acc[i] = curr_phase;
    env_acc[i] = 1.0f;
}
```

**Problema:**
- Bin 150: W detecta pico (mag=0.8) → **escreve**
- Bin 150: X tem mag=0.3 (< threshold) → **NÃO escreve**
- Bin 150: Y tem mag=0.15 → **NÃO escreve**
- **Resultado:** W tem energia mas X,Y,Z não → espacialização quebrada

**Por que isso importa em Ambisonics:**

A localização é definida pela **relação entre canais**:
- Vetor de intensidade = (X/W, Y/W, Z/W)
- Se W=0.8 mas X=0 (deveria ser 0.3), o som aparece no lugar errado

**Analogia:** É como ter o volume da caixa esquerda sem o da direita - você perde a imagem estéreo.

#### Solução: Sample & Hold Sincronizado

**Conceito:** Quando W detecta pico, **TODOS** os canais devem escrever seus valores naquele bin, independente de magnitude.

**Implementação:**

1. **Novo método no core (`core_ap_spectrails.h`):**
```cpp
void force_write_peak(int bin_idx, float magnitude, float phase) {
    if (bin_idx < 0 || bin_idx >= num_bins_) return;
    
    auto out_mag_acc = out_mag_.accessor<float, 1>();
    auto out_phase_acc = out_phase_.accessor<float, 1>();
    auto env_acc = envelope_positions_.accessor<float, 1>();
    
    out_mag_acc[bin_idx] = magnitude;
    out_phase_acc[bin_idx] = phase;
    env_acc[bin_idx] = 1.0f;  // Reset envelope
}
```

2. **Lógica no PureData external:**
```cpp
// 1. W processa e detecta
auto w_outputs = x->processors_[0]->process_frame(w_mag_tensor, w_phase_tensor);
auto w_peaks = x->processors_[0]->get_detected_peaks();

// 2. Sincroniza envelope (decay compartilhado)
auto w_envelope = x->processors_[0]->get_envelope_positions();
for (int ch = 1; ch < x->num_channels_; ++ch) {
    x->processors_[ch]->set_envelope_positions(w_envelope);
}

// 3. FORÇA escrita em X,Y,Z quando W detecta pico
auto w_peaks_acc = w_peaks.accessor<float, 1>();
for (size_t bin = 0; bin < num_bins; ++bin) {
    if (w_peaks_acc[bin] > 0.0f) {  // W detectou pico neste bin
        for (int ch = 1; ch < x->num_channels_; ++ch) {
            float mag = in_mags[ch][bin];    // Valor EXATO daquele momento
            float phase = in_phases[ch][bin]; // Fase EXATA daquele momento
            x->processors_[ch]->force_write_peak(bin, mag, phase);
        }
    }
}

// 4. Processa normalmente (vai usar valores forçados ou atuais)
for (int ch = 1; ch < x->num_channels_; ++ch) {
    auto outputs = x->processors_[ch]->process_frame(...);
}
```

**Resultado Esperado:**
- Bin 150: W detecta pico → W=0.8, X=0.3, Y=0.15, Z=0.1 (TODOS escrevem)
- Relação X/W = 0.3/0.8 = 0.375 (preservada!)
- Decay sincronizado reduz todos proporcionalmente
- Espacialização mantida no decoding

**Status:** 🔧 IMPLEMENTADO, AGUARDANDO COMPILAÇÃO

---

## Estado Atual do Código

### Arquivos Modificados

1. **`core/include/core_ap_spectrails.h`**
   - ✅ Adicionado: `force_write_peak(int bin_idx, float magnitude, float phase)`
   - ✅ Já existe: `get_detected_peaks()` (retorna tensor com flags de picos)

2. **`wrappers/puredata/src/torch.amb.spectrails~.cpp`**
   - ✅ Corrigido: Remoção de ponteiro de função do `dsp_vec`
   - ✅ Corrigido: W processado apenas uma vez
   - ✅ Otimizado: Uso de `memcpy` ao invés de `accessor`
   - ✅ Implementado: Lógica de `force_write_peak` sincronizada
   - ✅ Removido: Logs de debug (console limpo)

### Pendente

- [ ] Compilar e testar nova versão
- [ ] Validar espacialização com decoding
- [ ] Testar com diferentes ordens ambisonics (1ª, 2ª, 3ª)
- [ ] Monitorar crashes em uso prolongado
- [ ] Ajustar parâmetros (threshold, attack, decay) se necessário

---

## Lições Aprendidas

### Pure Data DSP Chain
- **NUNCA** incluir ponteiro de função em `dsp_vec` - PD adiciona automaticamente em `w[0]`
- Retorno de `perform()` deve ser `w + (número de args + 1)`
- Validar todos os ponteiros antes de usar no perform (thread de áudio)

### LibTorch em Real-Time Audio
- Evitar alocações no `perform()` (usar `reserve()` antes)
- `.contiguous()` é necessário antes de `data_ptr<>()`
- `memcpy` é mais seguro e rápido que accessor em loops
- Cuidado com views vs. cópias de tensors

### Ambisonics Spectral Processing
- Detecção de picos apenas em W (omnidirecional)
- **TODOS** os canais devem escrever simultaneamente quando W detecta
- Preservar **magnitude E fase** exatas do momento do pico
- Decay/sustain sincronizado mantém proporções relativas (X/W, Y/W, Z/W)

### Debugging
- PlugData auto-habilita DSP → dificulta debug de crashes
- Pure Data com console é melhor para capturar logs
- Crash em `PC=0x40` geralmente indica ponteiros corrompidos no DSP chain
- `KERN_INVALID_ADDRESS` em thread de áudio → problema no `perform()`

---

## Referências Técnicas

**Ambisonics:**
- Ordem N → Canais B = (N+1)²
- 1ª ordem: 4 canais (W,X,Y,Z)
- 2ª ordem: 9 canais (W,X,Y,Z,R,S,T,U,V)
- 3ª ordem: 16 canais

**Vetor de Intensidade:**
```
I = (X/W, Y/W, Z/W)
Azimuth = atan2(Y, X)
Elevation = atan2(Z, sqrt(X² + Y²))
```

**Sample & Hold Complexo:**
```
Bin k no momento do pico:
W_k = M_w * e^(jφ_w)
X_k = M_x * e^(jφ_x)
Y_k = M_y * e^(jφ_y)
Z_k = M_z * e^(jφ_z)

Sustain deve manter TODAS essas 4 tuplas (M,φ) sincronizadas.
```

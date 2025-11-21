# Guia: Modo Prominence - torch.amb.spectrails~

## Visão Geral

O modo **prominence** implementa detecção de picos espectrais baseada em **saliência relativa**, inspirado no algoritmo do `sigmund~` de Miller Puckette. É especialmente eficaz para detectar transientes (como cantos de pássaros) em ambientes ruidosos.

## Diferença entre Modos

### Modo 0: SLOPE (padrão)
- Detecta picos por **slope negativo** após máximo local
- Rastreia magnitude ao longo do tempo
- Bom para sons sustentados e harmônicos estáveis
- Mais conservador - evita retriggering

### Modo 1: PROMINENCE
- Detecta picos por **saliência relativa** aos bins vizinhos
- Compara magnitude com bins adjacentes (±2 bins)
- Usa **parabolic interpolation** para refinamento de frequência
- Melhor para transientes e eventos rápidos
- Mais sensível a mudanças espectrais

## Parâmetros

### Criação do Objeto

```pd
[torch.amb.spectrails~ -order 1 -mode 1 -prominence 0.6 -threshdb -60]
```

**Flags de criação:**
- `-mode <0|1>`: 0=slope (padrão), 1=prominence
- `-prominence <0-1>`: threshold de saliência relativa (padrão: 0.6)
- `-threshdb <dB>`: threshold absoluto em dB (padrão: -60)
- `-pregaindb <dB>`: ganho prévio para elevar sinal fraco (padrão: 0)
- `-order <N>`: ordem ambisônica (padrão: 1)
- `-device <cpu|cuda|mps>`: dispositivo de processamento

### Mensagens Runtime

```pd
[mode 1(          # Alterna para modo prominence
[mode 0(          # Volta para modo slope
[prominence 0.8(  # Ajusta threshold de saliência (0-1)
[threshdb -50(    # Threshold em dB
[pregaindb 20(    # Pregain em dB
[attack 0.8(      # Taxa de attack (0-1)
[decay 0.995(     # Fator de decay (0-1)
[reset(           # Limpa memória
```

## Como Funciona (Modo Prominence)

### 1. Detecção de Pico Local
```
is_peak = (mag[i] > mag[i-1]) && (mag[i] > mag[i+1]) && (mag[i] > threshold)
```

### 2. Teste de Saliência Relativa
```
neighbor_power = mag[i-2] + mag[i+2]
prominent = mag[i] > prominence_threshold * neighbor_power
```

**Interpretação:**
- `prominence=0.6` → pico deve ser 60% maior que vizinhos
- Menor valor = mais sensível (detecta picos menores)
- Maior valor = mais seletivo (apenas picos muito proeminentes)

### 3. Parabolic Interpolation
```
detune = (right² - left²) / (2 * (2*center - left - right))
refined_freq = (bin + detune) * Hz_per_bin
```

Refina posição do pico entre bins (sub-bin accuracy).

### 4. Masking Espacial
- Evita detectar o mesmo pico múltiplas vezes
- Respeita `min_peak_distance` em Hz

## Casos de Uso

### Detecção de Bird Calls
```pd
# Sinais fracos, transientes rápidos em ruído
[torch.amb.spectrails~ -mode 1 -prominence 0.5 -threshdb -70 -pregaindb 20]
```

**Por quê?**
- `prominence=0.5`: mais sensível a picos pequenos
- `threshdb=-70`: threshold muito baixo (detecta sinais fracos)
- `pregaindb=20`: eleva sinal antes da detecção

### Sons Harmônicos Sustentados
```pd
# Vozes, instrumentos, harmônicos estáveis
[torch.amb.spectrails~ -mode 0 -threshdb -50 -attack 0.7 -decay 0.999]
```

**Por quê?**
- `mode=0` (slope): melhor para rastreamento contínuo
- Decay alto mantém harmônicos sustentados

### Percussão e Ataques
```pd
# Transientes claros, ataques rápidos
[torch.amb.spectrails~ -mode 1 -prominence 0.7 -attack 0.9 -decay 0.99]
```

**Por quê?**
- `mode=1`: captura transientes instantaneamente
- `attack=0.9`: envelope rápido
- `decay=0.99`: decai rapidamente após ataque

## Comparação: Slope vs Prominence

| Característica | SLOPE (0) | PROMINENCE (1) |
|---------------|-----------|----------------|
| Detecção | Negative slope após peak | Saliência relativa |
| Sensibilidade | Moderada | Alta (ajustável) |
| Transientes | Bom | Excelente |
| Harmônicos | Excelente | Bom |
| Ruído | Mais imune | Mais sensível |
| Retriggering | Baixo | Moderado |
| CPU | Menor | Levemente maior |

## Ajuste Fino

### Muito sensível (muitos falsos positivos)?
1. Aumente `prominence` (0.6 → 0.8)
2. Aumente `threshdb` (-70 → -60)
3. Reduza `pregaindb` (20 → 10)
4. Aumente `min_peak_distance`

### Muito insensível (perde transientes)?
1. Reduza `prominence` (0.6 → 0.4)
2. Reduza `threshdb` (-60 → -80)
3. Aumente `pregaindb` (0 → 20)
4. Reduza `decay` para limpar rapidamente

### Picos "grudam" muito tempo?
1. Reduza `decay` (0.999 → 0.995)
2. Use `decaytime` em segundos: `[decaytime 0.5(`

### Ataques lentos?
1. Aumente `attack` (0.7 → 0.9)

## Patches de Teste

### 1. `test_prominence.pd`
- Teste básico com tom puro
- Controles interativos
- Comparação visual

### 2. `test_bird_detection.pd`
- Simulação de chirps + ruído
- Lado-a-lado slope vs prominence
- Métricas de detecção

## Algoritmo (Referência: sigmund~)

Baseado em Miller Puckette's `sigmund~.c`:

```c
// Threshold relativo (bins ±2)
if (peak > PROMINENCE_THRESHOLD * (neighbor[i-2] + neighbor[i+2]))
    detect_peak();

// Parabolic interpolation para sub-bin accuracy
detune = (right_mag² - left_mag²) / (2 * windpower);
freq_refined = (bin + 2*detune) * Hz_per_bin;
```

**Constantes sigmund~:**
- `PEAKTHRESHFACTOR = 0.6` (nosso `prominence_threshold`)
- Bins comparados: ±2 posições

## Troubleshooting

### Crash ao usar `-mode 1`?
- Verifique se recompilou após adicionar o código
- Rode `[mode 1(` via mensagem em vez de flag

### Prominence não faz diferença?
- Teste com sinal ruidoso (não tom puro)
- Adicione transientes: `[noise~] → [bp~ 3000]`
- Ajuste `prominence` para valores extremos (0.1 ou 0.9)

### Performance ruim?
- Modo prominence tem custo levemente maior
- Reduza FFT size: `-n 1024` → `-n 512`
- Use CPU em vez de GPU para buffers pequenos

## Referências

- Miller Puckette - `sigmund~.c` (spectral peak tracking)
- Parabolic interpolation for frequency estimation
- Spectral leakage mitigation techniques

---

**Autor:** Implementado a partir de análise do `sigmund~`  
**Data:** 2025-11-21  
**Versão:** contorchionist_secondnature v1.0+prominence

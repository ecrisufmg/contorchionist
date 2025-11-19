# IRFFT Output Windowing: Power Compensation Problem

## Context

A introdução do parâmetro `-wo` (window output) no `torch.irfft~` permite aplicar janelamento ao sinal de tempo reconstruído pela IRFFT. Embora útil para suavizar transições, isso cria um problema de conservação de energia que precisa ser compensado.

## Observações Experimentais

### Caso 1: Sem Window Output (`-wo` desabilitado)
- ✅ **Potência preservada**: Sinal de saída mantém mesma potência que entrada
- ✅ **Compensação de overlap funciona**: Usar `-of 4` (overlap factor 4) compensa corretamente a sobreposição
- ✅ **Independente da janela**: Funciona com qualquer tipo de janela (Hann, Hamming, Blackman, etc.)
- ✅ **Independente do tamanho**: Funciona com qualquer tamanho de janela/FFT

### Caso 2: Com Window Output (`-wo` habilitado)
- ❌ **Perda de potência**: Aproximadamente **-2.5 dB** de atenuação observada
- ❌ **Compensação de overlap insuficiente**: O `-of` não compensa a atenuação da janela de saída
- ⚠️ **Atenuação depende da janela**: Janelas diferentes (Hann vs Rectangular) causam perdas diferentes

## Análise do Problema

### Por que a potência é perdida?

Quando aplicamos uma janela `w[n]` ao sinal de tempo após a IRFFT:

```
y_out[n] = y_irfft[n] * w[n]
```

A energia do sinal é multiplicada pelo quadrado dos valores da janela:

```
E_out = Σ (y_irfft[n] * w[n])²
      = Σ y_irfft[n]² * w[n]²
```

Para janelas comuns:
- **Rectangular**: w[n] = 1 → sem atenuação (0 dB)
- **Hann**: Σw²[n]/N ≈ 0.375 → atenuação de ~-4.26 dB
- **Hamming**: Σw²[n]/N ≈ 0.397 → atenuação de ~-4.01 dB
- **Blackman**: Σw²[n]/N ≈ 0.252 → atenuação de ~-5.99 dB

### Por que -2.5 dB especificamente?

Se você está observando -2.5 dB, provavelmente está usando:
1. **Janela Hann** (~-4.26 dB) + **overlap 4x** com soma ponderada
2. Ou uma **combinação parcial** de efeitos de overlap-add que cancela parte da atenuação

## Soluções Possíveis

### Opção 1: Normalização por Soma da Janela ao Quadrado
**Ideia**: Dividir pela raiz quadrada da soma dos quadrados da janela.

```cpp
// Em process_irfft(), após aplicar janela:
if (windowing_enabled_ && window_prepared_) {
    time_signal = time_signal * window_;
    
    // Compensação de potência
    float window_power_factor = std::sqrt(current_sum_sq_window_ / current_window_n_);
    time_signal = time_signal / window_power_factor;
}
```

**Prós**:
- ✅ Conserva energia do sinal
- ✅ Simples de implementar
- ✅ Já temos `current_sum_sq_window_` calculado

**Contras**:
- ⚠️ Pode não ser correto para overlap-add (depende de como as janelas se somam)

### Opção 2: Normalização COLA (Constant Overlap-Add)
**Ideia**: Para overlap-add, a soma das janelas deslocadas deve ser constante.

Para overlap de 50% (factor 2) com Hann:
```
w[n] + w[n + N/2] = 1  (constante)
```

Neste caso, a compensação já está embutida na propriedade COLA.

**Implementação**:
```cpp
if (windowing_enabled_ && window_prepared_) {
    time_signal = time_signal * window_;
    
    // Para janelas COLA com overlap correto, dividir pelo overlap factor
    if (overlap_factor_ > 1.0f) {
        // A soma das janelas já é normalizada pelo COLA
        // Não precisa compensação extra
    } else {
        // Sem overlap, precisa compensar pela potência da janela
        float window_power_factor = std::sqrt(current_sum_sq_window_ / current_window_n_);
        time_signal = time_signal / window_power_factor;
    }
}
```

**Prós**:
- ✅ Teoricamente correto para overlap-add
- ✅ Diferencia casos com/sem overlap

**Contras**:
- ⚠️ Assume que janela tem propriedade COLA
- ⚠️ Não funciona para todas as combinações janela/overlap

### Opção 3: Fator de Compensação Configurável
**Ideia**: Deixar usuário controlar o fator de compensação.

```cpp
// Adicionar membro:
T window_output_gain_;  // Ganho para compensar janelamento de saída

// Método de configuração:
void set_window_output_gain(T gain) {
    window_output_gain_ = gain;
}

// Em process_irfft():
if (windowing_enabled_ && window_prepared_) {
    time_signal = time_signal * window_ * window_output_gain_;
}
```

**Interface Pd**:
```
[winoutgain 2.5(  → Compensa com +2.5 dB (~1.33x)
```

**Prós**:
- ✅ Flexibilidade máxima
- ✅ Usuário pode calibrar empiricamente
- ✅ Funciona para qualquer situação

**Contras**:
- ❌ Requer ajuste manual
- ❌ Usuário precisa entender o conceito

### Opção 4: Auto-calibração por Tipo de Janela
**Ideia**: Tabela lookup com fatores de compensação pré-calculados.

```cpp
T get_window_power_compensation(Type window_type, int overlap_factor) {
    if (overlap_factor == 1) {
        // Sem overlap: compensar pela RMS da janela
        switch (window_type) {
            case Type::HANN: return 1.0f / std::sqrt(0.375f);
            case Type::HAMMING: return 1.0f / std::sqrt(0.397f);
            case Type::BLACKMAN: return 1.0f / std::sqrt(0.252f);
            case Type::RECTANGULAR: return 1.0f;
            // etc...
        }
    } else if (overlap_factor == 2) {
        // Overlap 2x: janelas COLA já compensam
        return 1.0f;
    } else if (overlap_factor == 4) {
        // Overlap 4x: depende da janela...
        // Requer análise caso a caso
    }
    return 1.0f;
}
```

**Prós**:
- ✅ Automático
- ✅ Preciso para casos conhecidos

**Contras**:
- ❌ Complexo de implementar
- ❌ Não cobre todos os casos

## Recomendação

### Implementação Híbrida (Melhor opção)

1. **Cálculo automático** baseado em `current_sum_sq_window_` quando `overlap_factor_ == 1`
2. **Sem compensação** quando `overlap_factor_ >= 2` e janela tem propriedade COLA
3. **Override manual** via mensagem `winoutgain` para casos especiais

```cpp
// Em process_irfft(), após aplicar janela:
if (windowing_enabled_ && window_prepared_) {
    time_signal = time_signal * window_;
    
    T compensation_factor = 1.0f;
    
    if (window_output_gain_override_ > 0.0f) {
        // Usuário definiu ganho manual
        compensation_factor = window_output_gain_override_;
    } else if (overlap_factor_ <= 1.0f) {
        // Sem overlap: compensar pela RMS da janela
        T window_rms = std::sqrt(current_sum_sq_window_ / current_window_n_);
        compensation_factor = 1.0f / window_rms;
    }
    // else: overlap >= 2, assume COLA, sem compensação
    
    if (compensation_factor != 1.0f) {
        time_signal = time_signal * compensation_factor;
    }
}
```

### Interface Pd proposta:

```pd
# Uso normal com overlap 2x (COLA automático):
[torch.irfft~ -wo -of 2 -w hann]

# Sem overlap (compensação automática):
[torch.irfft~ -wo -w hann]

# Override manual para casos especiais:
[torch.irfft~ -wo -of 4 -w blackman]
|
[winoutgain 1.5(  # Ajuste fino empírico
```

## Questões para Investigação

1. **Qual janela você está usando?** (Hann, Hamming, Blackman?)
2. **Qual overlap factor?** (-of 4 significa overlap de 75%)
3. **A perda de -2.5 dB é exata ou aproximada?**
4. **Você precisa de compensação automática ou manual?**

## Próximos Passos

- [ ] Implementar cálculo de `current_sum_sq_window_` no `core_ap_rfft.h`
- [ ] Adicionar lógica de compensação em `process_irfft()`
- [ ] Adicionar método `set_window_output_gain()` para override manual
- [ ] Criar mensagem `winoutgain` no wrapper Pd
- [ ] Testar com diferentes combinações janela/overlap
- [ ] Documentar fatores de compensação para janelas comuns

# Guia de Controles do `torch.spectrails~`

Este guia detalha o comportamento atual dos controles de ataque, onset e reset do `SpectralTrailsProcessor` expostos pelo wrapper `torch.spectrails~`. Todas as rotinas estão implementadas em `core/include/core_ap_spectrails.h` (função `process_frame`) e no wrapper Pure Data `wrappers/puredata/src/torch.spectrails~.cpp`.

## Fluxo Geral
1. **Decay universal:** antes de qualquer reforço, o conteúdo de memória é multiplicado por `decay` (ou pelo fator derivado de `decay6db`).
2. **Threshold:** bins com entrada `> threshold` tornam-se candidatos a reforço. Bins abaixo do threshold apenas sofrem decay e alimentam o contador de reset.
3. **Aplicação de ataque:** o coeficiente de ataque final leva em conta `attack`, `attackadapt` e, quando habilitado, `attackonset` + `onsetfloor`.
4. **Reset parcial:** se `resetframes` for maior que zero e o contador por bin exceder esse valor, a memória sofre a escala definida por `resetmult` e a fase é zerada.
5. **Limiter:** depois de atualizada, a magnitude passa pelo limitador (`limiter`, `limitersoft`, `maxvalue`).

## Controles de Ataque
- **`attack`** (`attack <0..1>`): coeficiente base usado na interpolação linear `memory = attack * input + (1 - attack) * memory`. Define a rapidez da resposta global.
- **`attackadapt`** (`attackadapt <rate>` / criação `@attackadapt`): aplica um ganho dependente da diferença `|input - memory|`, usando `adaptive_factor = 1 - exp(-|Δ| * rate)`. O ataque final vira `attack + (1 - attack) * adaptive_factor`. Use valores maiores quando quiser que grandes variações subam rapidamente sem perder a suavidade em pequenas variações.
- **`attackonset`** (`attackonset <0..1>`; valores negativos desativam): substitui o ataque final por um coeficiente fixo quando `memory <= onsetfloor` e `input > threshold`, identificando onsets “novos”.
- **`onsetfloor`** (`onsetfloor <f>`): piso de magnitude que define quando um bin é considerado “zerado”. Ajuste para separar ruído residual de silêncio real. Valores muito altos podem tratar bins fracos como novos com excessiva frequência.

## Reset Parcial
- **`resetframes`** (`resetframes <int>`): conta quantos frames consecutivos um bin permaneceu com `input <= threshold`. Com zero, a funcionalidade fica desligada.
- **`resetmult`** (`resetmult <0..1>`): multiplicador aplicado à memória quando `resetframes` é atingido. `0.0` limpa completamente; valores intermediários aceleram o decaimento; `1.0` mantém a magnitude intacta (apenas zera a fase).

## Relação com Threshold, Decay e Limiter
- `threshold` define as regiões de reforço versus pura atenuação. Ele dirige tanto o ataque (pois determina os bins elegíveis) quanto o reset (contagem só ocorre enquanto a entrada ficar abaixo dele).
- `decay`/`decay6db` atuam sempre antes do ataque. Um decay lento exige ataques mais altos ou resets para evitar resquícios; um decay muito rápido exige vigiar `attackonset` para que o sustain não desmorone.
- O `limiter` é aplicado após todas as atualizações de memória. Assim, mesmo com `attackonset` próximo de 1.0, o valor final fica contido. `limitersoft` controla o joelho suave para preservar continuidade.

## Sugestões de Uso
1. **Configure o comportamento base** com `threshold` e `decay`/`decay6db` para o tipo de material (percussivo, harmônico, ruído).
2. **Ajuste `attack` e `attackadapt`** para responder aos transientes que precisam de mais punch sem sacrificar o sustain.
3. **Ative `attackonset`/`onsetfloor`** quando notar cliques ou atraso na entrada de novos bins. Um valor típico inicial é `attackonset 0.85` com `onsetfloor 1e-3`.
4. **Use `resetframes`/`resetmult`** para evitar que bins “grudem” em sessões com períodos de silêncio. Experimente `resetframes 8` e `resetmult 0.4` como ponto de partida.
5. **Finalize com o limiter**, ajustando `limitersoft` e `maxvalue` apenas após equilibrar os demais parâmetros.

## Mensagens Pure Data
Todas as configurações podem ser feitas por argumentos na criação (`@attack 0.75`, `@attackonset 0.9`, etc.) ou via mensagens:
```
threshold 0.02
attack 0.6
attackadapt 0.4
attackonset 0.9
onsetfloor 0.0005
resetframes 12
resetmult 0.3
decay6db 1.5
limiter 1
limitersoft 0.4
maxvalue 1.2
```
Verifique o console do Pd: cada mensagem gera um `post` confirmando o valor aplicado.

# Spectral Trails Roadmap

Este documento organiza a evolução planejada do `SpectralTrailsProcessor` e do wrapper `torch.spectrails~`, com foco em tornar o objeto robusto para sinais variáveis e minimizar artefatos de ataque. As entregas estão ordenadas por prioridade.

## Prioridade 1 — Decaimento configurável por segundos (`decay6db`) OK
- **Objetivo**: permitir definir o tempo, em segundos, para que cada bin decaia -6 dB (redução para 50% do valor).
- **Abordagem**:
  - Manter o parâmetro atual `decay` (fator por frame) para compatibilidade.
  - Introduzir `decay6db`; ao recebê-lo, calcular `decay_factor = pow(0.5, 1.0 / (time_s * frames_per_second))`, onde `frames_per_second = sample_rate / hop_size`.
  - Atualizar o wrapper PD (`@decay6db`, mensagem `decay6db <f>`) e validar a coexistência com o modo antigo.

## Prioridade 2 — Ataque independente por bin (andamento)
- **Objetivo**: reduzir saltos na ativação de novos bins sem depender de `maxvalue` extremamente baixo.
- **Progresso**:
  - `attackadapt` já controla a aproximação do ataque em função do delta.
  - Novo parâmetro `@attackonset` (desativável com valores negativos) aplica ataque dedicado quando a memória está abaixo do piso definido e o bin cruza o threshold.
  - `@onsetfloor` define o piso que caracteriza “novo bin” (base para o gatilho de `@attackonset`).
  - `@resetframes` conta quantos frames consecutivos um bin passou abaixo do `threshold`; ao atingir o valor configurado, o reset parcial é disparado.
  - `@resetmult` define o multiplicador aplicado quando o reset parcial acontece (0 limpa completamente, 1 não altera o bin, valores intermediários aceleram o decaimento).
  - Rampas de `@attackonsetramp` agora usam smoothstep cúbico (in/out suave), blend progressivo da memória e rampagem conjunta de fase, mantendo o primeiro frame intacto e convergindo ao alvo ao final para eliminar degraus sincronizados ao hop.
- **Próximos passos**:
  - Refinar curvas adaptativas para graves/agudos se ainda houver ruído perceptível.
  - Medir audições A/B com e sem `attackonset` para ajustar valores padrão.

## Prioridade 3 — Limiter suave OK
- **Objetivo**: controlar picos sem “achatamento” abrupto da memória.
- **Abordagem**:
  - Substituir o clamp rígido por uma função suave (tanh, soft knee ou normalização relativa ao pico recente).
  - Permitir ajuste de intensidade/compressão.
  - Implementado: parâmetro `@limitersoft` (alias `@limitersmooth`) aplica joelho racional `over / (1 + k·over)` mantendo continuidade; testar valores típicos `0.1`–`2.0` e registrar comportamento extremo.
  - Garantir que a energia média permaneça próxima dos valores originais.

## Prioridade 4 — Reset parcial inteligente (protótipo)
- **Objetivo**: liberar memória mais rápido quando bins permanecem abaixo do threshold, evitando reativações altas.
- **Progresso**:
  - `@resetframes` controla o número de frames consecutivos abaixo do threshold antes de aplicar reset.
  - `@resetmult` define o multiplicador aplicado ao bin quando o reset dispara (0.0 zera completamente, valores próximos de 1.0 apenas aceleram o decaimento).
- **Próximos passos**:
  - Testar em materiais com longos silêncios/ruído de fundo para validar o comportamento.
  - Avaliar se o reset também deve zerar fase ou usar uma interpolação mais suave.

## Prioridade 5 — Smoothing dependente do hop (hop-aware)
- **Objetivo**: alinhar o smoothing de magnitude/fase com a sobreposição usada em `torch.rfft~` / `torch.irfft~`.
- **Abordagem**:
  - Calcular coeficientes de ataque/decay em função do overlap e do hop size.
  - Validar que diferentes configurações de overlap produzem caudas coerentes.

## Próximos passos gerais
- Atualizar testes e documentação após cada etapa.
- Registrar medições (picos, energia média) antes/depois para confirmar os ganhos.
- Preparar exemplos em Pd/PlugData mostrando as melhorias.

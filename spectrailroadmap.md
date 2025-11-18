# Spectral Trails Roadmap

Este documento organiza a evolução planejada do `SpectralTrailsProcessor` e do wrapper `torch.spectrails~`, com foco em tornar o objeto robusto para sinais variáveis e minimizar artefatos de ataque. As entregas estão ordenadas por prioridade.

## Prioridade 1 — Decaimento configurável por segundos (`decay6db`)
- **Objetivo**: permitir definir o tempo, em segundos, para que cada bin decaia -6 dB (redução para 50% do valor).
- **Abordagem**:
  - Manter o parâmetro atual `decay` (fator por frame) para compatibilidade.
  - Introduzir `decay6db`; ao recebê-lo, calcular `decay_factor = pow(0.5, 1.0 / (time_s * frames_per_second))`, onde `frames_per_second = sample_rate / hop_size`.
  - Atualizar o wrapper PD (`@decay6db`, mensagem `decay6db <f>`) e validar a coexistência com o modo antigo.

## Prioridade 2 — Ataque independente por bin
- **Objetivo**: reduzir saltos na ativação de novos bins sem depender de `maxvalue` extremamente baixo.
- **Abordagem**:
  - Manter tempos de ataque separados para magnitude e fase.
  - Ajustar o ataque com base na diferença entre entrada e memória (p.ex., interpolação exponencial dependente da magnitude do erro).
  - Expor novos parâmetros conforme necessário e medir a redução de artefatos auditivos.

## Prioridade 3 — Limiter suave
- **Objetivo**: controlar picos sem “achatamento” abrupto da memória.
- **Abordagem**:
  - Substituir o clamp rígido por uma função suave (tanh, soft knee ou normalização relativa ao pico recente).
  - Permitir ajuste de intensidade/compressão.
  - Garantir que a energia média permaneça próxima dos valores originais.

## Prioridade 4 — Reset parcial inteligente
- **Objetivo**: liberar memória mais rápido quando bins permanecem abaixo do threshold, evitando reativações altas.
- **Abordagem**:
  - Monitorar quantos frames consecutivos cada bin fica abaixo do threshold.
  - Após um número configurável de frames, aplicar decaimento acelerado ou reset parcial daquele bin.
  - Testar com sinais impulsivos e sons com sustain longo.

## Prioridade 5 — Smoothing dependente do hop (hop-aware)
- **Objetivo**: alinhar o smoothing de magnitude/fase com a sobreposição usada em `torch.rfft~` / `torch.irfft~`.
- **Abordagem**:
  - Calcular coeficientes de ataque/decay em função do overlap e do hop size.
  - Validar que diferentes configurações de overlap produzem caudas coerentes.

## Próximos passos gerais
- Atualizar testes e documentação após cada etapa.
- Registrar medições (picos, energia média) antes/depois para confirmar os ganhos.
- Preparar exemplos em Pd/PlugData mostrando as melhorias.

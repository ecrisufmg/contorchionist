# Spectral Trails Roadmap — Crossfade Simplification

Objetivo: Resolver os impulsos sincronizados ao hop durante novos onsets no `SpectralTrailsProcessor`, mantendo somente os controles essenciais enquanto depuramos a arquitetura.

## Etapa 0 — Escopo mínimo
- Controles ativos: `threshold`, `decay` / `decay6db`, `frame_timing` (hopsize + sample rate).
- Temporariamente desabilitar (ou fixar em valores neutros): `attackadapt`, `attackonset`, `attackonsetramp`, `resetframes`, `resetmult`, `onsetfloor`, `limiter`, `phase smoothing extra`.
- Confirmar que o build atual reproduz o problema original (pulsos alinhados ao hop) para ter baseline.

## Etapa 1 — Pré-preenchimento de memória (`pending` buffers)
- Detectar transição `below threshold -> above threshold` para cada bin.
- Ao detectar onset, copiar magnitude e fase atuais para buffers `pending_mag`, `pending_phase` e marcar `pending_active`.
- Enquanto `pending_active`, o output do bin usa `pending` diretamente, evitando degeneração da janela.
- Em paralelo, aplicar normalmente `decay` + reforço na memória principal para que ela acompanhe.

## Etapa 2 — Handshake entre `pending` e memória
- Definir critério para liberar `pending` (ex.: após `pending_hold_frames` hops ou quando a memória atingir proximidade do input).
- Ao liberar, copiar a memória atual para produção e limpar `pending_*`. Garantir que transições não criem degraus.
- Opcional: tornar `pending_hold_frames` derivado do overlap real (p.ex. 1 hop = default).

## Etapa 3 — Validação
- Regenerar `output.wav` com o patch Pd atual.
- Rodar `analyze_spectrail.py` (waveforms + spectrogram). Confirmar ausência de impulsos no residual.
- Se falha, ajustar lógica (ex.: fundir `pending` e memória gradualmente em vez de passo brusco).

## Etapa 4 — Reintrodução incremental de recursos
- Reativar parâmetros extras um a um (`attackadapt`, `attackonset`, etc.) e validar novamente.
- Atualizar documentação e roadmap principal após estabilizar o comportamento.

## Métricas de sucesso
- Residual não apresenta pulsos > -20 dBFS alinhados ao hop.
- Forma de onda inicial do output não exibe a "mordida" triangular.
- Decay contínuo funciona conforme tempo configurado.

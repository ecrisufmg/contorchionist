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
- **`attackonset`** (`attackonset <0..1>`; valores negativos desativam): substitui o ataque final por um coeficiente fixo quando `memory <= onsetfloor` e o bin supera o `threshold` pelo tempo necessário definido pela histerese.
- **`attackonsetramp`** (`attackonsetramp <frames>` / `attackonsetramptime <segundos>`): suaviza a transição entre `attack` e `attackonset`. O ataque especial entra gradualmente ao longo do número de frames (ou tempo) especificado, reduzindo estalos.
- **`onsetfloor`** (`onsetfloor <f>`): piso de magnitude que define quando um bin é considerado “zerado”. Ajuste para separar ruído residual de silêncio real. Valores muito altos podem tratar bins fracos como novos com excessiva frequência.
- **`onsethyst`** (`onsethyst <frames>` / `onsethystime <segundos>`): exige que o bin permaneça acima do `threshold` por pelo menos esse tempo antes de acionar `attackonset`. Evita disparos em falsos positivos ou jitter.
- **`frametiming`** (`frametiming <sample_rate> <hop>`): informa explicitamente o par (taxa de amostragem, hop size) para que parâmetros baseados em tempo (`attackonsetramptime`, `onsethystime`, `decay6db`) sejam convertidos corretamente em frames. Normalmente não é necessário: o wrapper segue `sys_getsr()` automaticamente e conserva o hop passado na criação (`@hopsize`/`@hop`) ou via mensagem `hopsize`/`hop`. Utilize quando quiser forçar valores específicos ou sincronizar vários objetos manualmente.

## Reset Parcial
- **`resetframes`** (`resetframes <int>` / `resettime <segundos>`): conta quantos frames (ou tempo equivalente) um bin permaneceu com `input <= threshold`. Com zero, a funcionalidade fica desligada.
- **`resetmult`** (`resetmult <0..1>`): multiplicador aplicado à memória quando `resetframes` é atingido. `0.0` limpa completamente; valores intermediários aceleram o decaimento; `1.0` mantém a magnitude intacta (apenas zera a fase).

## Relação com Threshold, Decay e Limiter
- `threshold` define as regiões de reforço versus pura atenuação. Ele dirige tanto o ataque (pois determina os bins elegíveis) quanto o reset (contagem só ocorre enquanto a entrada ficar abaixo dele).
- `decay`/`decay6db` atuam sempre antes do ataque. Um decay lento exige ataques mais altos ou resets para evitar resquícios; um decay muito rápido exige vigiar `attackonset` para que o sustain não desmorone.
- O `limiter` é aplicado após todas as atualizações de memória. Assim, mesmo com `attackonset` próximo de 1.0, o valor final fica contido. `limitersoft` controla o joelho suave para preservar continuidade.

## Estratégias Para Evitar Artefatos de Ataque
1. **Sincronize tempo e hop**: mantenha `@hop`/`@hopsize` coerente com o deslocamento real; a taxa de amostragem é atualizada automaticamente pelo wrapper. Caso precise forçar combinações específicas (por exemplo, em roteamentos multirrepique ou para compensar downsampling), dispare `frametiming <sr> <hop>` para alinhar todos os objetos.
2. **Combine histerese com piso adequado**: use `onsetfloor` ligeiramente acima do ruído residual e defina `onsethyst 2` (ou `onsethystime 0.02`) para garantir que apenas onsets reais disparem o ataque rápido.
3. **Use rampas para transientes suaves**: valores como `attackonset 0.9` com `attackonsetramp 3` (ou `attackonsetramptime 0.01`) suavizam a transição e evitam cliques mesmo em sinal rico em harmônicos.
4. **Trate bins espaçados no tempo**: `resettime 0.1` combinado com `resetmult 0.2` remove resíduos sem derrubar o sustain de notas longas, reduzindo reativações abruptas.
5. **Recalibre `attackadapt`**: se o ataque parecer “duro”, diminua `attackadapt` para valores entre `0.1` e `0.3`. Se estiver lento demais, aumente para `0.6+` e compense com rampas maiores.
6. **Vigie o limiter**: `limitersoft 0.3`–`0.8` costuma evitar cortes bruscos após um grande ataque. Ajuste `maxvalue` para manter headroom adequado.

### Fluxos sugeridos
- **Percussão seca**: `frametiming 48000 256`, `threshold 0.015`, `attack 0.65`, `attackadapt 0.5`, `attackonset 0.95`, `attackonsetramp 2`, `onsethyst 2`, `resettime 0.08`, `resetmult 0.35`, `limitersoft 0.5`.
- **Pads sustentados**: `threshold 0.01`, `attack 0.4`, `attackadapt 0.2`, `attackonset 0.75`, `attackonsetramptime 0.03`, `onsetfloor 1e-4`, `resetframes 0`, `limitersoft 0.2`.
- **Material ruidoso**: elevar `onsetfloor` para `5e-3`, `onsethystime 0.03` e baixar `attackonset` para `0.7`; combine com `resetmult 0.5` para evitar que ruído branco cause pumping.

## Mensagens Pure Data
Todas as configurações podem ser feitas por argumentos na criação (`@attack 0.75`, `@attackonset 0.9`, etc.) ou via mensagens. Exemplos:
```
frametiming 48000 256
threshold 0.02
attack 0.6
attackadapt 0.35
attackonset 0.9
attackonsetramp 3
onsetfloor 0.0005
onsethyst 2
resettime 0.12
resetmult 0.3
hop 256
decay6db 1.5
limiter 1
limitersoft 0.4
maxvalue 1.2
```
Verifique o console do Pd: cada mensagem gera um `post` confirmando o valor aplicado. Reenvie `frametiming` após alterar `hopsize` ou sample rate.

# Sugestões de Arquitetura e Simplificação: `sntv2` + `torch.spectrails`

O objetivo é simplificar o patch, garantir consistência entre os objetos e aumentar o controle criativo sobre a textura (circulação de vozes vs. sustentação), mantendo a lógica de alocação SATB.

## 1. Diagnóstico do Problema Atual

A complexidade atual advém da **duplicação de responsabilidades**:
1.  **C++ (`torch.spectrails`)**: Controla a "vida útil" do som via `decay` espectral.
2.  **Lua (`sntv2`)**: Controla o disparo e a alocação, mas também tenta manipular o tempo via `jitter` e `dephase`.
3.  **Patch PD**: Tenta coordenar os dois anteriores enviando mensagens separadas.

A "inconsistência" (menos notas no `spectrails`) provavelmente ocorre porque os parâmetros de *Threshold* e *Gain* no C++ não estão casando com o *Gate (mindb)* no Lua. Se o C++ decai um pouco mais rápido ou tem um ganho ligeiramente menor, o Lua corta a nota prematuramente.

---

## 2. Proposta de Solução: "Olho vs. Cérebro"

Vamos separar as funções claramente:
*   **`torch.spectrails` (O Olho)**: Deve apenas **detectar** o que existe, com parâmetros mais "crus". Não tente fazer a "música" aqui (ex: decays muito longos). Deixe-o sensível.
*   **`sntv2` (O Cérebro)**: Deve decidir **como** cantar o que foi detectado. A manipulação artística (textura, duração, atraso) deve acontecer aqui.

### A. Simplificação de Parâmetros (Abstração Mestra)

Em vez de controlar cada objeto individualmente, crie uma abstração `[choir_control]` que receba parâmetros musicais de alto nível e os traduza para os dois objetos.

**Parâmetros Sugeridos para a Abstração:**

1.  **Density (Densidade)**:
    *   Afeta `threshold` (C++) e `mindb` (Lua).
    *   *Baixa*: Só picos muito fortes passam.
    *   *Alta*: Qualquer sussurro espectral vira nota.

2.  **Blur / Smear (Borrão)**:
    *   Afeta `decay` (C++) E `jitter` (Lua).
    *   *Baixo*: Notas curtas, precisas, rítmicas.
    *   *Alto*: Decaimento espectral longo + jitter alto (nuvem de som).

3.  **Spread (Abertura)**:
    *   Afeta `dephase` (Lua).
    *   Controla o atraso entre as entradas das vozes.

### B. Novas Funcionalidades no `sntv2` (Lua)

Para permitir a manipulação de "circulação" vs "prisão" de notas sem depender apenas do FFT, sugiro implementar no Lua:

#### 1. Modo "Sustain/Freeze" (Independente do Espectro)
Um parâmetro `freeze` (0 ou 1).
*   Se `1`: O Lua ignora mensagens de "Note Off" (flag -1 ou 0 com amplitude baixa) vindas do C++. As vozes atuais continuam cantando indefinidamente, mesmo que o espectro mude.
*   Isso permite criar acordes estáticos a partir de um momento espectral.

#### 2. Parâmetro "Voice Lifetime" (Rotação Forçada)
Para criar a "circulação" que você deseja.
*   Adicionar um parâmetro `max_duration` (ms).
*   Se uma voz ficar alocada por mais tempo que `max_duration`, o Lua força um *Release* e coloca essa voz em um tempo de "resfriamento" (cooldown), obrigando o alocador a escolher **outra voz** ou outra nota para a mesma parcial.
*   **Efeito**: A textura fica "viva", as vozes trocam de lugar mesmo em um acorde sustentado.

#### 3. "Smart Gate" Automático
Em vez de configurar `mindb` manualmente e correr o risco de silenciar o `spectrails`:
*   Fazer o `sntv2` ler o `gaindb` que você envia para o `spectrails`.
*   O `mindb` interno seria calculado automaticamente: `mindb = threshold_db_do_cpp + gaindb - margem_seguranca`.

---

## 3. Roteiro de Implementação

1.  **Padronização**:
    *   Garanta que `torch.spectrails` e `torch.amb.spectrails` recebam **exatamente** os mesmos valores de `gain`, `pregain` e `threshold`.
    *   Use o `[print]` sugerido na análise anterior para calibrar o `mindb` do Lua.

2.  **Atualização do `sntv2.pd_lua`**:
    *   Implementar `max_duration` (para forçar rotação de vozes).
    *   Implementar `freeze` (para ignorar note-offs).

3.  **Limpeza do Patch**:
    *   Remova os envios diretos de parâmetros espalhados.
    *   Centralize em uma única interface que envia para `sntv2` e `spectrails` simultaneamente.

## 4. Exemplo de Lógica para "Rotação" (Lua)

```lua
-- No loop tick()
if self.max_duration > 0 and voice.active then
    local duration = now - voice.start_time
    if duration > self.max_duration then
        -- Força release para dar lugar a outra voz/nota
        self:schedule_output(voice, voice.last_midi_note, 0, -144, -1)
        voice.active = false
        voice.cooldown = now + 0.5 -- 500ms de descanso
    end
end
```

Isso criaria o movimento orgânico que você busca, independente da estabilidade do FFT.

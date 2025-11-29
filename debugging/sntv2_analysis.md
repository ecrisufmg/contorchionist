# Análise de sntv2 e torch.spectrails~ / torch.amb.spectrails~

Este documento analisa o funcionamento do objeto Lua `sntv2` e sua interação com os externals `torch.spectrails~` e `torch.amb.spectrails~`, investigando a inconsistência relatada (menos listas de picos geradas pelo `torch.spectrails~`).

## 1. Visão Geral do Fluxo de Dados

1.  **Externals C++ (`torch.spectrails~` / `torch.amb.spectrails~`)**:
    *   Recebem áudio (FFT/sinal).
    *   Processam espectralmente (detecção de picos, envelopes).
    *   Geram listas de picos detectados via `outlet_list` (último outlet).
    *   Formato da lista: `[bin_index, freq, mag_val, state, rank]`
        *   `bin_index`: Índice do bin da FFT (usado como ID estável).
        *   `freq`: Frequência em Hz (ou MIDI se `midi_mode` ativo).
        *   `mag_val`: Magnitude em dB (ou Velocity se `velocity_mode` ativo).
        *   `state`: 1 (novo/ataque), 0 (sustentação), -1 (fim/release).
        *   `rank`: Classificação por magnitude (0 = mais forte).

2.  **Objeto Lua (`sntv2`)**:
    *   Recebe a lista no **Inlet 1**.
    *   Interpreta: `atoms[1]` (bin_index), `atoms[2]` (freq), `atoms[3]` (db), `atoms[4]` (flag/state).
    *   **Alocação de Vozes**: Usa `bin_index` para rastrear parciais.
        *   Se `flag == 1` (Novo): Aloca uma voz disponível (S, A, T, B) baseada em tessitura e distância.
        *   Se `flag == 0` (Sustentação): Atualiza a voz correspondente ao `bin_index`.
        *   Se `flag == -1` (Fim): Libera a voz.
    *   **Gate (`mindb`)**: Filtra notas com amplitude abaixo de `self.min_db`.
    *   **Agendamento**: Adiciona eventos a uma fila com *jitter* e *dephase* para humanização.
    *   **Saída**: Envia listas formatadas para síntese/notação.

## 2. Comparação dos Externals C++

A lógica de detecção e envio de mensagens é **idêntica** em ambos os externals, baseada na classe `SpectralTrailsProcessor` (`core_ap_spectrails.h`).

### torch.spectrails~ (Mono/Stereo)
*   **Processador**: Instância única de `SpectralTrailsProcessor`.
*   **Entrada**: Magnitude e Fase (inlets de sinal).
*   **Detecção**: Baseada no sinal de entrada único.
*   **Saída de Controle**: Envia picos detectados pelo processador único.

### torch.amb.spectrails~ (Ambisonic)
*   **Processadores**: Múltiplas instâncias (uma por canal).
*   **Entrada**: Múltiplos canais (W, X, Y, Z...).
*   **Detecção**: **Exclusivamente baseada no Canal 0 (W)**.
    *   `x->processors_[0]` é o mestre da detecção.
    *   Os outros canais são forçados a seguir os envelopes de W para manter a coerência espacial.
*   **Saída de Controle**: Envia picos detectados por `x->processors_[0]` (W).

### Conclusão da Comparação de Código
Se o sinal conectado ao `torch.spectrails~` for o mesmo sinal conectado ao primeiro canal (W) do `torch.amb.spectrails~`, e os parâmetros (threshold, attack, decay, etc.) forem idênticos, **a saída de dados deve ser idêntica**.

## 3. Possíveis Causas para a Discrepância

Se o usuário percebe "menos listas" (menos atividade) no `torch.spectrails~`, as causas prováveis são:

1.  **Diferença de Sinal de Entrada**:
    *   O `torch.amb.spectrails~` usa o canal W (omnidirecional). Se o `torch.spectrails~` estiver recebendo um sinal diferente (ex: um microfone direcional ou uma mixagem diferente), a detecção será diferente.
    *   **Verificação**: Certifique-se de que o sinal entrando no `torch.spectrails~` é exatamente o mesmo que entra no primeiro inlet do `torch.amb.spectrails~`.

2.  **Diferença de Parâmetros (Threshold/Gain)**:
    *   O `sntv2` agora possui um **Gate (`mindb`)**. Se o `torch.spectrails~` estiver enviando valores de dB ligeiramente menores que o `torch.amb.spectrails~`, eles podem estar sendo cortados pelo gate.
    *   **Ganho de Saída**: Verifique se `gaindb` ou `gain` está configurado de forma diferente. Ambos aplicam ganho na saída de áudio E na magnitude reportada na lista.
    *   **Pregain**: Verifique se `pregaindb` está igual.

3.  **Sobrecarga de Mensagens (Pd Scheduler)**:
    *   Ambos usam `clock_delay(x->info_clock_, 0)` para enviar dados. Se o `torch.spectrails~` estiver em um patch mais leve, ele poderia teoricamente disparar *mais* rápido, mas o buffer é atualizado por bloco de áudio.
    *   Se o `torch.amb.spectrails~` consome mais CPU (processando 4+ canais), isso pode alterar sutilmente o agendamento do Pd, mas não deveria *aumentar* a quantidade de mensagens úteis, a menos que o `torch.spectrails~` esteja sofrendo *dropouts* de mensagens por excesso de fluxo (o que é raro para esse tipo de dado).

4.  **Configuração de `max_peaks`**:
    *   Verifique se o argumento `max_peaks` (ou `mp`) está limitando o número de parciais reportadas no `torch.spectrails~`.

## 4. Análise do `sntv2.pd_lua`

O código Lua parece robusto para lidar com o fluxo.
*   **Gate (`mindb`)**:
    ```lua
    if db < self.min_db then
        if flag == 1 then return end -- Ignora novos
        if flag == 0 then flag = -1 end -- Força fim
    end
    ```
    Isso é crítico. Se o `torch.spectrails~` reportar -61dB e o gate for -60dB, a nota não toca. Se o `torch.amb.spectrails~` reportar -59dB (por alguma diferença de ganho ou soma de canais interna - embora a detecção seja só em W), a nota toca.

*   **Alocação de Vozes**:
    *   Depende estritamente de receber o `flag == 1` (Attack) ou `flag == 0` (Update) para um `bin_index` não rastreado.
    *   Se mensagens forem perdidas, notas podem não disparar.

## 5. Ação Recomendada

1.  **Monitorar Saída Bruta**: Use um `[print]` logo após a saída dos objetos `torch.*` (antes do `sntv2`) para comparar os valores de dB brutos.
2.  **Verificar Níveis**: Confirme se os níveis de dB reportados são consistentes entre os dois objetos para o mesmo sinal.
3.  **Ajustar `mindb`**: Tente diminuir o `mindb` no `sntv2` (ex: -144) para ver se as notas reaparecem no `torch.spectrails~`.

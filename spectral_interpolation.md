# Estratégia de Interpolação Espectral

Este documento descreve a estratégia de interpolação de frequência utilizada no código `torch.amb.spectrails~.cpp` (e `core_ap_spectrails.h`) e propõe uma mecânica inversa para a criação de filtros "brickwall" com precisão de frequência.

## 1. Estratégia de Interpolação (Bins -> Frequência)

O código utiliza uma técnica de **Interpolação Parabólica** para refinar a estimativa de magnitude e fase dos picos espectrais, além de uma estratégia de **Redistribuição de Energia** para posicionar o pico "entre bins" na ressíntese.

### A. Detecção e Cálculo do "Detune"
Para cada bin $i$ identificado como um pico local (maior que seus vizinhos e acima do threshold), o algoritmo calcula um fator de desvio (`detune` ou $\delta$) que indica o quão longe o verdadeiro pico está do centro do bin $i$.

A fórmula utilizada no código é uma variação da interpolação parabólica que considera as magnitudes ao quadrado (potência) para o numerador e a curvatura linear para o denominador:

$$ \delta = \frac{M_{right}^2 - M_{left}^2}{2 \cdot (M_{left} + M_{right} - 2 M_{center})} $$

Onde:
- $M_{center}$ é a magnitude do bin do pico ($i$).
- $M_{left}$ é a magnitude do bin $i-1$.
- $M_{right}$ é a magnitude do bin $i+1$.
- O resultado é clampeado entre -0.5 e 0.5.

### B. Correção de Magnitude e Fase
Com base no `detune` calculado:

1.  **Correção de Amplitude**: A magnitude é corrigida para compensar a atenuação causada pela janela (scalloping loss). O código aplica uma correção baseada na janela de Hann:
    $$ M_{interp} = M_{center} \cdot \frac{1}{1 - 0.5 \cos(\pi \delta)} $$

2.  **Interpolação de Fase**: A fase é ajustada linearmente em direção à fase do vizinho para onde o pico se inclina:
    $$ \phi_{interp} = \phi_{center} + |\delta| \cdot (\phi_{neighbor} - \phi_{center}) $$

### C. Redistribuição de Energia (Ressíntese)
Ao invés de simplesmente reportar a frequência interpolada, o algoritmo utiliza o `detune` para **redistribuir a energia do pico entre dois bins** na memória espectral (que será usada para a ressíntese/trails).

- Se $|\delta| > 0.01$:
    - O bin principal ($i$) recebe uma fração da energia: $1 - |\delta|$.
    - O bin adjacente ($i \pm 1$) recebe o restante: $|\delta|$.
- Isso cria um efeito de "anti-aliasing" espectral, permitindo que o rastro (trail) do pico decaia em uma posição virtual entre os bins, suavizando a transição de frequências.

---

## 2. Mecânica Inversa: Filtros Brickwall com Precisão de Frequência

Para desenhar filtros "brickwall" (passa-baixa, passa-alta, passa-banda) com precisão de frequência arbitrária (não limitada a bins inteiros), deve-se calcular a **cobertura fracionária** de cada bin em relação à banda de frequência desejada.

### O Conceito
Um bin $k$ na FFT não representa apenas a frequência central $f_k = k \cdot \frac{SR}{N}$, mas cobre teoricamente a faixa $[k - 0.5, k + 0.5]$ em unidades de bins (ou uma largura de banda de $SR/N$ centrada em $f_k$).

Para um filtro ideal que vai de $F_{start}$ a $F_{end}$ (em Hz), o ganho $G_k$ de cada bin $k$ deve ser proporcional à área desse bin que está "dentro" da faixa do filtro.

### Algoritmo

1.  **Converter Frequências para Bins Fracionários**:
    $$ b_{start} = F_{start} \cdot \frac{N_{fft}}{SR} $$
    $$ b_{end} = F_{end} \cdot \frac{N_{fft}}{SR} $$

2.  **Calcular o Ganho por Bin (Máscara)**:
    Para cada bin $k$ (de 0 a $N/2$):
    
    - Definir os limites do bin $k$:
        $$ k_{min} = k - 0.5 $$
        $$ k_{max} = k + 0.5 $$
    
    - Calcular a interseção entre o intervalo do bin $[k_{min}, k_{max}]$ e o intervalo do filtro $[b_{start}, b_{end}]$:
        $$ overlap_{start} = \max(k_{min}, b_{start}) $$
        $$ overlap_{end} = \min(k_{max}, b_{end}) $$
    
    - O ganho é o tamanho da interseção (limitado a 0 se não houver interseção):
        $$ G_k = \max(0, overlap_{end} - overlap_{start}) $$

### Exemplo Prático

Se quisermos um filtro Low Pass em 1000 Hz, com $SR=48000$ e $N=1024$:
- Largura do bin $\approx 46.875$ Hz.
- $b_{cutoff} = 1000 / 46.875 = 21.333$ bins.

Para o bin 21 (cobre 20.5 a 21.5):
- $overlap_{start} = \max(20.5, 0) = 20.5$ (assumindo filtro começa em 0)
- $overlap_{end} = \min(21.5, 21.333) = 21.333$
- $G_{21} = 21.333 - 20.5 = 0.833$

Para o bin 22 (cobre 21.5 a 22.5):
- $overlap_{start} = \max(21.5, 0) = 21.5$
- $overlap_{end} = \min(22.5, 21.333) = 21.333$
- $overlap_{end} < overlap_{start}$, logo $G_{22} = 0$.

Bins 0 a 20 terão ganho 1.0. O bin 21 terá ganho 0.833. Bins 22+ terão ganho 0.0.
Isso suaviza a borda do filtro (anti-aliasing da máscara), evitando artefatos abruptos de quantização de frequência.

# Como Usar o Waveshaper com torch.ts~

## Estrutura do Patch

```
[osc~ 440]                    <- gera sinal de teste
    |
[*~ 0.5]                      <- reduz amplitude
    |
[torch.ts~ waveshaper.ts]     <- processa com o modelo
    |         |       |    |
    |         |       |    +-- [mix]     inlet 3: dry/wet (0-1)
    |         |       +------- [tone]    inlet 2: brilho (0-1)
    |         +--------------- [drive]   inlet 1: distorção (0-1)
    +------------------------- inlet 0: áudio (sinal contínuo)
    |
[dac~]                        <- saída de áudio
```

**Importante**: O `torch.ts~` processa automaticamente o áudio em blocos do tamanho 
configurado no PureData (geralmente 64 samples). Não é necessário usar `snake~` para 
converter o sinal - o objeto faz isso internamente!

## Parâmetros

### Drive (Inlet 1)
- **Valor**: 0.0 a 1.0
- **Efeito**: Controla a quantidade de distorção
  - 0.0 = limpo (gain 1x)
  - 0.5 = distorção média (gain 5.5x)
  - 1.0 = distorção pesada (gain 10x)
- **Como enviar**: Conecte um número ou slider ao inlet 1

### Tone (Inlet 2)
- **Valor**: 0.0 a 1.0
- **Efeito**: Controla o brilho (ênfase de frequências altas)
  - 0.0 = escuro (corta agudos)
  - 0.5 = neutro (resposta plana)
  - 1.0 = brilhante (realça agudos)
- **Como enviar**: Conecte um número ou slider ao inlet 2

### Mix (Inlet 3)
- **Valor**: 0.0 a 1.0
- **Efeito**: Mistura dry/wet
  - 0.0 = 100% dry (sinal original)
  - 0.5 = 50% wet / 50% dry
  - 1.0 = 100% wet (totalmente processado)
- **Como enviar**: Conecte um número ou slider ao inlet 3

## Exemplo Prático

### Patch mínimo:
```
[osc~ 440]
    |
[torch.ts~ waveshaper.ts]
    |
[dac~]
```

### Objeto básico:
```
[torch.ts~ waveshaper.ts -m forward]
```

### Com modo verboso:
```
[torch.ts~ waveshaper.ts -m forward -v]
```
Mostra informações sobre os canais detectados:
```
torch.ts~: forward_in_ch = 1
torch.ts~: forward_out_ch = 1
```

### Com processamento assíncrono:
```
[torch.ts~ waveshaper.ts -m forward -async 512]
```
Usa buffer de 512 amostras para reduzir latência

## Conectando Parâmetros

### Opção 1: Sliders (recomendado)
```
[hsl 128 15 0 1 0 0 empty empty drive]
    |
[floatatom]
    |
[torch.ts~ waveshaper.ts] inlet 1
```

### Opção 2: Number boxes diretos
```
[nbx 5 14 0 1 0 0 empty empty drive]
    |
[torch.ts~ waveshaper.ts] inlet 1
```

### Opção 3: Mensagens
```
[0.7(  <- clique para enviar o valor
    |
[torch.ts~ waveshaper.ts] inlet 1
```

## Valores Recomendados

### Para Distorção Sutil:
- drive: 0.2 - 0.4
- tone: 0.5 - 0.7
- mix: 0.5 - 0.8

### Para Distorção Pesada:
- drive: 0.7 - 1.0
- tone: 0.3 - 0.5 (evita excesso de brilho)
- mix: 0.8 - 1.0

### Para Efeito Criativo:
- drive: 0.8
- tone: 0.8 (muito brilhante)
- mix: 1.0

## Troubleshooting

### "No method found": 
O modelo não foi carregado. Verifique se `waveshaper.ts` está no mesmo diretório do patch.

### Sem som:
1. Verifique se o DSP está ligado (`[; pd dsp 1(`)
2. Verifique se os sliders estão gerando valores (use `[print]` para debug)
3. Certifique-se de que o áudio está entrando no inlet 0

### Som distorcido demais:
- Reduza o `drive`
- Reduza o `mix` para misturar mais sinal limpo
- Reduza a amplitude do sinal de entrada antes do `[snake~ in]`

### Latência alta:
Use o modo assíncrono com buffer menor:
```
[torch.ts~ waveshaper.ts -async 128]
```

## Arquivos de Teste

- **test_waveshaper.pd**: Patch completo de teste com sliders
- **waveshaper_effect.pd**: Patch original com estrutura básica
- **03_Waveshaper_audiotest_model.py**: Script Python que gera arquivos WAV de teste

## Performance do Modelo

- **Erro médio (MAE)**: ~0.039 (muito bom!)
- **Latência**: 64 amostras = 1.33ms @ 48kHz
- **Canais**: 1 entrada, 1 saída (mono)
- **Processamento**: Stateless (sem memória entre blocos)

1. Mensagem de Animação (O método "One-Shot")
Esta é a forma mais comum. Você envia uma lista com 3 elementos para o primeiro inlet:
[ValorAlvo Tempo(ms) NomeDaCurva]
Exemplos de mensagens (message boxes):
Movimento Suave (Cubic): Ir para 0.8 em 1 segundo.
code
Text
0.8 1000 cubic-out
Efeito Elástico (Back): Ir para -10dB (no modo db) com um pequeno "overshoot" (passa um pouco e volta).
code
Text
-10 600 back-out
Efeito de Pulo (Bounce): Simular um objeto caindo e quicando no zero.
code
Text
0 1500 bounce-out
Aceleração Exponencial: Começar muito devagar e acelerar no final (útil para fade-outs dramáticos).
code
Text
0 2000 expo-in
2. Definir o Padrão (Global)
Se você quiser definir um tipo de curva e depois apenas enviar valores e tempos sem repetir o nome da curva, use a mensagem curve.
Envie: curve back-out
Agora envie: 0.5 1000 (Isso usará back-out automaticamente).
3. Tabela de Nomes Disponíveis
Aqui estão todas as strings que você pode usar na terceira posição da lista.
Nota: Se você escrever apenas o nome base (ex: cubic), o código assume automaticamente o modo -inout.
Família	Nomes (Sufixos: -in, -out, -inout)	Sensação / Uso em UX
Linear	linear, lin	Robótico, mecânico. Sem aceleração.
Sine	sine-in, sine-out, sine-inout	Suave, "padrão". Bom para fades de áudio.
Quad	quad-in, quad-out, quad-inout	Aceleração leve (Potência de 2).
Cubic	cubic-in, cubic-out, cubic-inout	O favorito para UI. Movimento natural.
Quart	quart-in, quart-out, quart-inout	Mais acentuado que o Cubic.
Quint	quint-in, quint-out, quint-inout	Muito rápido no meio, parada suave.
Expo	expo-in, expo-out, expo-inout	Dramático. Bom para zooms ou entradas rápidas.
Circ	circ-in, circ-out, circ-inout	Rápido, mas "arredondado".
Back	back-in, back-out, back-inout	Elástico. Passa do alvo e volta. Ótimo para dar "peso".
Bounce	bounce-in, bounce-out, bounce-inout	Quicar. A bola bate e quica até parar.
Dicas de UX para escolher a direção (in vs out)
Use -out (Ex: cubic-out, back-out):
Quando o objeto entra na tela ou se move para uma posição final.
O movimento começa rápido (resposta imediata ao clique) e desacelera suavemente até parar. É o que "parece certo" para 90% dos sliders de interface.
Use -in (Ex: cubic-in):
Quando o objeto está saindo da tela. Ele começa devagar e acelera até sumir.
Use -inout (Ex: sine-inout):
Para loops ou automações contínuas (ex: um LFO visual). Evita mudanças bruscas de direção.
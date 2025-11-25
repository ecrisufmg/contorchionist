Vamos revisar o objeto sntv.

À esquerda entra sempre uma lista <num_parcial> <frequencia> <db> <flag>. 
No meio temos um controle de openess (o quão próximas serão alocadas as notas nas vozes)
No terceiro inlet temos um tempo em <min> <seg> (segundos com ponto flutuante)

O objeto tem duas funções: criar uma lista de notas a serem sintetizadas separadamente a partir dos parâmetros processados e guardar a informação de tempo junto com as demais informações usadas para síntese. 

Listas da entrada 1:
<num_parcial> <frequencia> <db> <flag>
A flag pode ser:
    1, para novo parcial;
    0, para continuar um parcial;
    -1, fim de parcial

Os atributos de inicialização do objetos são processados com a função ArgParser (pd_arg_parser.lua)

Os atributos permitem:
    - delimitar o range midi de cada voz (S, A, T, B)
    - delimitar o número de vozes por tipo (S, A, T, B)

Em resumo, o processamento do objeto faz o seguinte: recebe as listas de parciais, e para cada parcial novo, aloca ele transposto em oitavas dentro do range de uma voz livre (flag -1). Para isso, precisamos definir uma estratégia de transposição e alocação de vozes que segue o seguinte princípio:

1. O cálculo de alocação de voz só ocorre para vozes que estejam livres (flag -1). Se uma voz está ocupada não faremos cálculos para ela quando ocorrer um novo parcial, apenas continuaremos o parcial nela (mantendo a mesma transposição usada no início).
2. Após converter a freq para midi, vamos transpor a altura às notas possíveis para cada voz, em oitavas diferentes, dentro do range de cada voz disponível/livre (se uma nota cabe 2 vezes no range, a voz terá duas notas candidatas)
3. Decidir em função:
3.1 da menor distância em relação a ultima nota cantada pela voz
3.2 da distribuição (abertura em termos de somatoria de distancia da nota cantada e das outras notas cantadas no momento pelas outras vozes). Range será informado como um parâmetro a mais no segundo inlet (deixaremos "time" no 3 inlet)



criaremos um processo que aloca o próximo parcial que chega à próxima voz que não está sustentando algum parcial. Os parciais devem ser converitdo para midi e transpostos em oitava ao range da voz livre, buscando ainda criar a maior distância possível em relação a outros parciais que estão em estado de sustentados naquele momento (o objeto deve ter uma memória do estado de cada voz, portanto).

crie o objeto em lua na pasta wrappers/puredata/sn_pdlua
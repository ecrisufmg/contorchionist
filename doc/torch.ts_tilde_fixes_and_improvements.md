# Guia de Correções e Melhorias: `torch.ts~`, `core_nn_ts_manager` e `core_ap_torchts`

Este documento detalha todos os problemas identificados na revisão técnica dos arquivos [`core_nn_ts_manager.cpp`](../core/src/core_nn_ts_manager.cpp), [`torch.ts~.cpp`](../wrappers/puredata/src/torch.ts~.cpp), [`core_ap_torchts.cpp`](../core/src/core_ap_torchts.cpp) e [`pd_torch_types.h`](../wrappers/puredata/utils/include/pd_torch_types.h), acompanhados das causas-raiz e das propostas de correção no código.

---

## Sumário de Prioridades

| ID | Prioridade | Categoria | Arquivo Principal | Resumo |
| :--- | :--- | :--- | :--- | :--- |
| **1.1** | Alta (Bug) | Gerenciamento do Modelo | `core_nn_ts_manager.cpp` | Variável local sombreia (`shadowing`) o parâmetro de saída `max_buffer_size` |
| **1.2** | Média | Gerenciamento do Modelo | `core_nn_ts_manager.cpp` | `extract_channels_method` retorna `true` mesmo quando atributos obrigatórios faltam |
| **1.3** | Baixa | Gerenciamento do Modelo | `core_nn_ts_manager.cpp` | `extract_shape_method` cria tensor de teste sempre na CPU |
| **2.1** | Alta (Crash/UB) | Memória / Ciclo de Vida C++ | `torch.ts~.cpp` / `pd_torch_types.h` | Membros C++ não-triviais em `t_torch_ts_tilde` alocados via `pd_new` sem construtor/destrutor |
| **2.2** | Alta (Crash) | Buffer / DSP | `torch.ts~.cpp` | `x->in_blocks` não redimensiona os vetores internos em `torch_ts_tilde_AddDsp` |
| **3.1** | Alta (DSP) | Fluxo de Sinal / Janelamento | `torch.ts~.cpp` / `core_ap_torchts.cpp` | Descompasso entre descarte de `block_size` na entrada e escrita de `buffer_size` na saída |
| **3.2** | Média (DSP) | Capacidade do Buffer Circular | `core_ap_torchts.cpp` | Capacidade do `CircularBuffer` igual a `1x buffer_size` causa overwrite/underrun no modo assíncrono |
| **4.1** | Média (Thread-Safety) | CPU / GPU / Concorrência | `torch.ts~.cpp` | `torch_ts_tilde_device` e `torch_ts_tilde_set_parameter` acessam o modelo sem `model_mutex` |
| **4.2** | Baixa (Performance) | CPU / GPU / Memória | `core_ap_torchts.cpp` | Cópia redundante `.clone().to(device)` na entrada e falta de `.contiguous()` na saída |
| **5.1** | Média (Bug) | Batching / Validação | `core_ap_torchts.cpp` | `suggest_optimal_buffer_size` usa `model_buffer_size` como teto máximo em vez de `max_buffer_size` |
| **5.2** | Média (Modelos) | Batching / Formato de Tensor | `core_ap_torchts.cpp` | Inconsistência de dimensionalidade (3D com batching vs 2D sem batching) e impacto em modelos causais/recorrentes |
| **6.1** | Alta (RT-Safety) | Multithreading | `torch.ts~.cpp` | Criação de `std::thread` + `detach()` e alocação dinâmica a cada ciclo DSP na thread de áudio |
| **6.2** | Média (Thread-Safety) | Multithreading / Logging | `torch.ts~.cpp` | Chamada de `pd_error()` dentro da thread secundária do modelo |

---

## 1. Gerenciamento do Modelo (`core_nn_ts_manager.cpp`)

### 1.1. Sombreamento (`Variable Shadowing`) de `max_buffer_size`
* **Arquivo:** `core/src/core_nn_ts_manager.cpp` (linhas 152–163)
* **Problema:** Dentro de `extract_channels_method`, uma nova variável local `int max_buffer_size` é declarada dentro do bloco `if`, sombreando a referência `int& max_buffer_size` recebida nos parâmetros da função. Assim, o valor lido do modelo é descartado ao sair do escopo e `x->max_buffer_size` nunca é atualizado.

**Código Atual:**
```cpp
std::string max_buffer_size_attr = "max_buffer_size";
if (model->hasattr(max_buffer_size_attr)) {
    int max_buffer_size = model->attr(max_buffer_size_attr.c_str()).toInt(); // BUG: sombreia o parâmetro int& max_buffer_size
    if (log_messages) {
        log_messages->push_back(max_buffer_size_attr + ": Model max buffer size = " + std::to_string(max_buffer_size));
    }
}
```

**Correção Recomendada:**
```cpp
std::string max_buffer_size_attr = "max_buffer_size";
if (model->hasattr(max_buffer_size_attr)) {
    max_buffer_size = model->attr(max_buffer_size_attr.c_str()).toInt();
    if (log_messages) {
        log_messages->push_back(max_buffer_size_attr + ": Model max buffer size = " + std::to_string(max_buffer_size));
    }
}
```

---

### 1.2. Retorno de Sucesso Indevido em `extract_channels_method`
* **Arquivo:** `core/src/core_nn_ts_manager.cpp` (linhas 110–164)
* **Problema:** Se o modelo não possuir `<method>_in_ch` ou `<method>_out_ch`, a função atribui `-1` a `in_ch` / `out_ch`, escreve em `error_message`, mas ainda retorna `true` na linha 164. Além disso, `error_message` é sobrescrito pelos testes seguintes (`m_buffer_size`, `max_buffer_size`). Em `torch_ts_tilde_New`, `in_ch = -1` pode ser repassado para `create_circular_buffer` e `vector::resize(-1)` (gerando `std::bad_array_new_length`).

**Correção Recomendada:**
1. Retornar `false` caso `in_ch <= 0` ou `out_ch <= 0`, ou manter um valor padrão seguro (`1`) quando o atributo não existir.
2. Tratar `m_buffer_size` e `max_buffer_size` como atributos opcionais (registrando avisos em `log_messages` em vez de sobrescrever `error_message` crítico).

---

### 1.3. Criação de Tensor em `extract_shape_method` Sem Dispositivo do Modelo
* **Arquivo:** `core/src/core_nn_ts_manager.cpp` (linha 242)
* **Problema:** `*input_tensor = torch::randn(*input_shape, torch::TensorOptions().dtype(torch::kFloat));` cria o tensor sempre na CPU, mesmo que o modelo já tenha sido movido para CUDA ou MPS.
* **Correção Recomendada:** Aceitar `const torch::Device& device = torch::kCPU` na assinatura de `extract_shape_method` e usar `torch::TensorOptions().dtype(torch::kFloat).device(device)`.

---

## 2. Ciclo de Vida C++ e Gerenciamento de Memória (`torch.ts~.cpp` e `pd_torch_types.h`)

### 2.1. Construção e Destruição de Membros C++ em `t_torch_ts_tilde`
* **Arquivos:** `wrappers/puredata/utils/include/pd_torch_types.h` (linhas 218–281) e `wrappers/puredata/src/torch.ts~.cpp` (linhas 400–430, 681–690)
* **Problema:**
  1. `pd_new(torch_ts_tilde_class)` aloca a struct `t_torch_ts_tilde` usando `calloc` (memória bruta zerada) e **não invoca construtores C++**.
  2. `t_torch_ts_tilde` possui diversos membros C++ não-triviais diretamente na struct: `std::string`, `std::vector`, `std::unique_ptr`, `std::atomic`, `std::mutex`, `torch::Device`. Atribuir valores a `std::string`, `std::vector` ou `torch::Device` não construídos é **Comportamento Indefinido (Undefined Behavior)**.
  3. Em `torch_ts_tilde_Free`, apenas `delete x->tensors_struct;` é executado. Os destrutores de `x->model`, `x->in_buffers`, `x->out_buffers`, `x->in_blocks`, `x->model_mutex`, etc. **nunca são chamados**, causando vazamento de memória (memory leak) do modelo TorchScript e de todos os buffers a cada destruição do objeto.

**Correção Recomendada:**
Encapsular todo o estado C++ em uma struct/classe interna alocada com `new` no `torch_ts_tilde_New` e liberada com `delete` no `torch_ts_tilde_Free`, ou usar *placement new* e chamada explícita de destrutor para os membros C++:

```cpp
// Opção A (Recomendada): Ponteiro para struct C++ de implementação (PImpl)
struct TorchTsState {
    torch::Device device{torch::kCPU};
    std::unique_ptr<torch::jit::script::Module> model;
    std::string selected_method{"forward"};
    std::vector<std::string> available_methods;
    std::vector<std::string> available_attributes;
    std::vector<std::unique_ptr<contorchionist::core::util_circbuffer::CircularBuffer<float>>> in_buffers;
    std::vector<std::unique_ptr<contorchionist::core::util_circbuffer::CircularBuffer<float>>> out_buffers;
    std::vector<std::vector<float>> in_blocks;
    std::vector<const float*> in_ptrs;
    std::vector<t_sample*> in_sig_vec;
    std::vector<t_sample*> out_sig_vec;
    std::mutex model_mutex;
    // ...
};
```

---

### 2.2. Falta de Redimensionamento dos Blocos Internos em `torch_ts_tilde_AddDsp`
* **Arquivo:** `wrappers/puredata/src/torch.ts~.cpp` (linhas 342–353)
* **Problema:** Quando `!x->async_mode && x->block_size != x->buffer_size`, `x->in_blocks.resize(x->in_ch)` é chamado, mas o tamanho interno de cada canal (`block.resize(x->buffer_size)`) **não é atualizado**. Se o tamanho de bloco do Pure Data aumentar (por exemplo, com `[block~ 256]`), `x->in_buffers[ch]->peek(x->in_blocks[ch].data(), x->buffer_size)` fará escrita fora dos limites (*buffer overflow*).

**Correção Recomendada (em `torch_ts_tilde_AddDsp`):**
```cpp
if (!x->async_mode && x->block_size != x->buffer_size) {
    x->buffer_size = x->block_size;
    contorchionist::core::ap_torchts::create_circular_buffer(
        x->in_buffers, x->out_buffers, x->in_ch, x->out_ch, x->buffer_size);
    x->in_blocks.resize(x->in_ch);
    for (auto& block : x->in_blocks) {
        block.resize(x->buffer_size);
    }
    x->in_ptrs.resize(x->in_ch);
    if (x->verbose) {
        post("torch.ts~: Buffer size set to %d samples", x->buffer_size);
    }
}
```

---

## 3. Mecanismo de Processamento de Sinal e Buffers Circulares

### 3.1. Descompasso entre Descarte na Entrada (`block_size`) e Escrita na Saída (`buffer_size`)
* **Arquivos:** `wrappers/puredata/src/torch.ts~.cpp` (linhas 259–265) e `core/src/core_ap_torchts.cpp` (linhas 53, 89–101)
* **Problema:**
  Na rotina `torch_ts_tilde_perform`:
  ```cpp
  x->in_buffers[ch]->peek(x->in_blocks[ch].data(), x->buffer_size);
  x->in_ptrs[ch] = x->in_blocks[ch].data();
  x->in_buffers[ch]->discard(x->block_size);
  ```
  Quando `buffer_size > block_size` (por exemplo, `buffer_size = 2048` e `block_size = 64`):
  1. Ao descartar apenas `block_size` (64) amostras da entrada, no **próximo ciclo DSP** (após a chegada de mais 64 amostras), `in_buffers[ch]->getSamplesAvailable()` será novamente `2048` (`>= buffer_size`).
  2. Porém, `TorchTSProcessorSignal` divide as `2048` amostras em `num_batches = 2048 / model_buffer_size` blocos contíguos e independentes e escreve **`2048` amostras de uma vez** em `out_buffers[ch]` via `write_overwrite(..., buffer_size)`.
  3. Como `out_buffers[ch]` consome apenas `64` amostras por ciclo DSP (`try_read(..., block_size)`), disparar o modelo novamente no ciclo seguinte (ou assim que a thread terminar) **sobrescreve** as amostras de saída que ainda não foram tocadas, causando distorção temporal severa e trabalho computacional redundante.

**Correção Recomendada:**
* **Para processamento em lote (Batching / Block-by-Block sem overlap):**
  Consumir `x->buffer_size` amostras inteiras do buffer de entrada ao disparar o modelo (`x->in_buffers[ch]->try_read(x->in_blocks[ch].data(), x->buffer_size);` ou `discard(x->buffer_size)`). Assim, o modelo processa `buffer_size` amostras a cada `buffer_size / block_size` ciclos DSP, mantendo a taxa de produção exatamente igual à taxa de consumo (`1:1`).
* **Para processamento em Janela Deslizante (Sliding Window com `hop_size = block_size`):**
  Se um modelo específico exigir contexto passado (`buffer_size`) para prever apenas o próximo bloco (`block_size`), então `process_output_tensor` deve escrever apenas as últimas `block_size` amostras no buffer de saída (e não `buffer_size`), sem usar o batching que fatia o contexto em lotes independentes.

---

### 3.2. Capacidade dos Buffers Circulares no Modo Assíncrono
* **Arquivo:** `core/src/core_ap_torchts.cpp` (linhas 316–339)
* **Problema:** `create_circular_buffer` aloca `CircularBuffer<float>` com capacidade exata de `buffer_size`. No modo assíncrono (`-async`), enquanto a thread de inferência está processando o lote atual de `buffer_size` amostras, novos blocos de entrada continuam chegando a cada ciclo DSP. Se a capacidade for de apenas `1x buffer_size`, e se a leitura de saída e escrita da thread ocorrerem de forma assíncrona, qualquer variação de tempo (*jitter*) causa *underrun* ou sobrescrita de amostras não lidas.
* **Correção Recomendada:** Alocar os buffers circulares com capacidade de pelo menos `2 * buffer_size` (double-buffering space) quando em modo assíncrono, mantendo o limiar de disparo da inferência em `getSamplesAvailable() >= x->buffer_size`.

---

## 4. Uso de CPU e GPU (CUDA / MPS)

### 4.1. Proteção de Concorrência (`model_mutex`) ao Trocar Dispositivo ou Parâmetros
* **Arquivo:** `wrappers/puredata/src/torch.ts~.cpp` (linhas 80–103 e 107–161)
* **Problema:**
  * Em `torch_ts_tilde_device`, `x->device` é alterado e `x->model->to(x->device)` é chamado na thread de mensagens do Pd sem adquirir `x->model_mutex`. Se a thread de áudio ou a thread assíncrona estiver no meio de `TorchTSProcessorSignal`, os tensores e pesos do modelo podem estar em dispositivos diferentes ou sendo movidos durante a execução do grafo, causando falha fatal no LibTorch.
  * Em `torch_ts_tilde_set_parameter`, `x->model->get_method(method_name)(inputs)` também é executado sem `x->model_mutex`.

**Correção Recomendada:**
```cpp
// Em torch_ts_tilde_device:
{
    std::lock_guard<std::mutex> lock(x->model_mutex);
    pd_parse_and_set_torch_device(
        (t_object*)x, x->device, dev, x->verbose, "torch.ts~", true
    );
    if (x->model) {
        x->model->to(x->device);
    }
}

// Em torch_ts_tilde_set_parameter:
{
    std::lock_guard<std::mutex> lock(x->model_mutex);
    if (x->device.type() != torch::kCPU) t = t.to(x->device);
    inputs.push_back(t);
    x->model->get_method(method_name)(inputs);
}
```

---

### 4.2. Otimização de Transferência de Tensores (CPU/GPU) e Contiguidade
* **Arquivo:** `core/src/core_ap_torchts.cpp` (linhas 27, 37, 93–108)
* **Problema:**
  1. Na criação do tensor de entrada (`torch::from_blob(...).clone()` seguido de `t = t.to(device)`):
     * Quando `device` é GPU (`CUDA` ou `MPS`), `.clone()` aloca e copia o buffer na RAM da CPU, e `.to(device)` faz uma segunda cópia da RAM para a GPU.
     * Quando `device` é CPU, `in_buffers[ch]` já aponta para `in_blocks[ch]` (que não é modificado durante a chamada de `TorchTSProcessorSignal`), tornando `.clone()` desnecessário.
  2. Na saída (`process_output_tensor`, linha 27): `tensor = tensor.to(torch::kCPU);` seguido de `tensor.view(...)` e `tensor.data_ptr<float>()`. Se o modelo retornar um tensor não-contíguo na memória, `.view(...)` lançará exceção ou `data_ptr<float>()` lerá dados incorretos.

**Correção Recomendada:**
```cpp
// Na entrada (TorchTSProcessorSignal):
auto t = torch::from_blob(const_cast<float*>(in_buffers[ch]), {(int64_t)buffer_size}, torch::kFloat32);
t = t.reshape({num_batches, 1, (int64_t)model_buffer_size});
// Se device != CPU, .to(device) já cria uma cópia própria na GPU
t = (device.type() == torch::kCPU) ? t.clone() : t.to(device);

// Na saída (process_output_tensor):
tensor = tensor.to(torch::kCPU).contiguous();
```

---

## 5. Batching e Validação de Tamanhos de Buffer

### 5.1. Bug em `suggest_optimal_buffer_size` Limitando `buffer_size` a `model_buffer_size`
* **Arquivo:** `core/src/core_ap_torchts.cpp` (linhas 188–213 e 299–306)
* **Problema:**
  Em `validate_buffer_size`, `model_buffer_size` é validado como o **tamanho mínimo** exigido pelo modelo (`requested_size < model_buffer_size` gera erro), enquanto `max_buffer_size` é o **limite máximo** (`requested_size > max_buffer_size`).
  Contudo, `suggest_optimal_buffer_size` trata `model_buffer_size` como **limite máximo**:
  ```cpp
  // Check model limit
  if (model_buffer_size > 0 && power_of_2 > model_buffer_size) {
      // Fallback to largest valid multiple
      power_of_2 = (model_buffer_size / block_size) * block_size;
  ```
  Isso impede que `suggest_optimal_buffer_size` sugira tamanhos maiores que `model_buffer_size` (necessários para ativar o *batching*!).

**Correção Recomendada:**
Alterar a assinatura e a verificação de `suggest_optimal_buffer_size` para respeitar `model_buffer_size` como mínimo e `max_buffer_size` como máximo:
```cpp
int suggest_optimal_buffer_size(int requested_size, int block_size, int model_buffer_size, int max_buffer_size) {
    int min_size = std::max({requested_size, block_size, model_buffer_size});
    int multiple_size = ((min_size / block_size) + (min_size % block_size ? 1 : 0)) * block_size;

    int power_of_2 = 1;
    while (power_of_2 < multiple_size) {
        power_of_2 <<= 1;
    }
    while (power_of_2 % block_size != 0) {
        power_of_2 <<= 1;
    }
    if (max_buffer_size > 0 && power_of_2 > max_buffer_size) {
        power_of_2 = (max_buffer_size / block_size) * block_size;
        if (power_of_2 < block_size) {
            power_of_2 = block_size;
        }
    }
    return power_of_2;
}
```

---

### 5.2. Padronização de Shape 3D `[batch, 1, samples]` e Controle de Batching
* **Arquivo:** `core/src/core_ap_torchts.cpp` (linhas 89–110)
* **Problema:**
  1. Quando `can_batch == true` (`buffer_size > model_buffer_size`), o tensor enviado ao modelo tem 3 dimensões: `[num_batches, 1, model_buffer_size]`.
  2. Quando `can_batch == false` (ex.: `buffer_size == model_buffer_size`), o tensor enviado tem 2 dimensões: `[1, buffer_size]`.
  3. Modelos de áudio TorchScript exportados no padrão `[batch, channels, time]` (como RAVE / `nn~`) falham se receberem um tensor 2D quando `buffer_size == model_buffer_size`.
  4. Além disso, para modelos **causais com cache interno ou recorrentes**, empilhar janelas temporais consecutivas na dimensão `batch` (`[num_batches, 1, model_buffer_size]`) processa passos de tempo em paralelo como sequências independentes, quebrando a continuidade de fase/estado entre os sub-blocos.

**Correção Recomendada:**
* Para modelos que possuem `model_buffer_size > 0` ou atributo de shape 3D (`<method>_input_shape` com 3 dimensões), manter sempre 3 dimensões `[1, 1, buffer_size]` mesmo quando `num_batches == 1`, ou iterar sequencialmente pelos `num_batches` sub-blocos dentro da mesma chamada de thread quando o modelo for stateful/recorrente (em vez de empilhar no eixo `batch`).

---

## 6. Multithreading e Tempo Real (`torch.ts~.cpp`)

### 6.1. Substituição de `std::thread` + `detach()` por Worker Thread Persistente
* **Arquivo:** `wrappers/puredata/src/torch.ts~.cpp` (linhas 267–298)
* **Problema:**
  A cada janela pronta no modo assíncrono, a rotina de áudio `torch_ts_tilde_perform` executa:
  ```cpp
  x->model_thread = std::make_unique<std::thread>([x, in_blocks = x->in_blocks]() mutable { ... });
  x->model_thread->detach();
  ```
  Isso faz alocação de memória no heap (`make_unique` e cópia de todos os `std::vector<float>` de `in_blocks`) e cria uma nova thread do sistema operacional (`pthread_create`) dentro do callback de tempo real do áudio.

**Correção Recomendada (Arquitetura com Worker Thread Persistente):**
1. Iniciar uma única thread trabalhadora quando o modo assíncrono for ativado (em `torch_ts_tilde_New`), que aguarda em uma `std::condition_variable`.
2. Na rotina `torch_ts_tilde_perform`, quando houver amostras suficientes e a worker thread estiver livre (`!x->worker_busy.load(std::memory_order_acquire)`):
   * Copiar as amostras para o buffer pré-alocado da worker thread (`x->worker_in_blocks`).
   * Definir `x->worker_busy.store(true, std::memory_order_release)` e sinalizar `x->worker_cv.notify_one()`.
3. No destrutor (`torch_ts_tilde_Free`), sinalizar uma flag `x->stop_worker.store(true)`, chamar `x->worker_cv.notify_all()` e fazer `if (x->model_thread && x->model_thread->joinable()) x->model_thread->join();` — eliminando o `detach()` e o loop de `sleep_for(1ms)`.

---

### 6.2. Remoção de `pd_error` de Dentro da Thread Secundária
* **Arquivo:** `wrappers/puredata/src/torch.ts~.cpp` (linha 291)
* **Problema:** `pd_error(x, ...)` é chamado diretamente dentro da lambda executada pela `std::thread` secundária. A API do Pure Data não é thread-safe para chamadas fora da thread principal/scheduler do Pd.
* **Correção Recomendada:** Armazenar a mensagem de erro em uma variável protegida (ou flag atômica) e reportá-la na thread principal usando um `t_clock` (`clock_new` / `clock_delay(x->error_clock, 0)`).

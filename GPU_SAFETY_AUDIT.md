# Auditoria de Segurança GPU - Pure Data Externals

## Data: 21 de Novembro de 2025

## Objetivo
Verificar se todos os objetos Pure Data que usam `memcpy` ou `data_ptr()` garantem que tensores estejam em CPU antes de acessar a memória, evitando crashes com MPS/CUDA.

## Resultado da Auditoria

### ✅ Objetos CORRETOS (sem problemas)

1. **`torch.rfft~.cpp`**
   - Linha 82: `.cpu()` antes de processar complex_spectrum
   - Linhas 86-87: `.cpu().contiguous()` em ambos os componentes
   - **STATUS: SEGURO**

2. **`torch.irfft~.cpp`**
   - Linha 84: `time_signal.cpu().contiguous()`
   - **STATUS: SEGURO**

3. **`torch.linear~.cpp`**
   - Linha 51: `output = output.to(torch::kCPU);`
   - **STATUS: SEGURO**

4. **`torch.rmsoverlap~.cpp`**
   - Linha 253: `win_func_tensor.contiguous().cpu()`
   - Linha 280: `sum_tensor.contiguous().cpu()`
   - **STATUS: SEGURO**

### ❌ Objeto CORRIGIDO

5. **`torch.amb.spectrails~.cpp`**
   - **PROBLEMA ORIGINAL:** Tentava `memcpy` direto de tensores GPU
   - **CORREÇÃO APLICADA:** Adicionado `.to(torch::kCPU)` antes de todos os `memcpy`
   - Linhas 166-167: `w_outputs[0].to(torch::kCPU).contiguous()`
   - Linhas 189-190: `outputs[0].to(torch::kCPU).contiguous()`
   - **STATUS: CORRIGIDO**

## Padrão Recomendado

Para qualquer objeto que precise copiar dados de tensores PyTorch para buffers Pure Data:

```cpp
// ✅ CORRETO
auto tensor_cpu = tensor.to(torch::kCPU).contiguous();
std::memcpy(buffer, tensor_cpu.data_ptr<float>(), size);

// ❌ INCORRETO (crash com MPS/CUDA)
std::memcpy(buffer, tensor.data_ptr<float>(), size);
```

## Conclusão

Todos os objetos Pure Data que usam PyTorch estão agora seguros para uso com CPU, MPS e CUDA. O único objeto que tinha o problema (`torch.amb.spectrails~`) foi identificado e corrigido.

## Arquivos Auditados

- `wrappers/puredata/src/torch.rfft~.cpp`
- `wrappers/puredata/src/torch.irfft~.cpp`
- `wrappers/puredata/src/torch.linear~.cpp`
- `wrappers/puredata/src/torch.rmsoverlap~.cpp`
- `wrappers/puredata/src/torch.amb.spectrails~.cpp`

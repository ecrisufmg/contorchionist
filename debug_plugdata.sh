#!/bin/bash
# Script para debugar torch.amb.spectrails~ no PlugData

echo "=== PlugData Debug Script ==="
echo ""
echo "Iniciando PlugData com lldb..."
echo "Comandos úteis no lldb:"
echo "  - 'run' ou 'r' para iniciar"
echo "  - 'bt' para ver backtrace após crash"
echo "  - 'frame variable' para ver variáveis locais"
echo "  - 'continue' ou 'c' para continuar após breakpoint"
echo "  - 'quit' para sair"
echo ""
echo "Breakpoint será colocado em torch_amb_spectrails_tilde_new"
echo ""

lldb /Applications/plugdata.app/Contents/MacOS/plugdata -- \
  -b "torch_amb_spectrails_tilde_new" \
  -b "torch_amb_spectrails_tilde_dsp" \
  -b "torch_amb_spectrails_tilde_perform"

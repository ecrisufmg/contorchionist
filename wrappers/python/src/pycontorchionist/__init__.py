"""
pycontorchionist - Python bindings for Contorchionist Audio Processing Library

This package provides high-performance audio processing capabilities with LibTorch backend.
"""

# Import torch first to load LibTorch libraries into memory
# This is required for the compiled extension to find the shared libraries
import torch
import torchaudio

# Import all symbols from the compiled module
from .pycontorchionist import *

__version__ = "0.1.0"
__author__ = "Contorchionist Team"

# List all exported names (these are already imported via *)
__all__ = [
    "MelSpectrogramProcessor",
    "MelNormMode",
    "SpectrumDataFormat",
    "MelFormulaType", 
    "WindowType",
    "NormalizationType",
    "mel_norm_mode_to_string",
    "string_to_mel_norm_mode",
    "spectrum_data_format_to_string",
    "string_to_spectrum_data_format",
]

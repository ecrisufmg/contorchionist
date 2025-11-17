"""
Basic tests for pycontorchionist library
"""

import unittest
import numpy as np
import pycontorchionist as cc


class TestMelSpectrogramProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.processor = cc.MelSpectrogramProcessor()
        
    def test_processor_creation(self):
        """Test that MelSpectrogramProcessor can be created."""
        processor = cc.MelSpectrogramProcessor()
        self.assertIsInstance(processor, cc.MelSpectrogramProcessor)
        
    def test_enum_conversion(self):
        """Test enum conversion functions."""
        # Test MelNormMode conversion
        mode = cc.string_to_mel_norm_mode("energy")
        self.assertEqual(mode, cc.MelNormMode.ENERGY_POWER)
        
        mode_str = cc.mel_norm_mode_to_string(cc.MelNormMode.ENERGY_POWER)
        self.assertEqual(mode_str, "energy")
        
    def test_basic_parameters(self):
        """Test basic parameter getters and setters."""
        self.processor.set_sample_rate(44100.0)
        self.assertEqual(self.processor.get_sample_rate(), 44100.0)
        
        self.processor.set_n_mels(64)
        self.assertEqual(self.processor.get_n_mels(), 64)
        
        self.processor.set_n_fft(1024)
        self.assertEqual(self.processor.get_n_fft(), 1024)
        
        self.processor.set_hop_length(256)
        self.assertEqual(self.processor.get_hop_length(), 256)


if __name__ == '__main__':
    unittest.main()

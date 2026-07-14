#!/usr/bin/env python3
"""
Test suite for INT4 PTQ quantization script.
Tests syntax, imports, classes, functions, and integration.
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

class TestInt4PTQ(unittest.TestCase):
    """Test cases for INT4 PTQ implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.parent_dir = os.path.dirname(self.test_dir)
        self.root_dir = os.path.dirname(self.parent_dir)

        # Mock dependencies that might not be available
        self.mock_imports = {
            'brevitas': MagicMock(),
            'brevitas.nn': MagicMock(),
            'brevitas.quant': MagicMock(),
        }

    def test_script_syntax(self):
        """Test that the script has valid Python syntax."""
        script_path = os.path.join(self.test_dir, 'quantize_4bit_ptq.py')

        # Compile the script to check syntax
        with open(script_path, 'r') as f:
            code = f.read()

        try:
            compile(code, script_path, 'exec')
        except SyntaxError as e:
            self.fail(f"Syntax error in {script_path}: {e}")

    @patch.dict('sys.modules', {'brevitas': MagicMock(), 'brevitas.nn': MagicMock()})
    def test_imports(self):
        """Test that all imports work correctly."""
        try:
            # Mock brevitas if not available
            import sys
            if 'brevitas' not in sys.modules:
                sys.modules['brevitas'] = MagicMock()
                sys.modules['brevitas.nn'] = MagicMock()
                sys.modules['brevitas.quant'] = MagicMock()

            from quantize_4bit_ptq import (
                Int4QuantizationConfig,
                Int4PTQQuantizer,
                ModelEvaluator,
                QuantizedModelWrapper,
                load_checkpoint,
                save_quantized_model,
                create_calibration_loader,
                main
            )
        except ImportError as e:
            self.fail(f"Import error: {e}")

    def test_config_class(self):
        """Test Int4QuantizationConfig class."""
        try:
            from quantize_4bit_ptq import Int4QuantizationConfig

            config = Int4QuantizationConfig()
            self.assertEqual(config.weight_bits, 4)
            self.assertEqual(config.activation_bits, 4)
            self.assertEqual(config.weight_quant_min, -8)
            self.assertEqual(config.weight_quant_max, 7)
            self.assertEqual(config.activation_quant_min, -8)
            self.assertEqual(config.activation_quant_max, 7)
            self.assertTrue(config.use_brevitas)
        except Exception as e:
            self.fail(f"Int4QuantizationConfig test failed: {e}")

    def test_quantizer_class(self):
        """Test Int4PTQQuantizer class initialization."""
        try:
            from quantize_4bit_ptq import Int4PTQQuantizer, Int4QuantizationConfig

            config = Int4QuantizationConfig()
            quantizer = Int4PTQQuantizer(config)

            self.assertIsInstance(quantizer, Int4PTQQuantizer)
            self.assertEqual(quantizer.config.weight_bits, 4)
        except Exception as e:
            self.fail(f"Int4PTQQuantizer test failed: {e}")

    def test_model_evaluator_class(self):
        """Test ModelEvaluator class."""
        try:
            from quantize_4bit_ptq import ModelEvaluator

            evaluator = ModelEvaluator()
            self.assertIsInstance(evaluator, ModelEvaluator)
        except Exception as e:
            self.fail(f"ModelEvaluator test failed: {e}")

    def test_quantized_model_wrapper(self):
        """Test QuantizedModelWrapper class."""
        try:
            from quantize_4bit_ptq import QuantizedModelWrapper

            # Create a simple model for testing
            model = nn.Sequential(nn.Conv2d(3, 64, 3), nn.ReLU())
            wrapper = QuantizedModelWrapper(model)

            self.assertIsInstance(wrapper, QuantizedModelWrapper)
            self.assertEqual(wrapper.model, model)
        except Exception as e:
            self.fail(f"QuantizedModelWrapper test failed: {e}")

    def test_load_checkpoint_function(self):
        """Test load_checkpoint function."""
        try:
            from quantize_4bit_ptq import load_checkpoint

            # Test with non-existent file (should raise error)
            with self.assertRaises(FileNotFoundError):
                load_checkpoint("nonexistent.pth")
        except Exception as e:
            self.fail(f"load_checkpoint test failed: {e}")

    def test_save_quantized_model_function(self):
        """Test save_quantized_model function."""
        try:
            from quantize_4bit_ptq import save_quantized_model

            # Create a simple model and config for testing
            model = nn.Sequential(nn.Conv2d(3, 64, 3), nn.ReLU())
            config = {"test": "config"}

            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = os.path.join(temp_dir, "test_model.pt")
                config_path = os.path.join(temp_dir, "test_config.json")

                save_quantized_model(model, config, model_path, config_path)

                # Check files were created
                self.assertTrue(os.path.exists(model_path))
                self.assertTrue(os.path.exists(config_path))

                # Check config content
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                self.assertEqual(saved_config, config)

        except Exception as e:
            self.fail(f"save_quantized_model test failed: {e}")

    def test_create_calibration_loader_function(self):
        """Test create_calibration_loader function."""
        try:
            from quantize_4bit_ptq import create_calibration_loader

            # Test with invalid dataset (should handle gracefully)
            loader = create_calibration_loader("invalid_dataset", batch_size=1, num_samples=10)
            self.assertIsNone(loader)
        except Exception as e:
            self.fail(f"create_calibration_loader test failed: {e}")

    def test_main_function_exists(self):
        """Test that main function exists."""
        try:
            from quantize_4bit_ptq import main
            self.assertTrue(callable(main))
        except Exception as e:
            self.fail(f"main function test failed: {e}")

    def test_paths_and_imports(self):
        """Test that all relative path imports work."""
        try:
            # Test imports from parent directories
            sys.path.insert(0, self.root_dir)

            # These should work if paths are correct
            from models.loc_lic import LoC_LIC
            from utils.config import load_config
            from utils.metrics import calculate_psnr, calculate_msssim

            self.assertTrue(callable(LoC_LIC))
            self.assertTrue(callable(load_config))
            self.assertTrue(callable(calculate_psnr))
            self.assertTrue(callable(calculate_msssim))

        except ImportError as e:
            self.fail(f"Path/import test failed: {e}")
        finally:
            if self.root_dir in sys.path:
                sys.path.remove(self.root_dir)

    def test_brevitas_fallback(self):
        """Test that the script handles missing brevitas gracefully."""
        try:
            # Temporarily remove brevitas from modules
            original_brevitas = sys.modules.get('brevitas')
            if 'brevitas' in sys.modules:
                del sys.modules['brevitas']
            if 'brevitas.nn' in sys.modules:
                del sys.modules['brevitas.nn']
            if 'brevitas.quant' in sys.modules:
                del sys.modules['brevitas.quant']

            # Import should still work
            from quantize_4bit_ptq import Int4QuantizationConfig
            config = Int4QuantizationConfig()
            # Should default to False if brevitas not available
            self.assertFalse(config.use_brevitas)

        except Exception as e:
            self.fail(f"Brevitas fallback test failed: {e}")
        finally:
            # Restore brevitas if it was there
            if original_brevitas is not None:
                sys.modules['brevitas'] = original_brevitas

    def test_config_validation(self):
        """Test configuration validation."""
        try:
            from quantize_4bit_ptq import Int4QuantizationConfig

            # Valid config
            config = Int4QuantizationConfig(
                weight_bits=4,
                activation_bits=4,
                weight_quant_min=-8,
                weight_quant_max=7
            )
            self.assertEqual(config.weight_bits, 4)

            # Test with different values
            config2 = Int4QuantizationConfig(weight_bits=8)  # Should still work
            self.assertEqual(config2.weight_bits, 8)

        except Exception as e:
            self.fail(f"Config validation test failed: {e}")

    def test_file_structure(self):
        """Test that all expected files exist."""
        expected_files = [
            'quantize_4bit_ptq.py',
            'README.md'
        ]

        for filename in expected_files:
            filepath = os.path.join(self.test_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"Missing file: {filename}")

    def test_script_executable(self):
        """Test that the script can be executed (basic smoke test)."""
        import subprocess

        script_path = os.path.join(self.test_dir, 'quantize_4bit_ptq.py')

        # Test --help flag (should not require actual model files)
        try:
            result = subprocess.run(
                [sys.executable, script_path, '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Should exit successfully with help
            self.assertEqual(result.returncode, 0)
            self.assertIn('usage:', result.stdout.lower())
        except subprocess.TimeoutExpired:
            self.fail("Script --help timed out")
        except Exception as e:
            self.fail(f"Script execution test failed: {e}")


if __name__ == '__main__':
    # Set up test environment
    print("Running INT4 PTQ tests...")
    print(f"Test directory: {os.path.dirname(os.path.abspath(__file__))}")

    # Run tests
    unittest.main(verbosity=2)
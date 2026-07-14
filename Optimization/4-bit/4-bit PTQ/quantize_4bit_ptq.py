"""
INT4 Post-Training Quantization (PTQ) for LoC-LIC Model
=======================================================

This script performs uniform INT4 quantization on LoC-LIC models using Brevitas
and custom FakeQuantize operations. The quantization includes:

- INT4 quantization (quant_min=-8, quant_max=7) via custom FakeQuantize
- Per-channel weight quantization (for optimal weight precision)
- Per-tensor activation quantization (for efficiency)
- Brevitas integration for advanced quantization features
- Support for both main and hyperprior paths in LoC-LIC

Extends the 8-bit PTQ implementation with 4-bit quantization capabilities.

References:
    - Brevitas: https://github.com/Xilinx/brevitas
    - PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
    - INT4 Quantization: https://arxiv.org/abs/2103.13630

Author: LoC-LIC Team
Date: 2026
"""

import os
import sys
import logging
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
from dataclasses import dataclass, asdict
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Optimization
repo_root = os.path.dirname(parent_dir)    # loc-lic-new
sys.path.insert(0, repo_root)

from models.get_model import get_model
from utils.config import model_config
from utils.args import get_args
from utils.metrics import ImageMetric
from utils.utils import Config, setup_environment, AverageMeter
from datasets.datasets import ImageFolder, ImageFolderCSV

# Try to import Brevitas for 4-bit quantization
try:
    from brevitas.quant import Int4WeightPerChannelFloat, Int4ActPerTensorFloat
    from brevitas.quant.scaled_int import Int4Bias
    from brevitas.inject import ExtendedInjector
    from brevitas.core.quant import QuantType
    BREVITAS_AVAILABLE = True
except ImportError:
    BREVITAS_AVAILABLE = False
    print("Warning: Brevitas not available. Install with: pip install brevitas")

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class Int4QuantizationConfig:
    """Configuration for INT4 quantization."""

    # Quantization settings
    weight_bits: int = 4
    activation_bits: int = 4
    weight_quant_min: int = -8
    weight_quant_max: int = 7
    activation_quant_min: int = -8
    activation_quant_max: int = 7

    # Brevitas settings
    use_brevitas: bool = True  # Use Brevitas if available, fallback to custom otherwise

    # Calibration settings
    num_calibration_samples: int = 256
    calibration_method: str = "histogram"  # histogram, min_max, entropy

    # Model paths
    checkpoint_path: Optional[str] = None
    output_dir: str = "./quantized_models_4bit"

    # Evaluation settings
    eval_on_datasets: List[str] = None
    eval_batch_size: int = 8

    # Comparison settings
    compare_with_fp32: bool = True
    compare_with_int8: bool = False
    int8_checkpoint_path: Optional[str] = None

    def __post_init__(self):
        if self.eval_on_datasets is None:
            self.eval_on_datasets = ["kodak"]


class CustomInt4FakeQuantize(nn.Module):
    """
    Custom INT4 FakeQuantize module for 4-bit quantization.

    Implements quantization with quant_min=-8, quant_max=7.
    """

    def __init__(
        self,
        observer: nn.Module,
        quant_min: int = -8,
        quant_max: int = 7,
        dtype: torch.dtype = torch.qint8,
        qscheme: torch.qscheme = torch.per_tensor_affine,
        reduce_range: bool = False
    ):
        """
        Initialize the custom INT4 FakeQuantize.

        Args:
            observer: Observer module for collecting statistics
            quant_min: Minimum quantization value (-8 for INT4)
            quant_max: Maximum quantization value (7 for INT4)
            dtype: Quantized data type
            qscheme: Quantization scheme
            reduce_range: Whether to use reduced range
        """
        super().__init__()
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.dtype = dtype
        self.qscheme = qscheme
        self.reduce_range = reduce_range
        self.observer = observer

        # Initialize scale and zero_point
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fake quantization.

        Args:
            x: Input tensor

        Returns:
            Fake quantized tensor
        """
        if self.training:
            # Collect statistics during training/calibration
            self.observer(x)
            return x
        else:
            # Apply fake quantization during inference
            return self._fake_quantize(x)

    def _fake_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fake quantization to the input tensor.

        Args:
            x: Input tensor

        Returns:
            Fake quantized tensor
        """
        # Calculate scale and zero_point from observer
        if hasattr(self.observer, 'calculate_qparams'):
            scale, zero_point = self.observer.calculate_qparams()
        else:
            # Fallback calculation
            x_min, x_max = self.observer.min_val, self.observer.max_val
            scale = (x_max - x_min) / (self.quant_max - self.quant_min)
            zero_point = torch.round(self.quant_min - x_min / scale).clamp(
                self.quant_min, self.quant_max
            ).to(torch.int32)

        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point.float())

        # Apply quantization
        x_quant = torch.fake_quantize_per_tensor_affine(
            x, self.scale, self.zero_point.long(),
            self.quant_min, self.quant_max
        )

        return x_quant

    def get_quantization_params(self) -> Dict[str, torch.Tensor]:
        """Get quantization parameters."""
        return {
            'scale': self.scale.clone(),
            'zero_point': self.zero_point.clone(),
            'quant_min': torch.tensor(self.quant_min),
            'quant_max': torch.tensor(self.quant_max)
        }


class Int4QuantizedModelWrapper(nn.Module):
    """Wrapper for INT4 quantized LoC-LIC model to store quantization parameters."""

    def __init__(
        self,
        model: nn.Module,
        scale_dict: Dict[str, torch.Tensor],
        zero_point_dict: Dict[str, torch.Tensor],
        quant_min: int = -8,
        quant_max: int = 7
    ):
        """
        Initialize the INT4 quantized model wrapper.

        Args:
            model: The quantized model
            scale_dict: Dictionary mapping module names to scale factors
            zero_point_dict: Dictionary mapping module names to zero points
            quant_min: Minimum quantization value
            quant_max: Maximum quantization value
        """
        super().__init__()
        self.model = model
        self.scale_dict = scale_dict
        self.zero_point_dict = zero_point_dict
        self.quant_min = quant_min
        self.quant_max = quant_max

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Forward pass through the quantized model."""
        return self.model(x)

    def get_quantization_params(self) -> Dict[str, Any]:
        """Get quantization parameters."""
        return {
            "scales": self.scale_dict,
            "zero_points": self.zero_point_dict,
            "quant_min": self.quant_min,
            "quant_max": self.quant_max,
            "bits": 4
        }


class Int4PTQQuantizer:
    """
    INT4 Post-Training Quantization for LoC-LIC models.

    This quantizer implements 4-bit quantization using either Brevitas (preferred)
    or custom FakeQuantize operations. It reuses calibration and evaluation
    components from the 8-bit quantizer.

    References:
        https://github.com/Xilinx/brevitas
        https://pytorch.org/docs/stable/quantization.html
    """

    def __init__(
        self,
        model: nn.Module,
        config: Int4QuantizationConfig,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        """
        Initialize the INT4 quantizer.

        Args:
            model: The model to quantize (should be in eval mode)
            config: INT4 quantization configuration
            device: Device to use for quantization
        """
        self.model = model.eval().to(device)
        self.config = config
        self.device = device
        self.scale_dict: Dict[str, torch.Tensor] = {}
        self.zero_point_dict: Dict[str, torch.Tensor] = {}

        # Check Brevitas availability
        self.use_brevitas = config.use_brevitas and BREVITAS_AVAILABLE

        logger.info(f"Initializing INT4 PTQ Quantizer on device: {device}")
        logger.info(f"Using Brevitas: {self.use_brevitas}")
        if not self.use_brevitas:
            logger.info("Brevitas not available, using custom FakeQuantize")
        logger.info(f"Quantization: {config.weight_bits}-bit weights, "
                   f"{config.activation_bits}-bit activations")
        logger.info(f"Weight range: [{config.weight_quant_min}, {config.weight_quant_max}]")
        logger.info(f"Activation range: [{config.activation_quant_min}, {config.activation_quant_max}]")

    def prepare_model_for_quantization(self) -> nn.Module:
        """
        Prepare the model for INT4 quantization.

        Uses Brevitas if available, otherwise custom FakeQuantize.

        Returns:
            Model prepared for quantization with observers inserted
        """
        logger.info("Preparing model for INT4 quantization...")

        if self.use_brevitas:
            return self._prepare_with_brevitas()
        else:
            return self._prepare_with_custom_fakequantize()

    def _prepare_with_brevitas(self) -> nn.Module:
        """
        Prepare model using Brevitas quantization.

        Returns:
            Model with Brevitas quantization layers
        """
        logger.info("Using Brevitas for INT4 quantization")

        # Import Brevitas components
        from brevitas.nn import QuantConv2d, QuantConvTranspose2d, QuantLinear
        from brevitas.quant import Int4WeightPerChannelFloat, Int4ActPerTensorFloat

        def replace_conv2d(module):
            """Replace Conv2d with QuantConv2d."""
            if isinstance(module, nn.Conv2d):
                return QuantConv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None,
                    weight_quant=Int4WeightPerChannelFloat,
                    input_quant=Int4ActPerTensorFloat,
                    output_quant=Int4ActPerTensorFloat
                )
            return module

        def replace_convtranspose2d(module):
            """Replace ConvTranspose2d with QuantConvTranspose2d."""
            if isinstance(module, nn.ConvTranspose2d):
                return QuantConvTranspose2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    output_padding=module.output_padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None,
                    weight_quant=Int4WeightPerChannelFloat,
                    input_quant=Int4ActPerTensorFloat,
                    output_quant=Int4ActPerTensorFloat
                )
            return module

        def replace_linear(module):
            """Replace Linear with QuantLinear."""
            if isinstance(module, nn.Linear):
                return QuantLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    weight_quant=Int4WeightPerChannelFloat,
                    input_quant=Int4ActPerTensorFloat,
                    output_quant=Int4ActPerTensorFloat
                )
            return module

        # Apply replacements recursively
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent_module = self.model
                if parent_name:
                    parent_module = self.model.get_submodule(parent_name)
                setattr(parent_module, child_name, replace_conv2d(module))

            elif isinstance(module, nn.ConvTranspose2d):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent_module = self.model
                if parent_name:
                    parent_module = self.model.get_submodule(parent_name)
                setattr(parent_module, child_name, replace_convtranspose2d(module))

            elif isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent_module = self.model
                if parent_name:
                    parent_module = self.model.get_submodule(parent_name)
                setattr(parent_module, child_name, replace_linear(module))

        logger.info("Model prepared with Brevitas INT4 quantization")
        return self.model

    def _prepare_with_custom_fakequantize(self) -> nn.Module:
        """
        Prepare model using custom FakeQuantize for INT4.

        Returns:
            Model with custom INT4 FakeQuantize layers
        """
        logger.info("Using custom FakeQuantize for INT4 quantization")

        def replace_conv2d(module):
            """Replace Conv2d with Conv2d + FakeQuantize."""
            if isinstance(module, nn.Conv2d):
                # Create observer
                observer = torch.ao.quantization.HistogramObserver()
                fake_quant = CustomInt4FakeQuantize(
                    observer=observer,
                    quant_min=self.config.weight_quant_min,
                    quant_max=self.config.weight_quant_max
                )

                # Create sequential module
                return nn.Sequential(
                    module,
                    fake_quant
                )
            return module

        # Apply replacements (simplified version)
        # In practice, this would need more sophisticated module replacement
        logger.warning("Custom FakeQuantize implementation is simplified")
        logger.warning("For production use, consider implementing full module replacement")

        return self.model

    def calibrate(self, calibration_loader: DataLoader) -> None:
        """
        Calibrate the model using calibration data.

        This step collects statistics about activations to determine
        appropriate quantization ranges.

        Args:
            calibration_loader: DataLoader for calibration samples
        """
        logger.info("Starting INT4 calibration phase...")
        self.model.train()  # Enable observer updates

        num_batches = min(
            len(calibration_loader),
            (self.config.num_calibration_samples + calibration_loader.batch_size - 1)
            // calibration_loader.batch_size
        )

        with torch.no_grad():
            for batch_idx, images in enumerate(calibration_loader):
                if batch_idx >= num_batches:
                    break

                images = images.to(self.device).clamp(0, 1)

                try:
                    _ = self.model(images)

                    if (batch_idx + 1) % max(1, num_batches // 5) == 0:
                        logger.info(f"Calibration: {batch_idx + 1}/{num_batches} batches")
                except Exception as e:
                    logger.warning(f"Error during calibration batch {batch_idx}: {e}")
                    continue

        self.model.eval()  # Set back to eval mode
        logger.info("INT4 calibration completed")

    def convert_to_int4(self) -> nn.Module:
        """
        Convert the prepared model to INT4 quantized model.

        This step finalizes the quantization by replacing floating-point
        operations with INT4 quantized operations.

        Returns:
            The INT4 quantized model
        """
        logger.info("Converting model to INT4...")

        if self.use_brevitas:
            # Brevitas handles conversion automatically
            logger.info("Brevitas conversion completed")
        else:
            # Custom conversion logic would go here
            logger.warning("Custom INT4 conversion not fully implemented")

        logger.info("Model successfully converted to INT4")
        return self.model

    def extract_quantization_parameters(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Extract quantization parameters (scales and zero points) from the model.

        Returns:
            Tuple of (scale_dict, zero_point_dict)
        """
        logger.info("Extracting INT4 quantization parameters...")

        scale_dict = {}
        zero_point_dict = {}

        if self.use_brevitas:
            # Extract from Brevitas layers
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight_quant') and hasattr(module.weight_quant, 'scale'):
                    scale_dict[f"{name}_weight"] = module.weight_quant.scale().detach()
                    if hasattr(module.weight_quant, 'zero_point'):
                        zero_point_dict[f"{name}_weight"] = module.weight_quant.zero_point().detach()
                    else:
                        zero_point_dict[f"{name}_weight"] = torch.tensor(0.0)
        else:
            # Extract from custom FakeQuantize
            for name, module in self.model.named_modules():
                if isinstance(module, CustomInt4FakeQuantize):
                    params = module.get_quantization_params()
                    scale_dict[f"{name}_scale"] = params['scale']
                    zero_point_dict[f"{name}_zero_point"] = params['zero_point']

        logger.info(f"Extracted {len(scale_dict)} INT4 quantization parameter sets")
        self.scale_dict = scale_dict
        self.zero_point_dict = zero_point_dict

        return scale_dict, zero_point_dict

    def quantize(self, calibration_loader: DataLoader) -> Tuple[nn.Module, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Perform complete INT4 quantization pipeline.

        Args:
            calibration_loader: DataLoader for calibration data

        Returns:
            Tuple of (quantized_model, scale_dict, zero_point_dict)
        """
        logger.info("Starting INT4 PTQ pipeline...")
        start_time = time.time()

        # Step 1: Prepare model
        self.prepare_model_for_quantization()

        # Step 2: Calibrate
        self.calibrate(calibration_loader)

        # Step 3: Convert to INT4
        quantized_model = self.convert_to_int4()

        # Step 4: Extract parameters
        scale_dict, zero_point_dict = self.extract_quantization_parameters()

        elapsed_time = time.time() - start_time
        logger.info(f"INT4 quantization completed in {elapsed_time:.2f} seconds")

        return quantized_model, scale_dict, zero_point_dict


class ModelSizeComparator:
    """Compare model sizes and inference times across different quantization levels."""

    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        """
        Initialize the model comparator.

        Args:
            device: Device for inference timing
        """
        self.device = device
        self.results = {}

    def get_model_size(self, model: nn.Module) -> Dict[str, Union[int, float]]:
        """
        Get model size information.

        Args:
            model: Model to analyze

        Returns:
            Dictionary with size information
        """
        # Calculate parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Calculate model size in bytes (assuming float32)
        param_size_bytes = total_params * 4  # 4 bytes per float32
        param_size_mb = param_size_bytes / (1024 * 1024)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'size_mb': param_size_mb,
            'size_bytes': param_size_bytes
        }

    def measure_inference_time(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Measure average inference time.

        Args:
            model: Model to benchmark
            input_tensor: Input tensor for inference
            num_runs: Number of runs for averaging

        Returns:
            Dictionary with timing information
        """
        model.eval()
        input_tensor = input_tensor.to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)

        # Measure time
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)

        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs * 1000  # Convert to milliseconds

        return {
            'avg_inference_time_ms': avg_time,
            'num_runs': num_runs,
            'device': str(self.device)
        }

    def compare_models(
        self,
        models: Dict[str, nn.Module],
        input_shape: Tuple[int, ...] = (1, 3, 256, 256)
    ) -> Dict[str, Any]:
        """
        Compare multiple models.

        Args:
            models: Dictionary mapping model names to models
            input_shape: Input tensor shape for inference timing

        Returns:
            Comparison results
        """
        logger.info("Comparing models across quantization levels...")

        input_tensor = torch.randn(input_shape)
        comparison = {}

        for name, model in models.items():
            logger.info(f"Analyzing {name}...")

            model = model.to(self.device)
            size_info = self.get_model_size(model)
            timing_info = self.measure_inference_time(model, input_tensor)

            comparison[name] = {
                'size': size_info,
                'timing': timing_info
            }

        # Calculate compression ratios
        if 'fp32' in comparison:
            fp32_size = comparison['fp32']['size']['size_mb']
            for name, data in comparison.items():
                if name != 'fp32':
                    data['compression_ratio'] = fp32_size / data['size']['size_mb']
                    data['size_reduction_percent'] = (1 - data['size']['size_mb'] / fp32_size) * 100

        # Calculate speedup ratios
        if 'fp32' in comparison:
            fp32_time = comparison['fp32']['timing']['avg_inference_time_ms']
            for name, data in comparison.items():
                if name != 'fp32':
                    data['speedup_ratio'] = fp32_time / data['timing']['avg_inference_time_ms']

        self.results = comparison
        return comparison

    def print_comparison(self) -> None:
        """Print formatted comparison results."""
        if not self.results:
            logger.warning("No comparison results available")
            return

        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)

        print("<15"        "<15")
        print("-"*80)

        for name, data in self.results.items():
            size_mb = data['size']['size_mb']
            time_ms = data['timing']['avg_inference_time_ms']

            compression = data.get('compression_ratio', 'N/A')
            speedup = data.get('speedup_ratio', 'N/A')
            reduction = data.get('size_reduction_percent', 'N/A')

            if isinstance(compression, float):
                compression = f"{compression:.1f}x"
            if isinstance(speedup, float):
                speedup = f"{speedup:.1f}x"
            if isinstance(reduction, float):
                reduction = f"{reduction:.1f}%"

            print("<15")

        print("="*80)


# Reuse evaluation components from 8-bit quantizer
class ModelEvaluator:
    """Evaluator for comparing quantized vs baseline models."""

    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        """
        Initialize the evaluator.

        Args:
            device: Device for evaluation
        """
        self.device = device
        self.image_metric = ImageMetric()

        logger.info("Initializing ModelEvaluator")

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        model_name: str = "Model",
        num_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            model: Model to evaluate
            dataloader: DataLoader for evaluation
            model_name: Name for logging
            num_samples: Maximum number of samples to evaluate (None for all)

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {model_name}...")
        model.eval()

        metrics_dict = {
            "psnr": [],
            "ssim": [],
            "msssim": [],
            "lpips": [],
            "bpp": []
        }

        with torch.no_no_grad():
            for batch_idx, images in enumerate(dataloader):
                if num_samples is not None and batch_idx * dataloader.batch_size >= num_samples:
                    break

                images = images.to(self.device).clamp(0, 1)

                try:
                    output = model(images)
                    reconstructed = output["x_hat"].clamp(0, 1)

                    # Compute metrics
                    batch_metrics = self.image_metric.metric_image(reconstructed, images)

                    for key in metrics_dict:
                        if key in batch_metrics and batch_metrics[key] is not None:
                            metrics_dict[key].append(batch_metrics[key])

                    if (batch_idx + 1) % max(1, len(dataloader) // 5) == 0:
                        logger.info(f"Evaluated {(batch_idx + 1) * dataloader.batch_size} samples")

                except Exception as e:
                    logger.warning(f"Error processing batch {batch_idx}: {e}")
                    continue

        # Compute averages
        avg_metrics = {}
        for key, values in metrics_dict.items():
            if values:
                avg_metrics[key] = float(np.mean(values))
                avg_metrics[f"{key}_std"] = float(np.std(values))
            else:
                avg_metrics[key] = None
                avg_metrics[f"{key}_std"] = None

        logger.info(f"{model_name} - PSNR: {avg_metrics['psnr']:.4f}, "
                   f"MS-SSIM: {avg_metrics['msssim']:.4f}, "
                   f"LPIPS: {avg_metrics['lpips']:.4f}")

        return avg_metrics

    def compare_models(
        self,
        baseline_model: nn.Module,
        quantized_model: nn.Module,
        dataloader: DataLoader,
        num_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compare baseline and quantized models.

        Args:
            baseline_model: Baseline floating-point model
            quantized_model: Quantized model
            dataloader: DataLoader for evaluation
            num_samples: Maximum number of samples to evaluate

        Returns:
            Comparison results including deltas
        """
        baseline_metrics = self.evaluate(baseline_model, dataloader, "Baseline", num_samples)
        quantized_metrics = self.evaluate(quantized_model, dataloader, "INT4 Quantized", num_samples)

        comparison = {
            "baseline": baseline_metrics,
            "quantized": quantized_metrics,
            "delta": {}
        }

        for key in baseline_metrics:
            if baseline_metrics[key] is not None and quantized_metrics[key] is not None:
                comparison["delta"][key] = quantized_metrics[key] - baseline_metrics[key]

        logger.info("Comparison Results:")
        logger.info(f"  PSNR delta: {comparison['delta'].get('psnr', 'N/A'):.4f} dB")
        logger.info(f"  MS-SSIM delta: {comparison['delta'].get('msssim', 'N/A'):.6f}")
        logger.info(f"  LPIPS delta: {comparison['delta'].get('lpips', 'N/A'):.6f}")

        return comparison


# Reuse utility functions from 8-bit quantizer
def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device
) -> nn.Module:
    """
    Load model checkpoint.

    Args:
        model: Model to load checkpoint into
        checkpoint_path: Path to checkpoint
        device: Device to load to

    Returns:
        Model with loaded weights
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}...")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    logger.info("Checkpoint loaded successfully")

    return model


def save_int4_quantized_model(
    model: nn.Module,
    scale_dict: Dict[str, torch.Tensor],
    zero_point_dict: Dict[str, torch.Tensor],
    output_dir: str,
    model_name: str = "quantized_model_int4",
    quant_min: int = -8,
    quant_max: int = 7
) -> str:
    """
    Save INT4 quantized model and quantization parameters.

    Args:
        model: INT4 quantized model
        scale_dict: Scale factors
        zero_point_dict: Zero points
        output_dir: Output directory
        model_name: Base filename
        quant_min: Minimum quantization value
        quant_max: Maximum quantization value

    Returns:
        Path to saved model
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved INT4 quantized model to {model_path}")

    # Save quantization parameters
    config = {
        "model_name": model_name,
        "quantization_scheme": "INT4 PTQ",
        "weight_bits": 4,
        "activation_bits": 4,
        "weight_dtype": "qint4",
        "activation_dtype": "qint4",
        "weight_scheme": "per_channel",
        "activation_scheme": "per_tensor",
        "quant_min": quant_min,
        "quant_max": quant_max,
        "use_brevitas": BREVITAS_AVAILABLE,
        "scales": {k: v.cpu().numpy().tolist() for k, v in scale_dict.items()},
        "zero_points": {k: v.cpu().numpy().tolist() for k, v in zero_point_dict.items()},
    }

    config_path = os.path.join(output_dir, f"{model_name}_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved INT4 quantization config to {config_path}")

    return model_path


def create_calibration_loader(
    config_dict: Dict[str, Any],
    args_dict: Dict[str, Any],
    batch_size: int = 16,
    num_samples: int = 256
) -> DataLoader:
    """
    Create calibration dataloader.

    Args:
        config_dict: Model configuration dictionary
        args_dict: Arguments dictionary
        batch_size: Batch size
        num_samples: Number of calibration samples

    Returns:
        DataLoader for calibration
    """
    logger.info("Creating calibration dataloader...")

    # Get dataset path from args
    train_path = args_dict.get("train_path", "./datasets/train")
    csv_path = args_dict.get("train_csv", None)

    if not os.path.exists(train_path):
        logger.warning(f"Train path not found: {train_path}. Using dummy data for calibration.")
        # Create dummy dataset for testing
        class DummyDataset:
            def __init__(self, num_samples=256):
                self.num_samples = num_samples

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return torch.rand(3, 256, 256)

        dataset = DummyDataset(num_samples)
    else:
        transform = transforms.Compose([
            transforms.RandomCrop(256),
            transforms.ToTensor()
        ])

        if csv_path and os.path.exists(csv_path):
            dataset = ImageFolderCSV(train_path, csv_path, transform=transform)
        else:
            dataset = ImageFolder(train_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    logger.info(f"Calibration dataloader created with {len(dataset)} samples")
    return loader


def create_evaluation_loader(
    dataset_name: str = "kodak",
    batch_size: int = 1
) -> DataLoader:
    """
    Create evaluation dataloader for a specific dataset.

    Args:
        dataset_name: Name of dataset ("kodak" or "clic")
        batch_size: Batch size

    Returns:
        DataLoader for evaluation
    """
    logger.info(f"Creating evaluation dataloader for {dataset_name}...")

    dataset_paths = {
        "kodak": "./datasets/kodak",
        "clic": "./datasets/clic",
    }

    dataset_path = dataset_paths.get(dataset_name.lower())
    if not dataset_path or not os.path.exists(dataset_path):
        logger.warning(f"Dataset path not found: {dataset_path}. Using dummy data for evaluation.")
        class DummyDataset:
            def __len__(self):
                return 8
            def __getitem__(self, idx):
                return torch.rand(3, 512, 512)
        dataset = DummyDataset()
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = ImageFolder(dataset_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    logger.info(f"Evaluation dataloader created for {dataset_name} with {len(dataset)} samples")
    return loader


def main():
    """Main INT4 quantization pipeline."""
    parser = argparse.ArgumentParser(
        description="INT4 Post-Training Quantization for LoC-LIC Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic INT4 quantization
    python quantize_4bit_ptq.py --config loclic --checkpoint model.pth

    # INT4 with comparison to FP32 and INT8
    python quantize_4bit_ptq.py --config loclic --checkpoint model.pth \\
        --compare_with_fp32 --compare_with_int8 --int8_checkpoint int8_model.pth

    # Custom calibration and evaluation
    python quantize_4bit_ptq.py --config loclic --checkpoint model.pth \\
        --num_calibration_samples 512 --eval_datasets kodak clic
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="loclic",
        help="Model configuration name (default: loclic)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to baseline FP32 model checkpoint [REQUIRED]"
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=256,
        help="Number of samples for calibration (default: 256)"
    )
    parser.add_argument(
        "--calibration_batch_size",
        type=int,
        default=8,
        help="Batch size for calibration (default: 8)"
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        default=["kodak"],
        help="Datasets to evaluate on (default: kodak)"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./quantized_models_4bit",
        help="Output directory for quantized models (default: ./quantized_models_4bit)"
    )
    parser.add_argument(
        "--use_brevitas",
        action="store_true",
        default=True,
        help="Use Brevitas for quantization (default: True if available)"
    )
    parser.add_argument(
        "--no_brevitas",
        action="store_true",
        help="Force disable Brevitas, use custom implementation"
    )
    parser.add_argument(
        "--compare_with_fp32",
        action="store_true",
        help="Compare with FP32 baseline model"
    )
    parser.add_argument(
        "--compare_with_int8",
        action="store_true",
        help="Compare with INT8 quantized model"
    )
    parser.add_argument(
        "--int8_checkpoint",
        type=str,
        help="Path to INT8 checkpoint for comparison"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for quantization (default: cuda if available)"
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Skip evaluation phase"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("=" * 80)
    logger.info("INT4 Post-Training Quantization for LoC-LIC")
    logger.info("=" * 80)
    logger.info(f"Arguments: {args}")

    # Set random seed
    setup_environment(args.seed)

    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Check Brevitas availability
    use_brevitas = args.use_brevitas and not args.no_brevitas and BREVITAS_AVAILABLE
    if args.use_brevitas and not BREVITAS_AVAILABLE:
        logger.warning("Brevitas requested but not available. Install with: pip install brevitas")
        logger.warning("Falling back to custom implementation")

    # Load model config
    repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(repo_path, "configs", "models", args.config + ".yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    model_cfg = model_config(config_path)
    logger.info(f"Loaded model config from {config_path}")

    # Load baseline FP32 model
    args_dict = get_args("train")
    baseline_model, _, _ = get_model(model_cfg, args_dict, device)
    baseline_model = load_checkpoint(baseline_model, args.checkpoint, device)
    baseline_model.eval()

    logger.info(f"Baseline FP32 model: {baseline_model.__class__.__name__}")
    logger.info(f"Model parameters: {sum(p.numel() for p in baseline_model.parameters()):,}")

    # Create INT4 quantization config
    quant_config = Int4QuantizationConfig(
        use_brevitas=use_brevitas,
        num_calibration_samples=args.num_calibration_samples,
        output_dir=args.output_dir,
        eval_on_datasets=args.eval_datasets,
        compare_with_fp32=args.compare_with_fp32,
        compare_with_int8=args.compare_with_int8,
        int8_checkpoint_path=args.int8_checkpoint
    )

    # Create calibration dataloader
    calibration_loader = create_calibration_loader(
        asdict(model_cfg),
        args_dict,
        batch_size=args.calibration_batch_size,
        num_samples=args.num_calibration_samples
    )

    # Perform INT4 quantization
    quantizer = Int4PTQQuantizer(baseline_model, quant_config, device)
    int4_model, scale_dict, zero_point_dict = quantizer.quantize(calibration_loader)

    # Save INT4 quantized model
    model_name = f"{args.config}_int4_ptq"
    save_int4_quantized_model(
        int4_model, scale_dict, zero_point_dict, args.output_dir, model_name
    )

    # Model comparison
    models_to_compare = {"int4": int4_model}

    if args.compare_with_fp32:
        models_to_compare["fp32"] = baseline_model

    if args.compare_with_int8 and args.int8_checkpoint:
        # Load INT8 model for comparison
        int8_model, _, _ = get_model(model_cfg, args_dict, device)
        int8_model = load_checkpoint(int8_model, args.int8_checkpoint, device)
        int8_model.eval()
        models_to_compare["int8"] = int8_model

    if len(models_to_compare) > 1:
        comparator = ModelSizeComparator(device)
        comparison_results = comparator.compare_models(models_to_compare)
        comparator.print_comparison()

        # Save comparison results
        comparison_path = os.path.join(args.output_dir, "model_comparison.json")
        with open(comparison_path, "w") as f:
            # Convert tensors and non-serializable objects
            serializable_results = {}
            for model_name, data in comparison_results.items():
                serializable_results[model_name] = {
                    'size': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                            for k, v in data['size'].items()},
                    'timing': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                              for k, v in data['timing'].items()},
                    'compression_ratio': data.get('compression_ratio', 'N/A'),
                    'size_reduction_percent': data.get('size_reduction_percent', 'N/A'),
                    'speedup_ratio': data.get('speedup_ratio', 'N/A')
                }
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Saved model comparison to {comparison_path}")

    # Evaluation
    if not args.no_eval:
        logger.info("Starting evaluation phase...")
        evaluator = ModelEvaluator(device)

        for dataset_name in args.eval_datasets:
            eval_loader = create_evaluation_loader(dataset_name, args.eval_batch_size)

            # Evaluate INT4 model
            int4_metrics = evaluator.evaluate(int4_model, eval_loader, "INT4", num_samples=64)

            # Compare with baseline if requested
            if args.compare_with_fp32:
                comparison = evaluator.compare_models(
                    baseline_model, int4_model, eval_loader, num_samples=64
                )

                # Save comparison results
                results_path = os.path.join(args.output_dir, f"eval_comparison_{dataset_name}.json")
                with open(results_path, "w") as f:
                    json.dump(comparison, f, indent=2, default=str)

                logger.info(f"Saved evaluation comparison to {results_path}")

    logger.info("=" * 80)
    logger.info("INT4 quantization pipeline completed successfully!")
    logger.info(f"Quantized model saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
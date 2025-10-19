import sys
import os
from PIL import ImageFile, Image
import shutil
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import torch.distributed as dist
from pathlib import Path
from torchvision.transforms import ToPILImage
from utils.func import update_registered_buffers, remap_old_keys

try:
    from compressai.entropy_models import EntropyBottleneck, GaussianConditional
except ImportError:
    EntropyBottleneck = None
    GaussianConditional = None

# Import update_registered_buffers and remap_old_keys from func module
# Get the parent directory and add to sys.path if not already there
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)



def patch_compressai_buffer_loading():
    """
    Patches CompressAI to handle missing registered buffers gracefully during checkpoint loading.
    This prevents KeyError when loading checkpoints with missing registered buffers.
    """
    try:
        from compressai.models import utils as compressai_utils
        
        original_update_registered_buffer = compressai_utils._update_registered_buffer
        
        def patched_update_registered_buffer(
            module,
            buffer_name,
            state_dict_key,
            state_dict,
            policy="resize_if_empty",
            dtype=None,
        ):
            if state_dict_key not in state_dict:
                print(f"Warning: Skipping missing buffer '{state_dict_key}' in checkpoint")
                return
            
            return original_update_registered_buffer(
                module, buffer_name, state_dict_key, state_dict, policy, dtype
            )
        
        compressai_utils._update_registered_buffer = patched_update_registered_buffer
        print("CompressAI buffer loading patch applied successfully")
    except Exception as e:
        print(f"Warning: Could not apply CompressAI patch: {e}")


def setup_environment(seed: int):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # setup DDP.
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size() 
        seed = rank % torch.cuda.device_count()
        seed = seed + rank
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    
    random.seed(seed)
    torch.manual_seed(seed)


""" configuration json """
class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def read_body(fd):
    lstrings = []
    shape = read_uints(fd, 2)
    n_strings = read_uints(fd, 1)[0]
    for _ in range(n_strings):
        s = read_bytes(fd, read_uints(fd, 1)[0])
        lstrings.append([s])

    return lstrings, shape


def write_body(fd, shape, out_strings):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())


class AverageMeter(object):
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def save_checkpoint(accelerator, state, is_best, output_dir, model=None):
    """Save checkpoint using Accelerator for proper distributed training support.
    
    Args:
        accelerator: Accelerator instance
        state: Dictionary containing epoch, loss, and other metadata
        is_best: Whether this is the best checkpoint so far
        output_dir: Directory to save checkpoints
        model: The model to update entropy bottleneck buffers (optional)
    """
    if accelerator.is_main_process:
        # # Update entropy model buffers before saving
        # if model is not None and (EntropyBottleneck is not None or GaussianConditional is not None):
        #     # Get the unwrapped model to access the actual modules
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     unwrapped_model.update(force=True)
            
        #     # Get the model's state dict to update buffers
        #     model_state_dict = unwrapped_model.state_dict()
            
        #     # Update registered buffers for entropy bottleneck and gaussian conditional modules
        #     for name, module in unwrapped_model.named_modules():
        #         if EntropyBottleneck is not None and isinstance(module, EntropyBottleneck):
        #             update_registered_buffers(
        #                 module,
        #                 name,
        #                 ["_quantized_cdf", "_offset", "_cdf_length"],
        #                 model_state_dict,
        #             )
        #             model_state_dict = remap_old_keys(name, model_state_dict)

        #         if GaussianConditional is not None and isinstance(module, GaussianConditional):
        #             update_registered_buffers(
        #                 module,
        #                 name,
        #                 ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
        #                 model_state_dict,
        #             )

        # Save the additional state (epoch, loss, etc.)
        epoch = state.get('epoch', 0)
        checkpoint_name = f"checkpoint_{epoch:03d}"
        checkpoint_path = Path(output_dir) / checkpoint_name
        
        # Save using accelerator with safe_serialization=False to handle shared tensors
        # This prevents warnings about shared tensors in entropy bottleneck modules
        accelerator.save_state(str(checkpoint_path), safe_serialization=False)
        
        # Save additional metadata
        metadata = {
            'epoch': state.get('epoch'),
            'loss': state.get('loss'),
        }
        metadata_path = checkpoint_path / "metadata.pth"
        torch.save(metadata, metadata_path)
        
        # If best, also copy to best checkpoint
        if is_best:
            best_checkpoint_path = Path(output_dir) / "checkpoint_best_loss"
            if best_checkpoint_path.exists():
                shutil.rmtree(best_checkpoint_path)
            shutil.copytree(checkpoint_path, best_checkpoint_path)


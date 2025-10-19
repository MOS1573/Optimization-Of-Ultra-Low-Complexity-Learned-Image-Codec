"""Vector quantizer.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Reference:
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
    https://github.com/google-research/magvit/blob/main/videogvt/models/vqvae.py
    https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/distributions/distributions.py
    https://github.com/lyndonzheng/CVQ-VAE/blob/main/quantise.py
"""
from typing import Mapping, Text, Tuple

import torch
from einops import rearrange
from accelerate.utils.operations import gather
from torch.cuda.amp import autocast


class VectorQuantizer(torch.nn.Module):
    def __init__(self,
                 codebook_size: int = 1024,
                 token_size: int = 256,
                 commitment_cost: float = 0.25,
                 use_l2_norm: bool = False,
                 clustering_vq: bool = False
                 ):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_size = token_size
        self.commitment_cost = commitment_cost

        self.embedding = torch.nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

        self.use_l2_norm = use_l2_norm
        self.clustering_vq = clustering_vq
        if clustering_vq:
            self.decay = 0.99
            self.register_buffer("embed_prob", torch.zeros(self.codebook_size))

    # Ensure quantization is performed using f32
    @autocast(enabled=False)
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        z = z.float()
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = rearrange(z, 'b h w c -> (b h w) c')
        unnormed_z_flattened = z_flattened

        if self.use_l2_norm:
            z_flattened = torch.nn.functional.normalize(z_flattened, dim=-1)
            embedding = torch.nn.functional.normalize(self.embedding.weight, dim=-1)
        else:
            embedding = self.embedding.weight

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, embedding.T)

        min_encoding_indices = torch.argmin(d, dim=1) # num_ele
        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)

        if self.use_l2_norm:
            z = torch.nn.functional.normalize(z, dim=-1)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        codebook_loss = torch.mean((z_quantized - z.detach()) ** 2)

        if self.clustering_vq and self.training:
            with torch.no_grad():
                # Gather distance matrix from all GPUs.
                encoding_indices = gather(min_encoding_indices)
                if len(min_encoding_indices.shape) != 1:
                    raise ValueError(f"min_encoding_indices in a wrong shape, {min_encoding_indices.shape}")
                # Compute and update the usage of each entry in the codebook.
                encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=z.device)
                encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
                avg_probs = torch.mean(encodings, dim=0)
                self.embed_prob.mul_(self.decay).add_(avg_probs, alpha=1-self.decay)

                # Closest sampling to update the codebook.
                all_d = gather(d)
                all_unnormed_z_flattened = gather(unnormed_z_flattened).detach()
                if all_d.shape[0] != all_unnormed_z_flattened.shape[0]:
                    raise ValueError(
                        "all_d and all_unnormed_z_flattened have different length" + 
                        f"{all_d.shape}, {all_unnormed_z_flattened.shape}")
                
                indices = torch.argmin(all_d, dim=0)
                random_feat = all_unnormed_z_flattened[indices]

                # Decay parameter based on the average usage.
                decay = torch.exp(-(self.embed_prob * self.codebook_size * 10) / (1 - self.decay) - 1e-3).unsqueeze(1).repeat(1, self.token_size)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay

        loss = commitment_loss + codebook_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            min_encoding_indices=min_encoding_indices.view(z_quantized.shape[0], z_quantized.shape[2], z_quantized.shape[3])
        )
        return z_quantized, result_dict

    def get_codebook_entry(self, indices):
        if len(indices.shape) == 1:
            z_quantized = self.embedding(indices)
        elif len(indices.shape) == 2:
            z_quantized = torch.einsum('bd,dn->bn', indices, self.embedding.weight)
        else:
            raise NotImplementedError
        
        if self.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        return z_quantized


@autocast(enabled=False)
def create_diagonal_gaussian_distribution(parameters):
    """Create a diagonal Gaussian distribution dictionary from parameters.

    Args:
        parameters (torch.Tensor): The parameters for the Gaussian distribution. It is expected
            to be in shape [B, 2 * C, *], where B is batch size, and C is the embedding dimension.
            First C channels are used for mean, second C for logvar.

    Returns:
        dict: Dictionary containing mean, logvar, and parameters.
    """
    mean, logvar = torch.chunk(parameters.float(), 2, dim=1)
    logvar = torch.clamp(logvar, -30.0, 20.0)
    
    return {
        'parameters': parameters,
        'mean': mean,
        'logvar': logvar
    }


def gaussian_std(logvar):
    """Compute standard deviation from log variance."""
    return torch.exp(0.5 * logvar)


def gaussian_var(logvar):
    """Compute variance from log variance."""
    return torch.exp(logvar)





@autocast(enabled=False)
def n_points_gaussian(dist_dict, num_points: int = 16):
    """
    Generate a Gaussian function with num_points.

    Args:
        dist_dict (dict): Dictionary containing distribution parameters (mean, logvar, parameters).
        num_points (int): The number of points to generate.

    Returns:
        torch.Tensor: The generated samples.
    """
    mean = dist_dict['mean']
    logvar = dist_dict['logvar']
    parameters = dist_dict['parameters']
    
    x = torch.linspace(-3, 3, steps=num_points).to(device=parameters.device)
    x = x.view(1, 1, 1, num_points)  # Shape: [1, 1, 1, num_points]
    x = x.repeat(mean.shape[0], mean.shape[1], mean.shape[2], 1)  # Shape: [B, token_size, num_latent_tokens, num_points]
    mean_expanded = mean.float()  # Shape: [B, num_latent_tokens, num_latent_tokens, 1]
    std_expanded = gaussian_std(logvar).float()    # Shape: [B, num_latent_tokens, num_latent_tokens, 1]
    gaussian = (1 / (std_expanded * (2 * torch.pi)**0.5)) * torch.exp(-0.5 * ((x - mean_expanded) / std_expanded) ** 2)
    return gaussian


@autocast(enabled=False)
def sample_diagonal_gaussian(dist_dict, token_points_size_vae: int = 16):
    x = n_points_gaussian(dist_dict, num_points=token_points_size_vae)
    return x


@autocast(enabled=False)
def mode_diagonal_gaussian(dist_dict):
    """Get the mode (mean) of the diagonal Gaussian distribution.
    
    Args:
        dist_dict (dict): Dictionary containing distribution parameters.
        
    Returns:
        torch.Tensor: The mode (mean) of the distribution.
    """
    return dist_dict['mean']


@autocast(enabled=False)
def kl_diagonal_gaussian(dist_dict):
    """Compute KL divergence for the diagonal Gaussian distribution.
    
    Args:
        dist_dict (dict): Dictionary containing distribution parameters.
        
    Returns:
        torch.Tensor: The KL divergence.
    """
    mean = dist_dict['mean']
    logvar = dist_dict['logvar']
    var = gaussian_var(logvar)
    
    return 0.5 * torch.sum(torch.pow(mean.float(), 2)
                            + var.float() - 1.0 - logvar.float(),
                            dim=[1, 2])
        
class SplitIntegers(torch.nn.Module):
    def __init__(self, max_depth: int, split_sizes: list[int]):
        super().__init__()
        self.max_depth = max_depth
        self.split_sizes = split_sizes
        if sum(self.split_sizes) != self.max_depth:
            raise ValueError("The sum of split_sizes must be equal to max_depth.")
        if any(size <= 0 for size in self.split_sizes):
            raise ValueError("All split sizes must be positive integers.")

    def forward(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """
        Splits the input tensor with high bit depth into multiple tensors with lower bit depth.

        Args:
            tensor (torch.Tensor): The input tensor to be split. It is expected to be in shape [B, C, H, W].
            split_sizes (list of int): A list of integers specifying the sizes of each split.
        """

        total_bits = self.max_depth
        splits = []
        start_bit = total_bits

        for size in self.split_sizes:
            end_bit = start_bit - size
            mask = (1 << size) - 1
            split = (tensor >> end_bit) & mask
            splits.append(split)
            start_bit = end_bit

        return splits


class CombineIntegers(torch.nn.Module):
    def __init__(self, max_depth: int, split_sizes: list[int]):
        super().__init__()
        self.max_depth = max_depth
        self.split_sizes = split_sizes
        if sum(self.split_sizes) != self.max_depth:
            raise ValueError("The sum of split_sizes must be equal to max_depth.")
        if any(size <= 0 for size in self.split_sizes):
            raise ValueError("All split sizes must be positive integers.")

    def forward(self, splits: list[torch.Tensor]) -> torch.Tensor:
        """
        Combines multiple tensors with lower bit depth into a single tensor with higher bit depth.

        Args:
            splits (list of torch.Tensor): A list of tensors to be combined. Each tensor is expected to be in shape [B, C, H, W].
            split_sizes (list of int): A list of integers specifying the sizes of each split.
        """

        if len(splits) != len(self.split_sizes):
            raise ValueError("The number of splits must match the number of split sizes.")

        combined = torch.zeros_like(splits[0], dtype=torch.int32)
        total_bits = self.max_depth
        start_bit = total_bits

        for split, size in zip(splits, self.split_sizes):
            end_bit = start_bit - size
            combined |= (split.to(torch.int32) & ((1 << size) - 1)) << end_bit
            start_bit = end_bit

        return combined
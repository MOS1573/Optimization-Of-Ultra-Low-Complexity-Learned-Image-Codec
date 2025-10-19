from typing import cast
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from compressai.models import CompressionModel
from utils.ckbd import *
from modules.transform import *
from modules.layers import *    
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from modules.layers.quantizer import (VectorQuantizer, SplitIntegers, CombineIntegers,
                                       create_diagonal_gaussian_distribution, 
                                       sample_diagonal_gaussian, kl_diagonal_gaussian)
from utils.func import EntropyBottleneck_vq
from torch.cuda.amp import autocast

class Model(CompressionModel):
    '''
    This model is based on the following implementations:
    - https://github.com/bytedance/1d-tokenizer
    '''
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.encoder = Encoder({**config,**config['Encoder']})
        self.decoder = Decoder({**config,**config['Decoder']})
        self.codebook_size = config['codebook_size']
        self.num_latent_tokens_vq = config['num_latent_tokens_vq']
        self.num_latent_tokens_vae = config['num_latent_tokens_vae']
        self.token_expand_size_vae = config['token_expand_size_vae']
        self.conv_expand_size_vae = nn.Sequential(
            nn.Conv2d(self.num_latent_tokens_vae, self.token_expand_size_vae, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.token_expand_size_vae, self.token_expand_size_vae, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        with torch.autocast('cuda', enabled=False):
            self.entropy_bottleneck_vq = EntropyBottleneck_vq(self.num_latent_tokens_vq)
            self.entropy_bottleneck_vae = EntropyBottleneck(self.num_latent_tokens_vae)

        # self.hyper_analysis_vq = HyperAnalysis(M=self.num_latent_tokens_vq, N=self.num_latent_tokens_vq//4)
        # self.hyper_analysis_vae = HyperAnalysis(M=self.num_latent_tokens_vae, N=self.num_latent_tokens_vae//4)

        # self.hyper_synthesis_vq = HyperSynthesis(N=self.num_latent_tokens_vq//4, M=self.num_latent_tokens_vq)
        # self.hyper_synthesis_vae = HyperSynthesis(N=self.num_latent_tokens_vae//4, M=self.num_latent_tokens_vae)

        # self.entropy_bottleneck_vq = EntropyBottleneck(self.num_latent_tokens_vq//4)
        # self.entropy_bottleneck_vae = EntropyBottleneck(self.num_latent_tokens_vae//4)

        # self.gaussian_conditional_vq = GaussianConditional(None)
        # self.gaussian_conditional_vae = GaussianConditional(None)

        scale = self.encoder.width ** -0.5
        self.latent_tokens_vq = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens_vq, self.encoder.width))
        self.latent_tokens_vae = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens_vae, self.encoder.width))
        
        self.apply(self._init_weights)

        self.quantize_vq = VectorQuantizer(
            codebook_size=config['codebook_size'],
            token_size=config['token_size_vq'],
            commitment_cost=config['commitment_cost'],
            use_l2_norm=config['use_l2_norm'],
            clustering_vq=config.get('clustering_vq', False)
        )
    def expand_vae(self, z):
        # Expand the latent tokens using conv layers
        z = self.conv_expand_size_vae(z).permute(0,3,2,1)
        return z

    @autocast(enabled=False)
    def quantize(self, z_vq, z_vae):
        _, result_dict_vq = self.quantize_vq(z_vq)
        indices = result_dict_vq['min_encoding_indices']
        indices = indices.permute(0,2,1).unsqueeze(-1).float() / self.codebook_size
        indices, indices_likelihoods = self.entropy_bottleneck_vq(indices)
        indices = (indices * self.codebook_size).squeeze().long()
        indices = torch.clamp(indices, 0, self.codebook_size - 1)

        if torch.all(result_dict_vq['min_encoding_indices'].squeeze()==indices):
            z_quantized_vq = z_vq
        else:
            z_quantized_vq = self.quantize_vq.get_codebook_entry(indices.reshape(-1)).reshape_as(z_vq)
            print("Warning: Mismatch in indices after entropy bottleneck vq!")

        z_quantized_vae, z_vae_likelihoods = self.entropy_bottleneck_vae(z_vae.float())

        # indices_hyper = self.hyper_analysis_vq(indices.permute(0,3,2,1))
    #     indices_hyper, indices_hyper_likelihoods = self.entropy_bottleneck_vq(indices_hyper)
    #     scales_hat = self.hyper_synthesis_vq(indices_hyper)
    #     indices, indices_likelihoods = self.gaussian_conditional_vq(indices, scales_hat)
    #     indices = torch.round(indices).long()
    #     z_quantized_vq = self.quantize_vq.get_codebook_entry(indices.reshape(-1)).reshape_as(z_vq)

        z_quantized_vae, result_dict_vae = self.expand_vae(z_quantized_vae), {}

        likelihoods = {
            "z_likelihoods_token_vq": indices_likelihoods,
            "z_likelihoods_token_vae": z_vae_likelihoods,
        }
        return z_quantized_vq, result_dict_vq, z_quantized_vae, result_dict_vae, likelihoods

    def encode(self, x):
        z_vq, z_vae = self.encoder(pixel_values=x, latent_tokens_vq=self.latent_tokens_vq, latent_tokens_vae=self.latent_tokens_vae)
        z_quantized_vq, result_dict_vq, z_quantized_vae, result_dict_vae, likelihoods = self.quantize(z_vq, z_vae)

        return z_quantized_vq, result_dict_vq, z_quantized_vae, result_dict_vae, likelihoods

    def decode(self, z_quantized_vq, z_quantized_vae):
        decoded = self.decoder(z_quantized_vq, z_quantized_vae)
        return decoded


    def get_aux_loss(self):
        loss = sum(m.loss() for m in self.modules() if (isinstance(m, EntropyBottleneck) or isinstance(m, EntropyBottleneck_vq)))
        return loss

    def forward(self, x):
        z_quantized_vq, result_dict_vq, z_quantized_vae, result_dict_vae, likelihoods = self.encode(x)
        x_hat = self.decode(z_quantized_vq, z_quantized_vae)

        return {
            "x_hat": x_hat,
            "likelihoods": likelihoods,
            "result_dict_vq": result_dict_vq,
            "result_dict_vae": result_dict_vae,
        }


    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @staticmethod
    def reconstruct_diagonal_gaussian_distribution(result_dict_vae_serializable):
        """Reconstruct diagonal Gaussian distribution dict from serializable dictionary."""
        return create_diagonal_gaussian_distribution(
            result_dict_vae_serializable["parameters"]
        )

    def decode_tokens(self, tokens, residual=None):
        tokens = tokens.squeeze(1)
        batch, seq_len = tokens.shape # B x N
        z_quantized = self.quantize.get_codebook_entry(
            tokens.reshape(-1)).reshape(batch, 1, seq_len, -1)
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        decoded = self.decode(z_quantized, residual)
        return decoded

    def compress(self, x):
        start_time = time.time()
        y, residual, result_dict = self.encode(x)
        indices = result_dict['min_encoding_indices']
        B, P, I = indices.size()
        indices = indices.view(B * P, I, 1, 1).to(torch.float32)/self.codebook_size
        indices_strings = self.entropy_bottleneck_token.compress(indices)
        if self.residual:
            residual_strings = self.entropy_bottleneck_residual.compress(residual)
        else:
            residual_strings = ''
        end_time = time.time()
        cost_time = end_time - start_time
        strings = [indices_strings, residual_strings]
        shape = [indices.size()[-2:], residual.size()[-2:]] if self.residual else [indices.size()[-2:], None]
        return {
            "strings": strings,
            "shape": shape,
            "cost_time": cost_time
        }

    def decompress(self, strings, shape):
        start_time = time.time()
        end_time = time.time()
        indices = self.entropy_bottleneck_token.decompress(strings[0], shape[0])
        indices = indices.view(-1, *shape[0])
        indices = indices.permute(0, 2, 3, 1).contiguous()
        if self.residual:
            residual = self.entropy_bottleneck_residual.decompress(strings[1], shape[1])
        else:
            residual = None

        x_hat = self.decode_tokens(indices, residual)
        cost_time = end_time - start_time

        return {
            "x_hat": x_hat,
            "cost_time": cost_time
        }


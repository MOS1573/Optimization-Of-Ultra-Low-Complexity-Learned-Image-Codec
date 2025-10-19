import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from loss.perceptual_loss import PerceptualLoss
from loss.discriminator import NLayerDiscriminator


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def set_lmbda(self, lmbda):
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out


class RateDistortionLoss1dToken(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self,
                    lmbda_bpp_loss=1e-2,
                    lmbda_quantizer_loss=1.0,
                    lmbda_perceptual_loss=1.0,
                    metrics='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss("lpips-convnext_s-1.0-0.14").eval()
        self.lmbda_bpp_loss = lmbda_bpp_loss
        self.lmbda_quantizer_loss = lmbda_quantizer_loss
        self.lmbda_perceptual_loss = lmbda_perceptual_loss
        self.metrics = metrics

    def get_perceptual_loss(self, x, x_rec):
        return self.perceptual_loss(x, x_rec)
    
    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["quantizer_loss"] = torch.sum(output["result_dict_vq"]['quantizer_loss'])
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out['perceptual_loss'] = self.get_perceptual_loss(output["x_hat"], target)

        out["loss"] =   out["mse_loss"]
        + self.lmbda_bpp_loss  * out["bpp_loss"]
        + self.lmbda_quantizer_loss * out["quantizer_loss"]  
        + self.lmbda_perceptual_loss * out['perceptual_loss']

        return out


# class RDloss(nn.Module):
#     def __init__(self, conf, local_rank=0):
#         super().__init__()
#         self.lambda_A = conf['lambda_A']
#         self.lambda_B = conf['lambda_B']
#         self.rate_A = conf['rate_A']
#         self.rate_B = conf['rate_B']
#         self.dist_type = conf['dist_type']
#         if self.dist_type == 'PSNR':
#             self.dist_func = lambda x, y: 255**2 * torch.mean((x - y)**2, dim=(1, 2, 3))
#         elif self.dist_type == 'MS-SSIM':
#             self.dist_func = ms_ssim(data_range=1.0, size_average=False, channel=3).to(local_rank)
#         else:
#             raise NotImplementedError

#         self.use_perceptual = conf.get('use_perceptual', False)
#         if self.use_perceptual:
#             self.perceptual_loss = PerceptualLoss(conf['perceptual_model_name']).to(local_rank)
#             self.lambda_perceptual = conf['lambda_perceptual']

#     def get_perceptual_loss(self, x, x_rec):
#         if self.use_perceptual:
#             return self.perceptual_loss(x, x_rec)
#         return 0

#     def forward(self, x, x_rec, bits_A, bits_B, s=1.0):
#         dist = self.dist_func(x_rec, x)
#         rd_loss = torch.mean(self.lambda_A * self.rate_A * bits_A + self.lambda_B * self.rate_B * bits_B + dist)

#         if self.use_perceptual:
#             perceptual_loss = self.get_perceptual_loss(x, x_rec)
#             rd_loss += self.lambda_perceptual * perceptual_loss.mean()

#         return rd_loss, torch.mean(bits_A), torch.mean(bits_B), torch.mean(dist)


# def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
#     """Hinge loss for discrminator.
#     This function is borrowed from
#     https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py#L20
#     """
#     loss_real = torch.mean(F.relu(1.0 - logits_real))
#     loss_fake = torch.mean(F.relu(1.0 + logits_fake))
#     d_loss = 0.5 * (loss_real + loss_fake)
#     return d_loss


# class GANLoss(nn.Module):
#     def __init__(self, conf):
#         super().__init__()
#         self.discriminator = NLayerDiscriminator(
#             num_channels=conf.get('disc_num_channels', 3),
#             hidden_channels=conf.get('disc_hidden_channels', 128),
#             num_stages=conf.get('disc_num_stages', 3),
#             blur_resample=conf.get('disc_blur_resample', True),
#             blur_kernel_size=conf.get('disc_blur_kernel_size', 4)
#         )
#         self.discriminator_weight = conf.get('discriminator_weight', 1.0)

#     def forward(self, x, x_rec, mode, global_step):
#         if mode == 'generator':
#             # Generator loss
#             logits_fake = self.discriminator(x_rec)
#             g_loss = -torch.mean(logits_fake)
#             loss = self.discriminator_weight * g_loss
#             return loss, {'g_loss': g_loss}
#         elif mode == 'discriminator':
#             # Discriminator loss
#             logits_real = self.discriminator(x.detach())
#             logits_fake = self.discriminator(x_rec.detach())
#             d_loss = hinge_d_loss(logits_real, logits_fake)
#             loss = self.discriminator_weight * d_loss
#             return loss, {'d_loss': d_loss, 'logits_real': logits_real.mean(), 'logits_fake': logits_fake.mean()}
#         else:
#             raise ValueError(f"Unknown mode for GAN loss: {mode}")

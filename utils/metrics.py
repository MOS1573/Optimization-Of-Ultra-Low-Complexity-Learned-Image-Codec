import os
import torch
import numpy as np
import scipy.interpolate
import PIL.Image as Image
from typing import Dict, List, Optional, Tuple, Union
from pytorch_msssim import ms_ssim
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


class ImageMetric():
    def __init__(self,device):
        self.psnr = PeakSignalNoiseRatio().to(device)
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
        self.msssim = MultiScaleStructuralSimilarityIndexMeasure().to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        self.fid   = FrechetInceptionDistance(feature=2048).to(device)
        self.inception_score = InceptionScore().to(device)
        self.device = device
        
    def metric_image(self, output, target, size_of_image=None): 
        output, target, original_image_size = self.get_image_sizes(output, target)         
        output, target = output.to(self.device), target.to(self.device)
        psnr = self.psnr(output, target).item()
        ssim = self.ssim(output, target).item()
        msssim = self.msssim(output, target).item()

        try :
            lpips = self.lpips(output, target).item()
        except:
            lpips = 0
            print('Could not compute LPIPS')

        if original_image_size is not None and size_of_image is not None:
            bit_per_pixel = size_of_image * 8 / (original_image_size[0] * original_image_size[1])
        else:
            bit_per_pixel = None
            
        metrics = { 
            'psnr': psnr,
            'ssim': ssim, 
            'msssim': msssim,
            'lpips': lpips,
            'bpp': bit_per_pixel,
                }
        return metrics
    
    def get_image_sizes(self, output, target):    
        original_image_size = target.shape[-2:]
        reconstructed_image_size = output.shape[-2:]

        min_high = min(original_image_size[0], reconstructed_image_size[0])
        min_width = min(original_image_size[1], reconstructed_image_size[1])
        target = target[:,:,:min_high,:min_width]
        output = output[:,:,:min_high,:min_width]
        return target, output, (min_high, min_width)

    def get_FID_and_InceptionScore(self, output, target, clip_size=None):
        if clip_size is not None:
            output = [img[:,:,:clip_size[0],:clip_size[1]] for img in output]
            target = [img[:,:,:clip_size[0],:clip_size[1]] for img in target]
        else:   
            # clip the images to the same size which is the minimum size
            min_high = min([img.shape[-2] for img in output + target]) 
            min_width = min([img.shape[-1] for img in output + target])

            target = [img[:,:,:min_high,:min_width] for img in target]
            output = [img[:,:,:min_high,:min_width] for img in output]

        target = torch.cat(target, dim=0)
        output = torch.cat(output, dim=0)


        target = target.mul(255).clamp(0, 255).to(torch.uint8).to(self.device)
        output = output.mul(255).clamp(0, 255).to(torch.uint8).to(self.device)

        self.fid.update(target, real=True)
        self.fid.update(output, real=False)

        fid = self.fid.compute()
        self.fid.reset()
        inception_score = self.inception_score(output)

        print(f'FID: {fid.item()}, Inception Score Mean: {inception_score[0].item()}', 'Inception Score Std: ', inception_score[1].item())
        return {'fid': fid.item(), 'inception_score_mean': inception_score[0].item(), 'inception_score_std': inception_score[1].item()}
    
    def append_to_metrics_dict(self, new_metrics: Dict, metrics_dict: Dict=None) -> Dict:
        if metrics_dict is None:
            metrics_dict = {}
        for key, value in new_metrics.items():
            if key in metrics_dict:
                metrics_dict[key].append(value)
            else:
                metrics_dict[key] = [value]
        return metrics_dict
    
    def get_avg_metrics(self, metrics_dict: Dict) -> Dict:
        avg_metrics = {}
        for key, values in metrics_dict.items():
            avg_metrics[key] = np.mean(values)
        return avg_metrics

def write_metrics_to_csv(metrics: Dict, csv_path: str, overwrite: bool=False):
    metrics_col = ['model_name', 'quality', 'dataset_name', 'psnr', 'ssim', 'msssim', 'lpips', 'fid', 'inception_score_mean', 'inception_score_std', 'bpp', 'enc_time', 'dec_time']
    if overwrite or not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write(','.join(metrics_col) + '\n')
    with open(csv_path, 'a') as f:
        # if the metric is not computed, write an empty string
        row = [str(metrics.get(col, '')) for col in metrics_col]
        f.write(','.join(row) + '\n')
        
    


def bjontegaard_metric_psnr(R1, PSNR1, R2, PSNR2, piecewise=0):
    '''
    This function is taken from:https://github.com/Anserw/Bjontegaard_metric/blob/master/bjontegaard_metric.py
    '''

    lR1 = np.log(R1)
    lR2 = np.log(R2)

    PSNR1 = np.array(PSNR1)
    PSNR2 = np.array(PSNR2)

    p1 = np.polyfit(lR1, PSNR1, 3)
    p2 = np.polyfit(lR2, PSNR2, 3)

    # integration interval
    min_int = max(min(lR1), min(lR2))
    max_int = min(max(lR1), max(lR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(lR1), PSNR1[np.argsort(lR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(lR2), PSNR2[np.argsort(lR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapezoid(v1, dx=interval)
        int2 = np.trapezoid(v2, dx=interval)

    # find avg diff
    avg_diff = (int2-int1)/(max_int-min_int)

    return avg_diff


def bjontegaard_metric_rate(R1, PSNR1, R2, PSNR2, piecewise=0):
    '''
    This function is taken from:https://github.com/Anserw/Bjontegaard_metric/blob/master/bjontegaard_metric.py
    '''
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # rate method
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)

    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(PSNR1), lR1[np.argsort(PSNR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(PSNR2), lR2[np.argsort(PSNR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_exp_diff = (int2-int1)/(max_int-min_int)
    avg_diff = (np.exp(avg_exp_diff)-1)*100
    return avg_diff


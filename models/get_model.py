import torch
from models import *
from loss.rd_loss import RateDistortionLoss, RateDistortionLoss1dToken
from utils.utils import CustomDataParallel

def get_model(config, args, device):
    if config.Model == 'LoC_LIC':
        net=LoC_LIC(config=config)
        loss = RateDistortionLoss(lmbda=args.lmbda)
        vae = None
    else:
        raise ValueError(f"Model {config.model} not found.")

    if torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
        net = net.to(device) 
    return net, vae, loss
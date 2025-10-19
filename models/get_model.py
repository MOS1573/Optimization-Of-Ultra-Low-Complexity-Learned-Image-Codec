from models import *
from loss.rd_loss import RateDistortionLoss, RateDistortionLoss1dToken

def get_model(config, args, device):
    if config.Model == 'opod_tic':
        net=Model(config=config)
        loss = RateDistortionLoss1dToken(**args.losses)
        vae = None
    elif config.Model == 'LoC_LIC':
        net=LoC_LIC(config=config)
        loss = RateDistortionLoss(lmbda=args.lmbda)
        vae = None
    elif config.Model == 'LoC_LIC_Lite':
        net = LoC_LIC_Lite(config=config)
        loss = RateDistortionLoss(lmbda=args.lmbda)
        vae = None
    elif config.Model == 'LoC_LIC_residually':
        net = LoC_LIC_residually(config=config)
        loss = RateDistortionLoss(lmbda=args.lmbda)
        vae = None
    else:
        raise ValueError(f"Model {config.model} not found.")
    return net, vae, loss
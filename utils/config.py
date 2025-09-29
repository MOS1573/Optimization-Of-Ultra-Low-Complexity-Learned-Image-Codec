import torch.nn as nn
from utils.utils import Config
import yaml
import torch
def model_config(config_path=None):
    if config_path is not None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        # act to the config
        config["act"] = nn.GELU
        return Config(config)
    else:
        raise ValueError("config path is None")
    
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

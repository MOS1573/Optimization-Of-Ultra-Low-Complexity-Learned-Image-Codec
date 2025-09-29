import torch
import numpy as np
from PIL import Image
from utils.config import get_device
def read_image(image_path, device=get_device()):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = np.array(image)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
    return image
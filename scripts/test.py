import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
from re import T
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.args import test_options
from utils.config import model_config
from PIL import ImageFile, Image
from utils.metrics import ImageMetric, write_metrics_to_csv
from utils.utils import *
from PIL import Image
from models.get_model import get_model
from utils.config import get_device
from utils.image import read_image

def compress_one_image(model, x, stream_path, H, W, img_name):
    with torch.no_grad():
        out = model.compress(x)

    shape = out["shape"]
    output = os.path.join(stream_path, img_name)
    with Path(output).open("wb") as f:
        write_uints(f, (H, W))
        write_body(f, shape, out["strings"])

    size = filesize(output)
    bpp = float(size) * 8 / (H * W)
    return bpp, out["cost_time"]


def decompress_one_image(model, stream_path, img_name):
    output = os.path.join(stream_path, img_name)
    with Path(output).open("rb") as f:
        original_size = read_uints(f, 2)
        strings, shape = read_body(f)

    with torch.no_grad():
        out = model.decompress(strings, shape)

    x_hat = out["x_hat"]
    x_hat = x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
    cost_time = out["cost_time"]
    return x_hat, cost_time


def test_single_dataset(images_dir, net, save_dir, dataset_name, quality, model_name):
    save_dir_metrics = os.path.join(os.path.dirname(save_dir), 'metrics')
    os.makedirs(save_dir_metrics, exist_ok=True)
    metrics_csv_path = os.path.join(save_dir_metrics, model_name + '.csv')
    metrics_avg_csv_path = os.path.join(save_dir_metrics, 'average_' + model_name + '.csv')

    save_dir = os.path.join(save_dir, model_name)
    save_dir = os.path.join(save_dir, "images", dataset_name, quality, )
    os.makedirs(save_dir, exist_ok=True)

    save_log_dir = os.path.join(save_dir, 'results.csv')
    save_log_avg_dir = os.path.join(save_dir, 'results_avg.csv')
    
    net.eval()
    device = next(net.parameters()).device
    image_metrics=ImageMetric(device=device)

    # get all the images in the images_dir
    images = os.listdir(images_dir)
    images = [os.path.join(images_dir, img) for img in images if img.endswith('.png') or img.endswith('.jpg')]
    metrics_dict = {}


    with torch.no_grad():
        for i, img_path in enumerate(images):
            image_name = img_path.split('/')[-1].split('.')[0]
            img =read_image(img_path, device)
            img = img.to(device)
            B, C, H, W = img.shape
            pad_h = 0
            pad_w = 0
            if H % 64 != 0:
                pad_h = 64 * (H // 64 + 1) - H
            if W % 64 != 0:
                pad_w = 64 * (W // 64 + 1) - W
            img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
            # warmup GPU
            bitstream_path = os.path.join(save_dir, image_name)
            if i == 0:
                bpp, enc_time = compress_one_image(model=net, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=image_name)
            try:
                # avoid resolution leakage
                net.update_resolutions(16, 16)
            except:
                pass
            bpp, enc_time = compress_one_image(model=net, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=image_name)
            # avoid resolution leakage
            try:
                net.update_resolutions(16, 16)
            except:
                pass
            x_hat, dec_time = decompress_one_image(model=net, stream_path=save_dir, img_name=image_name)
            bitstream_size = os.path.getsize(bitstream_path)

            metrics = image_metrics.metric_image(output=x_hat, target=img, size_of_image=bitstream_size)
            metrics_dict = image_metrics.append_to_metrics_dict(metrics, metrics_dict)

            metrics['model_name'], metrics['dataset_name'], metrics['quality'] = model_name, dataset_name, quality

            rec = torch2img(x_hat)
            img = torch2img(img)
            rec.save(os.path.join(save_dir, image_name + '.png'))
            write_metrics_to_csv(metrics, metrics_csv_path, overwrite=False)
            write_metrics_to_csv(metrics, save_log_dir, overwrite=False)
        avg_metrics = image_metrics.get_avg_metrics(metrics_dict)
        avg_metrics['model_name'], avg_metrics['dataset_name'], avg_metrics['quality'] = model_name, dataset_name, quality
        write_metrics_to_csv(avg_metrics, metrics_avg_csv_path, overwrite=False)
        write_metrics_to_csv(avg_metrics, save_log_avg_dir, overwrite=False)
    

def main():
    args = test_options()
    target_checkpoint = "checkpoint_best_loss.pth.tar"
    checkpoint_path = os.path.join(args.main_path, args.checkpoint)
    dataset_dir = os.path.join(args.main_path, args.test_dataset)
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if checkpoint_path.endswith('.pth.tar'):
        checkpoints = [checkpoint_path]
    else:
        # find all the checkpoints in the directory with the same name 
        checkpoints = []
        for root, _, files in os.walk(checkpoint_path):
            for file in files:
                if file.endswith(target_checkpoint):
                    checkpoints.append(os.path.join(root, file))

    for checkpoint_path in checkpoints:
        torch.cuda.empty_cache()
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        Image.MAX_IMAGE_PIXELS = None
        torch.backends.cudnn.deterministic = True

        model_description = checkpoint_path.split('/')[-3]
        model_name = '_'.join(model_description.split('_')[0:-1])
        config_name = '_'.join(model_name.split('_'))
        quality = model_description.split('_')[-1]

        dataset_name = args.test_dataset.split('/')[-1]

        if args.config is not None:
            config_path = os.path.join(repo_path,'configs', 'models', args.config + '.yaml')
        else:
            config_path = os.path.join(repo_path,'configs', 'models', config_name + '.yaml')
        
        config = model_config(config_path)

        save_dir = os.path.join(args.main_path, args.output_dir, "models", )
        os.makedirs(save_dir, exist_ok=True)

        device = get_device()

        net, vae, criterion = get_model(config, args, device)
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['state_dict'])
        test_single_dataset(net=net, images_dir=dataset_dir, save_dir=save_dir, dataset_name=dataset_name, quality=quality, model_name=model_name)

if __name__ == '__main__':
    main()


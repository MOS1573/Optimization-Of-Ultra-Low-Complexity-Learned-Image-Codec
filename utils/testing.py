import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import compute_metrics, ImageMetric
from utils.utils import *
import random

def test_one_epoch(epoch, test_dataloader, model, vae, criterion, logger_val, tb_logger):
    model.eval()
    device = next(model.parameters()).device
    image_metric = ImageMetric(device)

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    ssim = AverageMeter()
    ms_ssim = AverageMeter()
    lpips = AverageMeter()

    with torch.no_grad():
        for i, image in enumerate(test_dataloader):
            image = image.to(device).clamp(0, 1)

            if vae is not None:    
                d = vae.encode(image)
            else:
                d = image

            out_net = model(d)
            out_criterion = criterion(out_net, d)

            if vae is not None:
                rec = vae.decode(out_net["x_hat"]).clamp(0, 1)
            else:
                rec = out_net["x_hat"].clamp(0, 1)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

            metrics = image_metric.metric_image(rec, image)

            psnr.update(metrics['psnr'])
            ssim.update(metrics['ssim'])
            ms_ssim.update(metrics['msssim'])
            lpips.update(metrics['lpips'])

    tb_logger.add_scalar('{}'.format('[val]: loss'), loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: aux_loss'), aux_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: bpp_loss'), bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: psnr'), psnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: ssim'), ssim.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: ms-ssim'), ms_ssim.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: lpips'), lpips.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: mse_loss'), mse_loss.avg, epoch + 1)

    # log the first image of the last batch in tensorboard
    image_index = random.randint(0, image.size(0) - 1)
    tb_logger.add_image('{}'.format('[val]: input'), image[image_index], epoch + 1)
    tb_logger.add_image('{}'.format('[val]: rec'), rec[image_index], epoch + 1)

    logger_val.info(
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.4f} | "
        f"Bpp loss: {bpp_loss.avg:.4f} | "
        f"Aux loss: {aux_loss.avg:.2f} | "
        f"PSNR: {psnr.avg:.6f} | "
        f"MS-SSIM: {ms_ssim.avg:.6f} | "
        f"LPIPS: {lpips.avg:.6f}" 
    )   

    return loss.avg

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



def test_model(test_dataloader, net, vae, logger_test, save_dir, epoch):
    net.eval()
    device = next(net.parameters()).device

    avg_psnr = AverageMeter()
    avg_ms_ssim = AverageMeter()
    avg_bpp = AverageMeter()
    avg_enc_time = AverageMeter()
    avg_dec_time = AverageMeter()

    with torch.no_grad():
        for i, img in enumerate(test_dataloader):
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
            bitstream_path = os.path.join(save_dir, str(i))
            if i == 0:
                bpp, enc_time = compress_one_image(model=net, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=str(i))
            # avoid resolution leakage
            net.update_resolutions(16, 16)
            bpp, enc_time = compress_one_image(model=net, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=str(i))
            # avoid resolution leakage
            net.update_resolutions(16, 16)
            x_hat, dec_time = decompress_one_image(model=net, stream_path=save_dir, img_name=str(i))
            rec = torch2img(x_hat)
            img = torch2img(img)
            # img.save(os.path.join(save_dir, '%03d_gt.png' % i))
            # rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
            rec.save(os.path.join(save_dir, f'{i}.png'))
            p, m = compute_metrics(rec, img)
            avg_psnr.update(p)
            avg_ms_ssim.update(m)
            avg_bpp.update(bpp)
            avg_enc_time.update(enc_time)
            avg_dec_time.update(dec_time)
            # get bitstream size
            bitstream_size = os.path.getsize(bitstream_path)
            logger_test.info(
                f"Image[{i}] | "
                # f"Bpp loss: {bpp:.2f} | "
                f"Bpp : {bpp} | "
                f"PSNR: {p:.4f} | "
                f"MS-SSIM: {m:.4f} | "
                f"Encoding Latency: {enc_time:.4f} | "
                f"Decoding Latency: {dec_time:.4f} | "
                f"Bitstream size: {bitstream_size}" 
            )
    logger_test.info(
        f"Epoch:[{epoch}] | "
        f"Avg Bpp: {avg_bpp.avg:.4f} | "
        f"Avg PSNR: {avg_psnr.avg:.4f} | "
        f"Avg MS-SSIM: {avg_ms_ssim.avg:.4f} | "
        f"Avg Encoding Latency:: {avg_enc_time.avg:.4f} | "
        f"Avg decoding Latency:: {avg_dec_time.avg:.4f}"
    )

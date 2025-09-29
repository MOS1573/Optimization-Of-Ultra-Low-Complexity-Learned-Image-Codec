import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import random
import logging
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import ImageFile, Image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from utils.logger import setup_logger
from utils.utils import save_checkpoint
from utils.optimizers import configure_optimizers
from utils.training import train_one_epoch  
from utils.testing import test_one_epoch
from utils.args import train_options
from utils.config import model_config
from datasets.datasets import ImageFolder, ImageFolderCSV
from models.get_model import get_model
from utils.schedules import get_cosine_schedule_with_warmup



def main():
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    args = train_options()
    model_config_path = os.path.join(repo_path, 'configs', 'models', args.config + '.yaml')
    config = model_config(model_config_path)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)

    experiment_path = os.path.join(args.main_path, 'experiments', args.config, args.experiment)
    os.makedirs(experiment_path, exist_ok=True)

    setup_logger('train', experiment_path, 'train_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    setup_logger('val', experiment_path, 'val_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)

    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')

    tb_logger = SummaryWriter(log_dir=experiment_path)

    checkpoint_path = os.path.join(experiment_path, 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.CenterCrop(args.patch_size)]
    )

    csv_file_path = os.path.join(repo_path, 'datasets', 'csv', args.dataset_csv + '.csv')
    dataset_path = os.path.join(args.main_path, args.dataset)
    train_dataset = ImageFolderCSV(dataset_path,csv_file=csv_file_path, transform=train_transforms)
    test_dataset = ImageFolder(os.path.join(dataset_path, 'test'), transform=test_transforms) # The test images is not included a csv file so the model will not be trained on them
    

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    net, vae, criterion = get_model(config, args, device)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.epochs//10, num_training_steps=args.epochs, base_lr=args.learning_rate, end_lr=1e-5)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)


    best_checkpoint = os.path.join(experiment_path, 'checkpoints', 'checkpoint_best_loss.pth.tar')
    if os.path.exists(best_checkpoint):
        checkpoint = torch.load(best_checkpoint)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print(lr_scheduler.state_dict())
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        current_step = start_epoch * math.ceil(len(train_dataloader.dataset) / args.batch_size)
        checkpoint = None
    else:
        start_epoch = 0
        best_loss = 1e10
        current_step = 0

    logger_train.info(args)
    logger_train.info(config)
    logger_train.info(net)
    logger_train.info(optimizer)
    optimizer.param_groups[0]['lr'] = args.learning_rate

    for epoch in range(start_epoch, args.epochs):
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        current_step = train_one_epoch(
            net,
            vae,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            logger_train,
            tb_logger,
            current_step,
        )

        loss = test_one_epoch(epoch, test_dataloader, net, vae, criterion, logger_val, tb_logger)

        lr_scheduler.step()
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                os.path.join(experiment_path, "checkpoints","checkpoint_%03d.pth.tar" % (epoch + 1))
            )
            if is_best:
                logger_val.info('best checkpoint saved.')

if __name__ == '__main__':
    main()

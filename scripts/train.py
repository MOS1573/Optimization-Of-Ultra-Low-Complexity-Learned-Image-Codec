import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import math
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from utils.logger import setup_logger
from utils.utils import save_checkpoint, patch_compressai_buffer_loading
from utils.optimizers import configure_optimizers
from utils.training import train_one_epoch  
from utils.testing import test_one_epoch
from utils.args import train_options
from utils.config import model_config
from datasets.datasets import ImageFolder, ImageFolderCSV
from models.get_model import get_model
from utils.schedules import get_cosine_schedule_with_warmup
from utils.utils import setup_environment
from utils.metrics import ImageMetric
patch_compressai_buffer_loading()



def main():
    args = train_options()
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_config_path = os.path.join(repo_path, 'configs', 'models', args.config + '.yaml')
    config = model_config(model_config_path)


    setup_environment(args.seed)

    # Configure DDP to handle unused parameters
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # Initialize Accelerator for distributed training
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,  # 'no', 'fp16', 'bf16'
        gradient_accumulation_steps=1,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device


    experiment_path = os.path.join(args.main_path, 'experiments', args.experiment)
    
    # Only create directories on main process
    if accelerator.is_main_process:
        os.makedirs(experiment_path, exist_ok=True)

    # Wait for main process to create directories
    accelerator.wait_for_everyone()

    setup_logger('train', experiment_path, 'train_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    setup_logger('val', experiment_path, 'val_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)

    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')

    # TensorBoard logger only on main process
    tb_logger = SummaryWriter(log_dir=experiment_path) if accelerator.is_main_process else None

    checkpoint_path = os.path.join(experiment_path, 'checkpoints')
    if accelerator.is_main_process:
        os.makedirs(checkpoint_path, exist_ok=True)
    
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.CenterCrop(args.patch_size)]
    )

    csv_file_path = os.path.join(repo_path, 'datasets', 'csv', args.dataset_csv + '.csv')
    dataset_path = args.dataset
    train_dataset = ImageFolderCSV(dataset_path,csv_file=csv_file_path, transform=train_transforms)
    test_dataset = ImageFolder(os.path.join(dataset_path, 'test'), transform=test_transforms) # The test images is not included a csv file so the model will not be trained on them
    

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    net, vae, criterion = get_model(config, args, device)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.epochs*accelerator.num_processes//10,num_training_steps=args.epochs*accelerator.num_processes,base_lr=args.learning_rate, end_lr=1e-5)
    lr_scheduler_aux = get_cosine_schedule_with_warmup(aux_optimizer, num_warmup_steps=args.epochs*accelerator.num_processes//10, num_training_steps=args.epochs*accelerator.num_processes, base_lr=args.aux_learning_rate, end_lr=1e-5)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    # lr_scheduler_aux = torch.optim.lr_scheduler.CosineAnnealingLR(aux_optimizer, T_max=args.epochs, eta_min=1e-5)
    image_metric = ImageMetric()

    # Prepare everything with accelerator
    net, criterion, optimizer, aux_optimizer, train_dataloader, test_dataloader, lr_scheduler, lr_scheduler_aux, image_metric = accelerator.prepare(
        net, criterion, optimizer, aux_optimizer, train_dataloader, test_dataloader, lr_scheduler, lr_scheduler_aux, image_metric
    )

    best_checkpoint = os.path.join(experiment_path, 'checkpoints', 'checkpoint_best_loss')
    if os.path.exists(best_checkpoint):
        accelerator.load_state(best_checkpoint,strict=False)
        metadata_path = os.path.join(best_checkpoint, 'metadata.pth')
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path, map_location=device)
            start_epoch = metadata['epoch']
            best_loss = metadata['loss']
            current_step = start_epoch * math.ceil(len(train_dataloader.dataset) / args.batch_size)
            if accelerator.is_main_process:
                print(f"Loaded checkpoint from epoch {start_epoch} with loss {best_loss}")
        else:
            start_epoch = 0
            best_loss = 1e10
            current_step = 0
    else:
        start_epoch = 0
        best_loss = 1e10
        current_step = 0

    if accelerator.is_main_process:
        logger_train.info(args)
        logger_train.info(config)
        logger_train.info(net)
        logger_train.info(optimizer)

    for epoch in range(start_epoch, args.epochs):

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
            accelerator,
        )

        loss = test_one_epoch(epoch, test_dataloader, net, vae, criterion, logger_val, tb_logger, image_metric, accelerator)

        lr_scheduler.step()
        lr_scheduler_aux.step()
        if accelerator.is_main_process:
            logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
            # Save checkpoints
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    accelerator,
                    {
                        "epoch": epoch + 1,
                        "loss": loss,
                    },
                    is_best,
                    checkpoint_path,
                    model=net
                )
                if is_best:
                    logger_val.info('best checkpoint saved.')
    
        # Synchronize all processes
        accelerator.wait_for_everyone()

if __name__ == '__main__':
    main()

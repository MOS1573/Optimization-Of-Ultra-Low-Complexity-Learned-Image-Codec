import torch

def get_unwrapped_model(model):
    """Get the unwrapped model from a potentially wrapped model (e.g., DataParallel, DistributedDataParallel)."""
    if hasattr(model, 'module'):
        return model.module
    return model

def train_one_epoch(
    model, vae, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger_train, tb_logger, current_step, accelerator,
):
    model.train()
    
    for i, image in enumerate(train_dataloader):
        with accelerator.autocast():
            if vae is not None:
                vae.eval()
                with torch.no_grad():
                    d = vae.encode(image)
            else:
                d = image

            optimizer.zero_grad()
            out_net = model(d)

            out_criterion = criterion(out_net, d) 
            loss = out_criterion["loss"]

            accelerator.backward(loss)
            if clip_max_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()
            

            aux_optimizer.zero_grad()
            aux_loss = get_unwrapped_model(model).get_aux_loss()
            accelerator.backward(aux_loss)
            aux_optimizer.step()



        if current_step % 100 == 0 and accelerator.is_main_process:
            if tb_logger is not None:
                tb_logger.add_scalar('{}'.format('[train]: loss'), out_criterion["loss"].item(), current_step)
                tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion["bpp_loss"].item(), current_step)
                tb_logger.add_scalar('{}'.format('[train]: lr'), optimizer.param_groups[0]['lr'], current_step)
                tb_logger.add_scalar('{}'.format('[train]: aux_loss'), aux_loss.item(), current_step)
                tb_logger.add_scalar('{}'.format('[train]: mse_loss'), out_criterion["mse_loss"].item(), current_step)
                if vae is not None:
                    tb_logger.add_scalar('{}'.format('[train]: quantizer_loss'), out_criterion["quantizer_loss"].item(), current_step)

            logger_train.info(
                f"Train epoch {epoch}: ["
                f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)] "
                f'Loss: {out_criterion["loss"].item():.4f} | '
                f'MSE loss: {out_criterion["mse_loss"].item():.4f} | '
                f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                f"Aux loss: {aux_loss.item():.2f}"
            )


        current_step += 1
        accelerator.wait_for_everyone()
    return current_step
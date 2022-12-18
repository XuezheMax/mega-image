# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

try:
    import wandb
except ImportError:
    wandb = None


def analyze_grad(model, threshold):
    res = []
    max_norm = 0.0
    max_name = None
    named_params = [(name, param) for name, param in model.named_parameters() if param.grad is not None]
    for name, param in named_params:
        gnorm = torch.norm(param.grad.detach(), p=2).item()
        if gnorm > max_norm:
            max_norm = gnorm
            max_name = name
        if gnorm > threshold:
            res.append((name, gnorm))
    if len(res) == 0:
        return [(max_name, max_norm)]
    else:
        return res


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0, adaptive_grad: float = 0, cooldown_epochs: int = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, lr_scheduler=None, step_starts=None, enable_amp: bool = False):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ", wandb=utils.is_main_process(), step_starts=step_starts)
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=100, fmt="{avg:.4f} ({global_avg:.4f})"))
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('gnorm', utils.SmoothedValue(window_size=100, fmt="{max:.4f} ({global_avg:.4f})"))
    metric_logger.add_meter('agc', utils.SmoothedValue(window_size=1, fmt="{total:.0f}"))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    train_step = 0
    if adaptive_grad > 0. and epoch < cooldown_epochs:
        adaptive_grad = 0.
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=enable_amp):
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        total_norm, agc = loss_scaler(loss, optimizer,
                                      clip_grad=clip_grad,
                                      adaptive_grad=adaptive_grad,
                                      parameters=model.parameters(),
                                      create_graph=is_second_order)
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        lr_scheduler.step_update(epoch * len(data_loader) + train_step)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(gnorm=total_norm)
        metric_logger.update(agc=agc)
        train_step += 1
        # if agc:
        #     print('Adaptive Grad Norm: {:.2f}'.format(total_norm))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    res = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if utils.is_main_process() and wandb:
        stats = {'loss': metric_logger.meters['loss'].global_avg, 'lr': metric_logger.meters['lr'].value,
                 'gnorm': metric_logger.meters['gnorm'].global_avg}
        metric_logger.wandb.log(stats, tag="train", step=metric_logger.tot_steps)

    res["steps"] = train_step
    return res


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt="{global_avg:.4f}"))
    metric_logger.add_meter('acc1', utils.SmoothedValue(window_size=1, fmt="{global_avg:.4f}"))
    metric_logger.add_meter('acc5', utils.SmoothedValue(window_size=1, fmt="{global_avg:.4f}"))
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10000, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=False):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

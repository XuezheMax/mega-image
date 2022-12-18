""" CUDA / AMP utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import torch

try:
    from apex import amp
    has_apex = True
except ImportError:
    amp = None
    has_apex = False

from .clip_grad import dispatch_clip_grad
from .agc import adaptive_clip_grad2


class ApexScaler:
    state_dict_key = "amp"

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(create_graph=create_graph)
        if clip_grad is not None:
            dispatch_clip_grad(amp.master_params(optimizer), clip_grad, mode=clip_mode)
        optimizer.step()

    def state_dict(self):
        if 'state_dict' in amp.__dict__:
            return amp.state_dict()

    def load_state_dict(self, state_dict):
        if 'load_state_dict' in amp.__dict__:
            amp.load_state_dict(state_dict)


class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self, enabled):
        self._scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', adaptive_grad=None, parameters=None, create_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        agc = 0
        total_norm = 0.0
        if clip_grad is not None and clip_grad > 0.:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            total_norm = dispatch_clip_grad(parameters, clip_grad, mode=clip_mode).item()
            if adaptive_grad is not None and 0 < adaptive_grad < total_norm:
                adaptive_clip_grad2(parameters, clip_factor=clip_grad / total_norm, eps=1e-5)
                agc = 1

        self._scaler.step(optimizer)
        self._scaler.update()

        return total_norm, agc

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

<h1 align="center">Mega: Moving Average Equipped Gated Attention</h1>

# Mega for Image Classification
Implementation of [Mega](https://arxiv.org/abs/2209.10655) on Image Classification. This folder is based on the repos of [Pytorch Image Models](https://github.com/rwightman/pytorch-image-models) and [DeiT](https://github.com/facebookresearch/deit)

## Requirements

Python 3.7, Pytorch 1.11.0 Cuda11.3

## Installation

```
pip install --editable pytorch-image-models/  --user
```

## Model Checkpoints

Task | Description                          | # params | Download
---|--------------------------------------|---|---
`ImageNet-1k` | Mega on ImageNet-1k      | 90M | [mega.imagenet.zip](https://dl.fbaipublicfiles.com/mega/mega.imagenet.zip)

## Training Mega on ImageNet-1k
To train Mega-base on ImageNet on a single node with 8 gpus for 300 epochs with `slurm`:

```bash
srun --label python -u deit/main.py \
        --model mega_base_patch16_224 \
        --batch-size 128 \
        --lr 2e-3 \
        --seed 42 \
        --prenorm \
        --norm-type 'layernorm' \
        --drop-path 0.3 \
        --epochs 300 \
        --warmup-epochs 20 \
        --weight-decay 0.05 \
        --warmup-lr 1e-5 \
        --clip-grad 1.0 \
        --opt-betas 0.9 0.98 \
        --world_size 8 \
        --reprob 0.25 \
        --repeated-aug 3 \
        --data-path ${DATA_PATH} \
        --output_dir ${MODEL_PATH}
```

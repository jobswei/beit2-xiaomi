# --------------------------------------------------------
# BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers (https://arxiv.org/abs/2208.06366)
# Github source: https://github.com/microsoft/unilm/tree/master/beitv2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Zhiliang Peng
# Based on BEiT, timm, DeiT and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import sys
sys.path.append("/home/xiaomi/unilm/beit2/")
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.models import create_model
from optim_factory import create_optimizer

from datasets import build_beit_pretraining_dataset
from engine_for_pretraining import train_one_epoch
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import modeling_pretrain
import modeling_vqkd
from einops import rearrange, repeat
from InferenceDataset import build_beit_inference_dataset

def get_args():
    parser = argparse.ArgumentParser('BEiT inference script', add_help=False)

    # tokenizer settings
    parser.add_argument("--tokenizer_weight", type=str)
    parser.add_argument("--tokenizer_model", type=str, default="vqkd_encoder_base_decoder_3x768x12_clip")
    
    # Model parameters
    parser.add_argument('--model', default='beit_base_patch16_224_8k_vocab', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--num_mask_patches', default=75, type=int,
                        help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--second_input_size', default=224, type=int,
                        help='images input size for discrete vae')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # cls-pretraining settings
    parser.add_argument('--early_layers', default=9, type=int, help='early_layers, default 9 for base and 21 for large')
    parser.add_argument('--head_layers', default=2, type=int, help='head_layers')
    parser.add_argument('--shared_lm_head', default=True, type=utils.bool_flag, help='head_layers')

    # Tokenizer parameters
    parser.add_argument('--codebook_size', default=8192, type=int, help='number of codebook')
    parser.add_argument('--codebook_dim', default=32, type=int, help='number of codebook')


    # Augmentation parameters
    parser.add_argument('--decoupling_aug', default=False, type=utils.bool_flag, help="use decoupling aug for tokenizer and vit")
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--second_interpolation', type=str, default='lanczos',
                        help='Interpolation for discrete vae (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--min_crop_scale', type=float, default=0.08, metavar='PCT',
                        help='min_crop_scale (default: 0.08)')


    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default='', type=str, help='dataset path')
    parser.add_argument('--data_set', default='image_folder',  type=str, help='dataset path')

    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')    
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    if 'cls_pt' in args.model:
        model = create_model(
            args.model,
            pretrained=False,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            use_shared_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            vocab_size=args.codebook_size,
            early_layers=args.early_layers,
            head_layers=args.head_layers,
            shared_lm_head=args.shared_lm_head,
        )
    else:
        model = create_model(
            args.model,
            pretrained=False,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            use_shared_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            vocab_size=args.codebook_size
        )

    return model

def get_visual_tokenizer(args):
    print(f"Creating visual tokenizer: {args.tokenizer_model}")
    model = create_model(
            args.tokenizer_model,
            pretrained=True,
            pretrained_weight=args.tokenizer_weight,
            as_tokenzer=True,
            n_code=args.codebook_size, 
            code_dim=args.codebook_dim,
        ).eval()
    return model

def evaluate_res(gt,pre):
    same=gt==pre
    return torch.sum(same)
def main(args):
    # utils.init_distributed_mode(args)

    device = torch.device(args.device)

    model = get_model(args).to(device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model']) 
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size
    # prepare visual tokenizer
    vqkd = get_visual_tokenizer(args).to(device)

    # get dataset
    dataset_train = build_beit_inference_dataset(args)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    for step, data in enumerate(data_loader_train):
        batch=data[0]
        break

    samples, images, bool_masked_pos = batch
    images = images.to(device, non_blocking=True)
    samples = samples.to(device, non_blocking=True)
    bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            input_ids = vqkd.get_codebook_indices(images)
        bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
        labels = input_ids[bool_masked_pos]
        with torch.cuda.amp.autocast(): # enabled=False
            outputs = model(samples, bool_masked_pos=bool_masked_pos)
    
    masked_tokens=outputs[1]
    masked_tokens=torch.argmax(masked_tokens,dim=1)
    a=evaluate_res(masked_tokens,labels)
    print(a)





if __name__ == '__main__':
    opts = get_args()
    opts.model="beit_base_patch16_224_8k_vocab_cls_pt"
    opts.tokenizer_model="vqkd_encoder_base_decoder_3x768x12_clip"
    opts.resume="/home/xiaomi/unilm/beit2/checkpoints/beitv2_base_patch16_224_pt1k.pth"
    opts.tokenizer_weight="/home/xiaomi/unilm/beit2/checkpoints/vqkd_encoder_base_decoder_3x768x12_clip.pth"
    opts.data_set= "image_folder" 
    opts.data_path="/home/xiaomi/unilm/beit2/demo"
    opts.output_dir="/home/xiaomi/unilm/beit2/inference_output"

    # opts.model="beit_base_patch16_224_8k_vocab_cls_pt"
    # opts.tokenizer_model="vqkd_encoder_base_decoder_1x768x12_dino"
    # opts.resume="/home/xiaomi/unilm/beit2/beit_output/checkpoint-59.pth"
    # opts.tokenizer_weight="/home/xiaomi/unilm/beit2/vqkd_outputs/checkpoint-99.pth"
    # opts.data_set= "image_folder" 
    # opts.data_path="/home/xiaomi/unilm/beit2/inference"
    # opts.output_dir="/home/xiaomi/unilm/beit2/inference_output"

    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)

    main(opts)

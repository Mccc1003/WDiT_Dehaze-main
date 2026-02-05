import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Evaluate Wavelet-Based Diffusion Model')
    parser.add_argument("--config", default='LOLv1.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default='ckpt/model.pth.tar', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--sampling_timesteps", type=int, default=30,
                        help="Number of implicit sampling steps")
    parser.add_argument("--image_folder", default='results/test', type=str,
                        help="Location to save restored images")
    parser.add_argument('--seed', default=230, type=int, metavar='N',
                        help='Seed for initializing training (default: 230)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config

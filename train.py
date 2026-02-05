import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Wavelet-Based Diffusion Model')
    parser.add_argument("--config", default='LOLv1.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--sampling_timesteps", type=int, default=200,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument("--image_folder", default='results/', type=str,
                        help="Location to save restored validation image patches")
    parser.add_argument('--seed', default=230, type=int, metavar='N',
                        help='Seed for initializing training (default: 230)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


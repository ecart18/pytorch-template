# -*- coding: utf-8 -*-
# Copyright © 2019 tao.hu <ecart.hut@gmail.com>

import os
import sys
import os.path as osp
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(CURRENT_DIR)
sys.path.append(osp.join(CURRENT_DIR, '..'))

from template.config import cfg
from template.model import ModelBuilder
from template.utils.osutils import load_params


def load_model(cfg):
    CUDA = torch.cuda.is_available()
    if CUDA:
        cudnn.benchmark = True
    device = 'cuda' if CUDA else 'cpu'
    model = ModelBuilder(cfg)
    model = load_params(model, cfg.Model_path).to(device).eval()
    return model


def test(args):
    cfg.merge_from_file(args.cfg)

    model = load_model(cfg)
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument('--cfg', type=str, default='config.yaml',
                        help='configuration of test')
    test(parser.parse_args())

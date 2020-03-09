# -*- coding: utf-8 -*-
# Copyright © 2019 tao.hu <ecart.hut@gmail.com>

from __future__ import print_function, absolute_import

import argparse
import sys
import os.path as osp
import torch
from torch import nn
from torch.backends import cudnn

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(CURRENT_DIR)
sys.path.append(osp.join(CURRENT_DIR, '..'))

from template.config import cfg
from template.model import ModelBuilder
from template.trainer import ScoreTrainer
from template.evaluator import ScoreEvaluator
from template.dataset import build_dataloader
from template.model.lr_scheduler import build_lr_scheduler
from template.utils.logger import Logger
from template.utils.osutils import save_checkpoint, load_checkpoint
from template.utils.env import set_random_seed


def train(args):
    cfg.merge_from_file(args.cfg)
    set_random_seed(cfg.TRAIN.Seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    sys.stdout = Logger(osp.join(cfg.IO.LOG_DIR, 'log.txt'))

    # Create data loaders
    train_loader, val_loader = build_dataloader(**cfg.DATALOADER.Dataloader_cfg)

    model = ModelBuilder(cfg)
    model = nn.DataParallel(model).cuda()

    # Optimizer
    optimizer_name = cfg.TRAIN.OPT.TYPE.upper()
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), **cfg.TRAIN.OPT.SGD)
    elif optimizer_name == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), **cfg.TRAIN.OPT.ADAM)
    else:
        raise ValueError('no available optimizer name: {}'.format(optimizer_name))

    # Load from checkpoint
    start_epoch = 0
    best_loss = 1e8
    if cfg.TRAIN.Resume:
        print("load checkpoint file from {} \n".format(cfg.TRAIN.Resume))
        checkpoint = load_checkpoint(cfg.TRAIN.Resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_loss = checkpoint['best_loss']
        print("=> Start epoch {}  best loss {:.5f}"
              .format(start_epoch, best_loss))

    lr_scheduler = build_lr_scheduler(optimizer, config=cfg.TRAIN.LR,
                                      config_warmup=cfg.TRAIN.LR_WARMUP,
                                      warm_up=cfg.TRAIN.Warmup,
                                      warmup_epochs=cfg.TRAIN.LR_WARMUP.EPOCH,
                                      total_epochs=cfg.TRAIN.Epochs,
                                      last_epoch=-1)
    lr = lr_scheduler.get_cur_lr()
    print('current epoch learning rate is: {:.5f} \n'.format(lr))

    # Trainer
    trainer = ScoreTrainer(cfg, model)
    # Evaluator
    evaluator = ScoreEvaluator(cfg, model)

    # Schedule learning rate
    def adjust_lr(epoch):
        lr_scheduler.step(epoch)
        lr = lr_scheduler.get_cur_lr()
        print('current epoch learning rate is: {:.5f} \n'.format(lr))

    # Start training
    for epoch in range(start_epoch, cfg.TRAIN.Epochs):
        if epoch > start_epoch:
            adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer, print_freq=10)
        if epoch < cfg.TRAIN.Start_save_epoch:
            continue

        loss = evaluator.eval(epoch, val_loader, print_freq=10)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if local_rank == 0:
            save_checkpoint(epoch, {
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, save_path=osp.join(cfg.IO.LOG_DIR, 'encoder'))

        print('\n * Finished epoch {:3d}  loss: {:.4f}  best: {:.4f}{}\n'.
              format(epoch, loss, best_loss, ' *' if is_best else ''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="template code")
    parser.add_argument('--cfg', type=str, default='config.yaml',
                        help='configuration of model')
    train(args=parser.parse_args())


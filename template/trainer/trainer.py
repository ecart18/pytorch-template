# -*- coding: utf-8 -*-
# Copyright © 2019 tao.hu <ecart.hut@gmail.com>

from __future__ import print_function, absolute_import

import time
import torch
from torch.autograd import Variable
from template.utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, cfg, model):
        super(BaseTrainer, self).__init__()
        self.cfg = cfg
        self.model = model

    def train(self, epoch, data_loader, optimizer, print_freq=10):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs = self._parse_data(inputs)
            loss = self._forward(inputs)
            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            if self.cfg.TRAIN.GRAD.Grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.TRAIN.GRAD.Max_grad)
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs):
        raise NotImplementedError


class ScoreTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        inputs['exp_data'] = Variable(inputs['patch_seqs']).cuda(non_blocking=True)
        return inputs

    def _forward(self, inputs):
        loss = self.model(inputs)
        loss = torch.sum(loss) / (loss.size(0))
        return loss


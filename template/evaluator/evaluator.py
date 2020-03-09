# -*- coding: utf-8 -*-
# Copyright © 2019 tao.hu <ecart.hut@gmail.com>

from __future__ import print_function, absolute_import

import time
import torch
from torch.autograd import Variable
from template.utils.meters import AverageMeter


class BaseEvaluator(object):
    def __init__(self, cfg, model):
        super(BaseEvaluator, self).__init__()
        self.cfg = cfg
        self.model = model

    def eval(self, epoch, data_loader, print_freq=10):
        self.model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        with torch.no_grad():
            end = time.time()
            for i, inputs in enumerate(data_loader):
                data_time.update(time.time() - end)

                inputs = self._parse_data(inputs)
                loss = self._forward(inputs)
                losses.update(loss.item())

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
        return losses.avg

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs):
        raise NotImplementedError


class ScoreEvaluator(BaseEvaluator):
    def _parse_data(self, inputs):
        inputs['exp_data'] = Variable(inputs['exp_data']).cuda(non_blocking=True)
        return inputs

    def _forward(self, inputs):
        loss = self.model(inputs)
        loss = torch.sum(loss) / (loss.size(0))
        return loss

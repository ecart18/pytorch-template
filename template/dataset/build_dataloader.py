# -*- coding: utf-8 -*-
# Copyright © 2019 tao.hu <ecart.hut@gmail.com>

from __future__ import absolute_import
import torch
from template.dataset.dataset_example import DatasetExample as Dataset


def build_dataloader(batch_size=16, num_workers=16, gpu_num=4):

    train_dataloader = Dataset(dataset_type='train')

    train_dataloader = torch.utils.data.DataLoader(train_dataloader,
                                                   batch_size=batch_size * gpu_num,
                                                   num_workers=num_workers,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   drop_last=True)

    val_dataloader = Dataset(dataset_type='val')
    val_dataloader = torch.utils.data.DataLoader(val_dataloader,
                                                 batch_size=batch_size * gpu_num,
                                                 num_workers=num_workers,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 drop_last=True)
    return train_dataloader, val_dataloader
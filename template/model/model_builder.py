# -*- coding: utf-8 -*-
# Copyright © 2019 tao.hu <ecart.hut@gmail.com>

import torch.nn as nn
from template.model.backbone import get_backbone


class ModelBuilder(nn.Module):

    def __init__(self, cfg):
        super(ModelBuilder, self).__init__()

        self.backbone = get_backbone(name=cfg.BACKBONE.Name, **cfg.BACKBONE.Backbone_cfgs)


    def forward(self, inputs):

        return self.backbone(inputs)




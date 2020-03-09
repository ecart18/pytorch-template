# -*- coding: utf-8 -*-
# Copyright © 2019 tao.hu <ecart.hut@gmail.com>

from __future__ import absolute_import

import os.path as osp


class DatasetExample(object):
    def __init__(self):
        self.len = None

    def __len__(self):
        # get the length of all the matching frame
        return self.len

    def __getitem__(self, item):
        raise NotImplementedError




from __future__ import absolute_import

import os.path as osp
from template.model.weights_init import init_weights, load_pretrain_state_dict
from .resnet import resnet18, resnet50, resnet101


__BACKBONES = {
  'resnet18': resnet18,
  'resnet50': resnet50,
  'resnet101': resnet101,
}

resnet_model_path = {
    'resnet18':  'pretrainmodel/resnet18-5c106cde.pth',  # ''https://download.pytorch.org/models/resnet18-5c106cde.pth'
    'resnet34':  'pretrainmodel/resnet34-333f7ec4.pth',  # 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    'resnet50':  'pretrainmodel/resnet50-19c8e357.pth',  # 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    'resnet101': 'pretrainmodel/resnet101-5d3b4d8f.pth', # 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
  }


def get_backbone(name='resnet50', pretrained=True, need_grad=True, **kwargs):
    """
    :param name: (str): model name
    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param need_grad: (bool): If True, set require grad
    :param kwargs: other parameters
    :return: backbone model
    """
    if name not in __BACKBONES:
        raise KeyError("Unknown backbone structure:", name)
    backbone = __BACKBONES[name](**kwargs)
    if pretrained:
        backbone = load_pretrain_state_dict(backbone, name=name,
                                            model_path=osp.join('../..', resnet_model_path[name]))
        if not need_grad:
            for param in backbone.parameters():
                param.requires_grad = False
            print('Gradients of backbone Network {0} are not required. \n'.format(name))
    else:
        backbone = init_weights(backbone)
    return backbone

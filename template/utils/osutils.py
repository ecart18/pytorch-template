# -*- coding: utf-8 -*-
# Copyright © 2019 tao.hu <ecart.hut@gmail.com>

import os
import errno
import torch
import os.path as osp
import shutil


def mkdir_p(path):
    """mimic the behavior of mkdir -p in bash"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def deep_copy_dict(tranformer_output, copy_batch):
    if tranformer_output is None:
        return tranformer_output
    copy_dict = dict()
    for layer_key in tranformer_output.keys():
        list_len = len(tranformer_output[layer_key])
        copy_dict[layer_key] = [None] * list_len
        for idx in range(list_len):
            copy_dict[layer_key][idx] = dict()
            for tensor_key in tranformer_output[layer_key][idx].keys():
                ts = tranformer_output[layer_key][idx][tensor_key]
                ts_shape = ts.shape  # head * batch * dimension
                copy_dict[layer_key][idx][tensor_key] = \
                    ts.data.clone().repeat(1, copy_batch, 1).view(copy_batch*ts_shape[0], ts_shape[1], ts_shape[2])
    return copy_dict


def save_checkpoint(epoch, state, is_best, save_path, save_per_epoch=False):
    if not osp.isdir(save_path):
        mkdir_p(save_path)
    filename = osp.join(save_path, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osp.join(save_path, 'model_best.pth.tar'))
    if save_per_epoch:
        shutil.copyfile(filename, osp.join(save_path, 'model_epoch_' + str(epoch) + '.pth.tar'))


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # filter 'num_batches_tracked'
    missing_keys = [x for x in missing_keys
                    if not x.endswith('num_batches_tracked')]
    if len(missing_keys) > 0:
        print('[Warning] missing keys: {}'.format(missing_keys))
    if len(unused_pretrained_keys) > 0:
        print('[Warning] unused_pretrained_keys: {}'.format(unused_pretrained_keys))
    assert len(used_pretrained_keys) > 0, \
        'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters
    share common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_params(model, pretrained_path):
    print('load pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path,
        map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'],
                                        'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    assert check_keys(model, pretrained_dict), 'load NONE from pretrained checkpoint'
    model.load_state_dict(pretrained_dict, strict=False)
    return model


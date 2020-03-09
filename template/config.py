from __future__ import absolute_import

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "Template"

# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

__C.BACKBONE.Name = "resnet50"

__C.BACKBONE.Backbone_cfgs = CN(new_allowed=True)


# ------------------------------------------------------------------------ #
# DATALOADER options
# ------------------------------------------------------------------------ #
__C.DATALOADER = CN()

__C.DATALOADER.Dataloader_cfg = CN(new_allowed=True)



# ------------------------------------------------------------------------ #
# TRAIN options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

__C.TRAIN.Seed = 1

__C.TRAIN.Resume = ""

__C.TRAIN.Epochs = 20

__C.TRAIN.Start_save_epoch = 0

__C.TRAIN.Warmup = True

__C.TRAIN.GRAD = CN()

__C.TRAIN.GRAD.Grad_clip = True

__C.TRAIN.GRAD.Max_grad = 1e4

__C.TRAIN.OPT = CN(new_allowed=True)

__C.TRAIN.OPT.TYPE = "ADAM"

__C.TRAIN.OPT.ADAM = CN(new_allowed=True)

__C.TRAIN.OPT.SGD = CN(new_allowed=True)

__C.TRAIN.LR = CN(new_allowed=True)

__C.TRAIN.LR.POLICY = "exponential"

__C.TRAIN.LR_WARMUP = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP.TYPE = "step"

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# IO options
# ------------------------------------------------------------------------ #
__C.IO = CN(new_allowed=True)
__C.IO.LOG_DIR = "./job_data"

from .schedules import LinearStepScheduler
from .schedules import LogScheduler
from .schedules import StepScheduler
from .schedules import CosStepScheduler
from .schedules import MultiStepScheduler
from .schedules import WarmUPScheduler


__LRs = {
    'exponential': LogScheduler,
    'step': StepScheduler,
    'multi-step': MultiStepScheduler,
    'linear': LinearStepScheduler,
    'cos': CosStepScheduler
}

__all__ = [
    'build_lr_scheduler',
]


def _build_lr_scheduler(optimizer, config, epochs, last_epoch=-1):
    return __LRs[config.POLICY](optimizer, last_epoch=last_epoch,
                              epochs=epochs, **config.KWARGS)


def _build_warm_up_scheduler(optimizer, config_warmup, config, warmup_epochs=5, total_epochs=50, last_epoch=-1):
    warmup_epoch = warmup_epochs
    sc1 = _build_lr_scheduler(optimizer, config_warmup,
                              warmup_epoch, last_epoch)
    sc2 = _build_lr_scheduler(optimizer, config,
                              total_epochs - warmup_epoch, last_epoch)
    return WarmUPScheduler(optimizer, sc1, sc2, total_epochs, last_epoch)


def build_lr_scheduler(optimizer, config, config_warmup, warm_up=True,
                       warmup_epochs=5, total_epochs=50, last_epoch=-1):
    if warm_up:
        return _build_warm_up_scheduler(optimizer, config_warmup, config,
                                        warmup_epochs, total_epochs, last_epoch)
    else:
        return _build_lr_scheduler(optimizer, config,
                                   total_epochs, last_epoch)

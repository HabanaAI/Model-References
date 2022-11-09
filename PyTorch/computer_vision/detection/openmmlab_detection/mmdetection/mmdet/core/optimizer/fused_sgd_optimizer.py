from mmcv.runner.optimizer import OPTIMIZERS
from habana_frameworks.torch.hpex.optimizers import FusedSGD

OPTIMIZERS.register_module(FusedSGD)
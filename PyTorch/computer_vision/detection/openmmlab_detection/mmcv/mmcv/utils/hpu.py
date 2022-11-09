# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
import os
import torch, mmcv
try:
    import habana_frameworks.torch.core as htcore
except:
    pass

# Singleton. First call must be passed in a valid meta
class HpuInfo(object):
    _instance = None

    @classmethod
    def is_hpu_enabled(cls):
        return HpuInfo._instance.hpu_enabled

    @classmethod
    def is_lazy_enabled(cls):
        return HpuInfo._instance.lazy_enabled

    @classmethod
    def is_hmp_enabled(cls):
        return HpuInfo._instance.hmp

    @classmethod
    def is_groundtruth_processing_on_cpu(cls):
        return HpuInfo._instance.groundtruth_processing_on_cpu

    def __new__(cls, hpu_enabled=False, lazy_enabled=False, hmp=False, hmp_level='O1', hmp_bf16='', hmp_fp32='', hmp_verbose=False, groundtruth_processing_on_cpu=True, *args, **kwargs):
        if cls._instance is None:
            print('Creating the HpuInfo object')
            if lazy_enabled:
                assert hpu_enabled, 'HPU must be enabled to have lazy mode'
            if hmp:
                assert hpu_enabled, 'HPU must be enabled to use hmp'
            cls._instance = super(HpuInfo, cls).__new__(cls, *args, **kwargs)
            cls._instance.hpu_enabled = hpu_enabled
            cls._instance.lazy_enabled = lazy_enabled
            cls._instance.hmp = hmp
            cls._instance.hmp_level = hmp_level
            cls._instance.hmp_bf16 = hmp_bf16
            cls._instance.hmp_fp32 = hmp_fp32
            cls._instance.hmp_verbose = hmp_verbose
            cls._instance.groundtruth_processing_on_cpu = groundtruth_processing_on_cpu
            if hpu_enabled:
                print('HPU is enabled')
                mode_str = ('Eager', 'Lazy')[lazy_enabled]
                print(f'{mode_str} mode is enabled')
                mode_num = ('2', '1')[lazy_enabled]
                os.environ["PT_HPU_LAZY_MODE"] = mode_num

                if hmp:
                    from habana_frameworks.torch.hpex import hmp
                    hmp.convert(opt_level=hmp_level, bf16_file_path=hmp_bf16,
                                fp32_file_path=hmp_fp32, isVerbose=hmp_verbose)
        return cls._instance

# must be first thing called
def register_hpuinfo(hpu_enabled, lazy_enabled, hmp=False, hmp_level='O1', hmp_bf16='', hmp_fp32='', hmp_verbose=False, groundtruth_processing_on_cpu=True):
    HpuInfo(hpu_enabled, lazy_enabled, hmp, hmp_level, hmp_bf16, hmp_fp32, hmp_verbose, groundtruth_processing_on_cpu)

def is_hpu_enabled():
    return HpuInfo.is_hpu_enabled()

def is_lazy_enabled():
    return HpuInfo.is_lazy_enabled()

def is_hmp_enabled():
    return HpuInfo.is_hmp_enabled()

def groundtruth_processing_on_cpu():
    return is_hpu_enabled() and HpuInfo.is_groundtruth_processing_on_cpu()

def mark_step_if_needed():
    if HpuInfo.is_hpu_enabled() and HpuInfo.is_lazy_enabled():
        htcore.mark_step()

def move_to_device(obj, device, tdtype=None):
    if isinstance(obj, torch.Tensor):
        obj = obj.to(device)
        if tdtype != None:
            obj = obj.type(tdtype)
        return obj
    elif isinstance(obj, tuple):
        return tuple(move_to_device(k, device, tdtype) for k in obj)
    elif isinstance(obj, list):
        return [move_to_device(k, device, tdtype) for k in obj]
    elif isinstance(obj, dict):
        return {k:move_to_device(obj[k], device, tdtype) for k in obj}
    elif isinstance(obj, mmcv.parallel.data_container.DataContainer):
        return move_to_device(obj.data[0], device, tdtype)
    else:
        return obj

# moves stuff to HPU if possible. can recurse through list/tuple/dict and also handles DataContainer
def move_to_hpu(obj):
    if not is_hpu_enabled():
        return obj
    return move_to_device(obj, 'hpu')

def move_model_to_hpu_if_needed(model):
    if not is_hpu_enabled():
        return model
    else:
        model = model.to('hpu')
        return model

def dev_mode_epoch_step():
    return int(os.environ.get('DEV_MODE_EPOCH_STEP','-1'))

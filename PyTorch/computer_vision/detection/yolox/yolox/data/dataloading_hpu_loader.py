###########################################################################
# Copyright (C) 2025 Habana Labs, Ltd. an Intel Company
###########################################################################

# Customized dataloader based on the following module:
# - habana_dataloader/habana_dataset.py

import habana_frameworks.torch.utils.experimental as htexp

import os

import torch.distributed as dist
import torch.utils.data

# maybe consolidate to single file
from .dataloading_hpu_mediapipe import YoloxMediaPipe, YoloxPytorchIterator

# helper functions
def _is_distributed():
    return dist.is_available() and dist.is_initialized()

def _get_world_size():
    if _is_distributed():
        return dist.get_world_size()
    else:
        return 1

def _get_rank():
    if _is_distributed():
        return dist.get_rank()
    else:
        return 0

def isGaudi2(device):
    return device == htexp.synDeviceType.synDeviceGaudi2

def isGaudi3(device):
    return device == htexp.synDeviceType.synDeviceGaudi3

def _is_coco_dataset(dataset):
    try:
        if "COCO 2017 Dataset" in dataset.data["info"]["description"]:
            return True
    except:
        return False
    return False

class YoloxDataLoader():
    def __init__(self, dataset=None, **kwargs):
        self.dataset = dataset

        # check for some required arguments
        required_args = ["batch_size", "drop_last"]
        for arg in required_args:
            if arg not in kwargs:
                raise TypeError(f"{arg} is required in kwargs")

        # any kwargs not processed here will be ignored
        self.batch_size      = kwargs.get("batch_size", 0)
        self.shuffle         = kwargs.get("shuffle", False)
        self.drop_last       = kwargs.get("drop_last", False)
        self.prefetch_factor = kwargs.get("prefetch_factor", 3)
        self.pad_last_batch  = kwargs.get('pad_last_batch', True)

        # do basic validation
        if (self.batch_size <= 0):
            raise Value(f"invalid batch_size: {self.batch_size}")

        # setting prefetch_factor to 0 will reduce overall throughput because image decode will not parallelize with running model
        if (self.prefetch_factor < 0):
            raise Value(f"invalid prefetch_factor: {self.prefetch_factor}")

        # assume device is Gaudi2 or Gaudi3 (already checked by caller)
        media_device_type = "legacy"
        num_instances = _get_world_size()
        instance_id = _get_rank()

        pipeline = YoloxMediaPipe(
            a_torch_transforms = self.dataset.transform,
            a_root             = self.dataset.img_folder,
            a_annotation_file  = self.dataset.annotate_file,
            a_batch_size       = self.batch_size,
            a_shuffle          = self.shuffle,
            a_drop_last        = self.drop_last,
            a_prefetch_count   = self.prefetch_factor,
            a_num_instances    = num_instances,
            a_instance_id      = instance_id,
            a_device           = media_device_type,
        )

        self.iterator = YoloxPytorchIterator(mediapipe=pipeline, pad_last_batch=self.pad_last_batch)

    def __iter__(self):
        return iter(self.iterator)

    def __len__(self):
        return len(self.iterator)

class MediaPipeDataLoader():
    def __init__(self, dataset=None, **kwargs):
        # dataset should be oftype COCODataset
        self.dataset = dataset
        self.batch_size = kwargs.get('batch_size')

        if not dataset:
            raise ValueError("dataset must be provided")
        
        if not _is_coco_dataset(dataset):
            raise ValueError("only COCO dataset is supported")

        self.DeviceType = htexp._get_device_type()
        if not (isGaudi2(self.DeviceType) or isGaudi3(self.DeviceType)):
            raise ValueError("only Gaudi2 or Gaudi3 are supported")

        # create the YoloxDataLoader dataloader
        try:
            self.dataloader = YoloxDataLoader(self.dataset, **kwargs)

        except Exception as e:
            fallback_enabled = os.getenv("DATALOADER_FALLBACK_EN", 0)
            print(f"Failed to initialize YoloxDataLoader, error: {str(e)}")
            if fallback_enabled:
                # Fallback to PT Dataloader
                print(f"Creating default DataLoader...")
                self.dataloader = torch.utils.data.DataLoader(self.dataset, **kwargs)
            else:
                print(f"Fallback to default DataLoader is disabled. To enable, set environment variable DATALOADER_FALLBACK_EN=1.")
                raise

    def __iter__(self):
        self.iter = iter(self.dataloader)
        return self

    def __next__(self):
        return next(self.iter)

    def __len__(self):
        return len(self.dataloader)

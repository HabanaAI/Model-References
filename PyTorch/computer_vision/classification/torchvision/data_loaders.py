# Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company

import os
from enum import Enum

import torch.utils.data
import torch.distributed as dist

import utils


class DataLoaderType(Enum):
    """
    Enum class defining types of data loaders.

    """
    Python = 0
    Aeon = 1
    MediaAPI = 2


class AeonDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, sampler, batch_size, num_workers, pin_memory=True, pin_memory_device=None):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device

        torch_transforms = self.dataset.transform
        aeon_data_dir = self.dataset.root
        use_prefetch = True
        channels_last = False
        drop_last = True

        from habana_dataloader.aeon_config import get_aeon_config
        from habana_dataloader.aeon_transformers import HabanaAeonTransforms
        from habana_dataloader.aeon_manifest import generate_aeon_manifest
        import habana_dataloader.habana_dl_app

        ht = HabanaAeonTransforms(torch_transforms)
        aeon_transform_config, is_train = ht.get_aeon_transforms()
        manifest_filename = generate_aeon_manifest(self.dataset.imgs)
        aeon_config_json = get_aeon_config(aeon_data_dir, manifest_filename, aeon_transform_config,
                                           self.batch_size, self.num_workers, channels_last, is_train)
        self.aeon = habana_dataloader.habana_dl_app.HabanaAcceleratedPytorchDL(aeon_config_json, pin_memory,
                                                                               use_prefetch, channels_last, drop_last)
        print("Running with Habana aeon DataLoader")

    def __len__(self):
        return len(self.aeon)

    def __iter__(self):
        return iter(self.aeon)


class MediaApiDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, sampler, batch_size, num_workers, pin_memory=True, pin_memory_device=None, is_training=False, seed=None):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size

        self.shuffle = (isinstance(self.sampler, torch.utils.data.RandomSampler) or
                       (isinstance(self.sampler, torch.utils.data.distributed.DistributedSampler) and (self.sampler.shuffle == True)))

        root = self.dataset.root
        num_instances = utils.get_world_size()
        instance_id = utils.get_rank()
        queue_depth = 3

        from resnet_media_pipe import ResnetMediaPipe
        pipeline = ResnetMediaPipe(is_training=is_training, root=root, batch_size=batch_size,
                                   shuffle=self.shuffle, drop_last=False, queue_depth=queue_depth,
                                   num_instances=num_instances, instance_id=instance_id, device="legacy", seed=seed)

        from habana_frameworks.mediapipe.plugins.iterator_pytorch import HPUResnetPytorchIterator
        self.iterator = HPUResnetPytorchIterator(mediapipe=pipeline)
        print("Running with Media API DataLoader")

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        return iter(self.iterator)


def choose_data_loader(dl_worker_type = "HABANA"):
    if dl_worker_type == "MP":
        return DataLoaderType.Python

    if utils.is_gaudi():
        return DataLoaderType.Aeon

    try:
        from habana_frameworks.mediapipe.mediapipe import MediaPipe
        return DataLoaderType.MediaAPI
    except (ImportError) as e:
        return DataLoaderType.Aeon


def build_data_loader(is_training, dl_worker_type, seed=None, **kwargs):
    data_loader_type = choose_data_loader(dl_worker_type)
    use_fallback = False

    try:
        if data_loader_type == DataLoaderType.MediaAPI:
            return MediaApiDataLoader(**kwargs, is_training=is_training, seed=seed)
        elif data_loader_type == DataLoaderType.Aeon:
            return AeonDataLoader(**kwargs)
    except Exception as e:
        if os.getenv('DATALOADER_FALLBACK_EN', "True") == "True":
            print(f"Failed to initialize Habana Dataloader, error: {str(e)}\nRunning with PyTorch Dataloader")
            return torch.utils.data.DataLoader(**kwargs)
        else:
            print(f"Habana dataloader configuration failed: {e}")
            raise

    if data_loader_type == DataLoaderType.Python:
        return torch.utils.data.DataLoader(**kwargs)
    else:
        raise ValueError(f"Unknown data_loader_type {data_loader_type}")

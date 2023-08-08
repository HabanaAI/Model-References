
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company

import os
from typing import Any, Optional
import torch
import torchvision.transforms as transforms
import pytorch_lightning as pl
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE
elif module_available("pytorch_lightning"):
    from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    import torchvision.datasets
    from torchvision import transforms as transform_lib

def configure_torch_datamodule(traindir, valdir, train_transforms, val_transforms):  # type: ignore[no-untyped-def]
    # check supported transforms
    if train_transforms is not None:
        for t in train_transforms:
            if (
                isinstance(t, transform_lib.RandomResizedCrop)
                or isinstance(t, transform_lib.RandomHorizontalFlip)
                or isinstance(t, transform_lib.ToTensor)
                or isinstance(t, transform_lib.Normalize)
            ):

                continue
            else:
                raise ValueError("Unsupported train transform: " + str(type(t)))

        train_transforms = transform_lib.Compose(train_transforms)

    if val_transforms is not None:
        for t in val_transforms:
            if (
                isinstance(t, transform_lib.Resize)
                or isinstance(t, transform_lib.CenterCrop)
                or isinstance(t, transform_lib.ToTensor)
                or isinstance(t, transform_lib.Normalize)
            ):

                continue
            else:
                raise ValueError("Unsupported val transform: " + str(type(t)))

        val_transforms = transform_lib.Compose(val_transforms)

    if "imagenet" not in traindir.lower() and "ilsvrc2012" not in traindir.lower():
        raise ValueError("Habana dataloader only supports Imagenet dataset")

    dataset_train = torchvision.datasets.ImageFolder(traindir, train_transforms)

    dataset_val = torchvision.datasets.ImageFolder(valdir, val_transforms)

    return dataset_train, dataset_val


def get_dataset_configs(datapath: str):
    data_path = datapath
    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    val_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
    return train_dir, val_dir, normalize, train_transforms, val_transforms


def get_data_module(data_path, dl_type, workers, batch_size, hpus):
    train_dir, val_dir, normalize, train_transforms, val_transforms = get_dataset_configs(data_path)

    data_module_type = TorchDataModule
    if dl_type=="HABANA":
        try:
            #from habana_lightning_plugins.datamodule import HPUDataModule
            from lightning_habana.pytorch.datamodule import HPUDataModule
            data_module_type=HPUDataModule
        except Exception as e:
            if os.getenv('DATALOADER_FALLBACK_EN', "True") == "True":
                print(f"Failed to initialize Habana Dataloader, error: {str(e)}\nRunning with PyTorch Dataloader")

    return data_module_type(
            train_dir=train_dir,
            val_dir=val_dir,
            num_workers=workers,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            train_transforms = train_transforms,
            val_transforms = val_transforms,
            normalize = normalize,
            #dl_type=dl_type,
            distributed=True if (hpus > 1) else False,
        )


class TorchDataModule(pl.LightningDataModule):
    name = "torch-dataset"

    def __init__(
        self,
        train_dir: str = "",
        val_dir: str = "",
        num_workers: int = 8,
        normalize: bool = False,
        seed: int = 42,
        batch_size: int = 32,
        train_transforms: Any = None,
        val_transforms: Any = None,
        pin_memory: bool = True,
        shuffle: bool = False,
        drop_last: bool = True,
        dl_type = "MP",
        distributed= False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.num_workers = num_workers
        self.normalize = normalize
        self.seed = seed
        self.batch_size = batch_size
        self.train_transform = train_transforms
        self.val_transform = val_transforms
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.distributed = distributed

    def setup(self, stage: Optional[str] = None):  # type: ignore[no-untyped-def]
        if not _TORCHVISION_AVAILABLE:
            raise ValueError("torchvision transforms not available")

        self.dataset_train, self.dataset_val = configure_torch_datamodule (
            self.train_dir, self.val_dir, self.train_transform, self.val_transform
        )

        dataset = self.dataset_train if stage == "fit" else self.dataset_val
        if dataset is None:
            raise TypeError("Error creating dataset")

    def train_dataloader(self):  # type: ignore[no-untyped-def]
        """train set removes a subset to use for validation."""
        sampler_train = torch.utils.data.distributed.DistributedSampler(self.dataset_train) if self.distributed else torch.utils.data.RandomSampler(self.dataset_train)
        loader = torch.utils.data.DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            sampler=sampler_train,
        )
        return loader

    def val_dataloader(self):  # type: ignore[no-untyped-def]
        """val set uses a subset of the training set for validation."""
        sampler_eval = torch.utils.data.distributed.DistributedSampler(self.dataset_val) if self.distributed else torch.utils.data.SequentialSampler(self.dataset_val)
        loader = torch.utils.data.DataLoader(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            sampler=sampler_eval,
        )
        return loader

    def test_dataloader(self):  # type: ignore[no-untyped-def]
        """test set uses the test split."""
        sampler_test = torch.utils.data.distributed.DistributedSampler(self.dataset_val) if self.distributed else torch.utils.data.SequentialSampler(self.dataset_val)
        loader = torch.utils.data.DataLoader(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            sampler=sampler_test,
        )
        return loader

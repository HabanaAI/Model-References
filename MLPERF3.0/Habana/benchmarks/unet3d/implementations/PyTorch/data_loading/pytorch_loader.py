###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import random
import torch
import numpy as np
import scipy.ndimage

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


def get_train_transforms():
    rand_flip = RandFlip()
    cast = Cast(types=(np.float32, np.uint8))
    rand_scale = RandomBrightnessAugmentation(factor=0.3, prob=0.1)
    rand_noise = GaussianNoise(mean=0.0, std=0.1, prob=0.1)
    train_transforms = transforms.Compose([rand_flip, cast, rand_scale, rand_noise])
    return train_transforms


class RandBalancedCrop:
    def __init__(self, patch_size, oversampling):
        self.patch_size = patch_size
        self.oversampling = oversampling

    def __call__(self, data):
        image, label = data["image"], data["label"]
        if random.random() < self.oversampling:
            image, label, cords = self.rand_foreg_cropd(image, label)
        else:
            image, label, cords = self._rand_crop(image, label)
        data.update({"image": image, "label": label})
        return data

    @staticmethod
    def randrange(max_range):
        return 0 if max_range == 0 else random.randrange(max_range)

    def get_cords(self, cord, idx):
        return cord[idx], cord[idx] + self.patch_size[idx]

    def _rand_crop(self, image, label):
        ranges = [s - p for s, p in zip(image.shape[1:], self.patch_size)]
        cord = [self.randrange(x) for x in ranges]
        low_x, high_x = self.get_cords(cord, 0)
        low_y, high_y = self.get_cords(cord, 1)
        low_z, high_z = self.get_cords(cord, 2)
        image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
        label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
        return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]

    def rand_foreg_cropd(self, image, label):
        def adjust(foreg_slice, patch_size, label, idx):
            diff = patch_size[idx - 1] - (foreg_slice[idx].stop - foreg_slice[idx].start)
            sign = -1 if diff < 0 else 1
            diff = abs(diff)
            ladj = self.randrange(diff)
            hadj = diff - ladj
            low = max(0, foreg_slice[idx].start - sign * ladj)
            high = min(label.shape[idx], foreg_slice[idx].stop + sign * hadj)
            diff = patch_size[idx - 1] - (high - low)
            if diff > 0 and low == 0:
                high += diff
            elif diff > 0:
                low -= diff
            return low, high

        cl = np.random.choice(np.unique(label[label > 0]))
        foreg_slices = scipy.ndimage.find_objects(scipy.ndimage.measurements.label(label == cl)[0])
        foreg_slices = [x for x in foreg_slices if x is not None]
        slice_volumes = [np.prod([s.stop - s.start for s in sl]) for sl in foreg_slices]
        slice_idx = np.argsort(slice_volumes)[-2:]
        foreg_slices = [foreg_slices[i] for i in slice_idx]
        if not foreg_slices:
            return self._rand_crop(image, label)
        foreg_slice = foreg_slices[random.randrange(len(foreg_slices))]
        low_x, high_x = adjust(foreg_slice, self.patch_size, label, 1)
        low_y, high_y = adjust(foreg_slice, self.patch_size, label, 2)
        low_z, high_z = adjust(foreg_slice, self.patch_size, label, 3)
        image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
        label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
        return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]


class RandFlip:
    def __init__(self):
        self.axis = [1, 2, 3]
        self.prob = 1 / len(self.axis)

    def flip(self, data, axis):
        data["image"] = np.flip(data["image"], axis=axis).copy()
        data["label"] = np.flip(data["label"], axis=axis).copy()
        return data

    def __call__(self, data):
        for axis in self.axis:
            if random.random() < self.prob:
                data = self.flip(data, axis)
        return data


class Cast:
    def __init__(self, types):
        self.types = types

    def __call__(self, data):
        data["image"] = data["image"].astype(self.types[0])
        data["label"] = data["label"].astype(self.types[1])
        return data


class RandomBrightnessAugmentation:
    def __init__(self, factor, prob):
        self.prob = prob
        self.factor = factor

    def __call__(self, data):
        image = data["image"]
        if random.random() < self.prob:
            factor = np.random.uniform(low=1.0 - self.factor, high=1.0 + self.factor, size=1)
            image = (image * (1 + factor)).astype(image.dtype)
            data.update({"image": image})
        return data


class GaussianNoise:
    def __init__(self, mean, std, prob):
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, data):
        image = data["image"]
        if random.random() < self.prob:
            scale = np.random.uniform(low=0.0, high=self.std)
            noise = np.random.normal(loc=self.mean, scale=scale, size=image.shape).astype(image.dtype)
            data.update({"image": image + noise})
        return data


class SyntheticDataset(Dataset):
    def __init__(self, channels_in=1, channels_out=3, shape=(128, 128, 128),
                 device="cpu", oversampling=0.4, augment=False):
        shape = tuple(shape)
        x_shape = (channels_in,) + shape

        self.size = 168
        self.x = torch.rand((self.size, *x_shape), dtype=torch.float32, device=device, requires_grad=False)
        self.y = torch.randint(low=0, high=channels_out - 1, size=(self.size, *shape), dtype=torch.int32,
                               device=device, requires_grad=False)
        self.y = torch.unsqueeze(self.y, dim=1)

        self.augment = augment
        self.train_transforms, self.rand_crop = None, None
        if self.augment:
            self.train_transforms = get_train_transforms()
            self.rand_crop = RandBalancedCrop(patch_size=shape, oversampling=oversampling)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        data = {"image": self.x[idx % self.size], "label": self.y[idx % self.size]}
        if self.augment:
            data["image"], data["label"] = data["image"].numpy(), data["label"].numpy()
            data = self.rand_crop(data)
            data = self.train_transforms(data)
        return data


class PytTrain(Dataset):
    def __init__(self, images, labels, patch_size, oversampling):
        self.images, self.labels = images, labels
        self.train_transforms = get_train_transforms()
        self.rand_crop = RandBalancedCrop(patch_size=patch_size, oversampling=oversampling)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data = {"image": np.load(self.images[idx]), "label": np.load(self.labels[idx])}
        data = self.rand_crop(data)
        data = self.train_transforms(data)
        return data


class PytVal(Dataset):
    def __init__(self, images, labels):
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": np.load(self.images[idx]), "label": np.load(self.labels[idx])}


def get_dataset(flags, image_files, label_files, train_mode: bool = False) -> Dataset:
    if image_files is None or label_files is None:
        if train_mode:
            dataset = SyntheticDataset(shape=flags.input_shape,
                                       oversampling=flags.oversampling,
                                       augment=True)
        else:
            dataset = SyntheticDataset(shape=flags.val_input_shape)
    else:
        if train_mode:
            dataset = PytTrain(image_files, label_files,
                               patch_size=flags.input_shape, oversampling=flags.oversampling)
        else:
            dataset = PytVal(image_files, label_files)

    return dataset


def get_pytorch_loader(flags, image_files: list = None, label_files: list = None,
                       train_mode: bool = True, num_shards: int = 1, batch_size: int = 1) -> DataLoader:
    dataset = get_dataset(flags, image_files, label_files, train_mode)
    sampler = None
    if train_mode and num_shards > 1:
        sampler = DistributedSampler(dataset, seed=flags.seed, drop_last=True)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=not flags.benchmark and sampler is None,
                            sampler=sampler,
                            num_workers=flags.num_workers,
                            pin_memory=True,
                            pin_memory_device=flags.device,
                            persistent_workers=True,
                            drop_last=train_mode)
    return dataloader

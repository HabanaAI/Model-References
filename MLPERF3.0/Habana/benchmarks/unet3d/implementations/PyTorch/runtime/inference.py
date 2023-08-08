###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import numpy as np
from scipy import signal
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import habana_frameworks.torch.core as htcore

from runtime.distributed_utils import reduce_tensor, get_world_size, get_rank


def evaluate(flags, model, loader, loss_fn, score_fn, device, epoch=0, is_distributed=False, sw_inference=None, warmup=False):
    rank = get_rank()
    world_size = get_world_size()

    if flags.load_ckpt_path:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(flags.load_ckpt_path, map_location=map_location)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['best_model_state_dict'])
        if is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[flags.local_rank],
                                                              output_device=flags.local_rank)

    model.eval()

    scores = []
    is_cached = len(sw_inference.cache) > 0
    loader = sw_inference.cache if is_cached else loader
    sw_inference_fn = sw_inference.inference_from_cache if is_cached else sw_inference.inference

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, disable=(rank != 0) or not flags.verbose)):
            image, label = batch["image"], batch["label"]

            numel = sw_inference.cache[i]["numel"] if is_cached else image.numel()
            if numel == 0:
                continue
            with autocast(enabled=flags.amp):
                output, label = sw_inference_fn(
                    inputs=image,
                    labels=label,
                    model=model,
                    device=device,
                    index=i,
                    warmup=warmup
                )
                if flags.device == "hpu":
                    htcore.mark_step()
                scores.append(score_fn(output, label))
            if flags.device == "hpu":
                htcore.mark_step()
            del output
            del label

    scores = reduce_tensor(torch.mean(torch.stack(scores, dim=0), dim=0), world_size).cpu().numpy()

    eval_metrics = {"epoch": epoch,
                    "L1 dice": scores[-2],
                    "L2 dice": scores[-1],
                    "mean_dice": (scores[-1] + scores[-2]) / 2}

    return eval_metrics


def pad_input(volume, roi_shape, strides, padding_mode, padding_val, dim=3):
    """
    mode: constant, reflect, replicate, circular
    """
    bounds = [(strides[i] - volume.shape[2:][i] % strides[i]) % strides[i] for i in range(dim)]
    bounds = [bounds[i] if (volume.shape[2:][i] + bounds[i]) >= roi_shape[i] else bounds[i] + strides[i]
              for i in range(dim)]
    paddings = [bounds[2] // 2, bounds[2] - bounds[2] // 2,
                bounds[1] // 2, bounds[1] - bounds[1] // 2,
                bounds[0] // 2, bounds[0] - bounds[0] // 2,
                0, 0,
                0, 0]

    return F.pad(volume, paddings, mode=padding_mode, value=padding_val), paddings


def gaussian_kernel(n, std):
    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    gaussian3D = np.outer(gaussian2D, gaussian1D)
    gaussian3D = gaussian3D.reshape(n, n, n)
    gaussian3D = np.cbrt(gaussian3D)
    gaussian3D /= gaussian3D.max()
    return torch.from_numpy(gaussian3D)


class SlidingwWindowInference:

    def __init__(self, roi_shape, overlap, mode, padding_val, device):
        self.cache = []
        self.norm_patch = None
        self.roi_shape = roi_shape
        self.overlap = overlap
        self.mode = mode
        self.padding_val = padding_val
        self.device = device
        self.precision = torch.float32

        if self.mode == "constant":
            self.norm_patch = torch.ones(size=self.roi_shape, dtype=self.precision, device=self.device)
        elif self.mode == "gaussian":
            self.norm_patch = gaussian_kernel(
                self.roi_shape[0], 0.125 * self.roi_shape[0]).type(self.precision).to(self.device)
        else:
            raise ValueError("Unknown mode. Available modes are {constant, gaussian}.")

    def inference(self, inputs, labels, model, device, padding_mode="constant",
                  index=-1, warmup=False, **kwargs):
        image_shape = list(inputs.shape[2:])
        dim = len(image_shape)
        strides = [int(self.roi_shape[i] * (1 - self.overlap)) for i in range(dim)]

        bounds = [image_shape[i] % strides[i] for i in range(dim)]
        bounds = [bounds[i] if bounds[i] < strides[i] // 2 else 0 for i in range(dim)]
        inputs = inputs[...,
                        bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
                        bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
                        bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2)]
        labels = labels[...,
                        bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
                        bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
                        bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2)]

        inputs, paddings = pad_input(inputs, self.roi_shape, strides, padding_mode, self.padding_val)

        padded_shape = inputs.shape[2:]
        result = torch.zeros(size=(1, 3, *padded_shape), dtype=inputs.dtype, device=inputs.device)
        norm_map = torch.zeros_like(result)
        result = result.to(device)
        norm_map = norm_map.to(device)
        labels = labels.to(device)

        size = [(padded_shape[i] - self.roi_shape[i]) // strides[i] + 1 for i in range(dim)]

        input_buffer = []
        for i in range(0, strides[0] * size[0], strides[0]):
            for j in range(0, strides[1] * size[1], strides[1]):
                for k in range(0, strides[2] * size[2], strides[2]):
                    input = inputs[
                        ...,
                        i:(self.roi_shape[0] + i),
                        j:(self.roi_shape[1] + j),
                        k:(self.roi_shape[2] + k)
                    ].to(device)
                    input_buffer.append(input)

        if not warmup:
            numel = inputs.numel()
            self.cache.append({"image": input_buffer, "label": labels,
                               "image_shape": image_shape, "padded_shape": padded_shape,
                               "paddings": paddings, "numel": numel, "result": result,
                               "norm_map": norm_map})

        output_buffer = []
        for i in range(len(input_buffer)):
            output_buffer.append(model(input_buffer[i]) * self.norm_patch)
            if device.type == "hpu":
                htcore.mark_step()

        result.zero_()
        norm_map.zero_()
        count = 0
        for i in range(0, strides[0] * size[0], strides[0]):
            for j in range(0, strides[1] * size[1], strides[1]):
                for k in range(0, strides[2] * size[2], strides[2]):
                    result[
                        ...,
                        i:(self.roi_shape[0] + i),
                        j:(self.roi_shape[1] + j),
                        k:(self.roi_shape[2] + k)] += output_buffer[count]
                    norm_map[
                        ...,
                        i:(self.roi_shape[0] + i),
                        j:(self.roi_shape[1] + j),
                        k:(self.roi_shape[2] + k)] += self.norm_patch
                    if device.type == "hpu":
                        htcore.mark_step()
                    count += 1

        # account for any overlapping sections
        # norm_map[norm_map == 0] = norm_map[norm_map > 0].min()
        result /= norm_map

        return result[
            ...,
            paddings[4]: image_shape[0] + paddings[4],
            paddings[2]: image_shape[1] + paddings[2],
            paddings[0]: image_shape[2] + paddings[0]
        ], labels

    def inference_from_cache(self, inputs, labels, model, device, padding_mode="constant",
                             index=-1, warmup=False, **kwargs):
        image_shape = self.cache[index]["image_shape"]
        dim = len(image_shape)
        strides = [int(self.roi_shape[i] * (1 - self.overlap)) for i in range(dim)]

        image_shape = self.cache[index]["image_shape"]
        padded_shape = self.cache[index]["padded_shape"]
        paddings = self.cache[index]["paddings"]
        result = self.cache[index]["result"]
        norm_map = self.cache[index]["norm_map"]

        size = [(padded_shape[i] - self.roi_shape[i]) // strides[i] + 1 for i in range(dim)]

        output_buffer = []
        for i in range(len(inputs)):
            output_buffer.append(model(inputs[i]) * self.norm_patch)
            if device.type == "hpu":
                htcore.mark_step()

        result.zero_()
        norm_map.zero_()
        count = 0
        for i in range(0, strides[0] * size[0], strides[0]):
            for j in range(0, strides[1] * size[1], strides[1]):
                for k in range(0, strides[2] * size[2], strides[2]):
                    result[
                        ...,
                        i:(self.roi_shape[0] + i),
                        j:(self.roi_shape[1] + j),
                        k:(self.roi_shape[2] + k)] += output_buffer[count]
                    norm_map[
                        ...,
                        i:(self.roi_shape[0] + i),
                        j:(self.roi_shape[1] + j),
                        k:(self.roi_shape[2] + k)] += self.norm_patch
                    if device.type == "hpu":
                        htcore.mark_step()
                    count += 1

        # account for any overlapping sections
        # norm_map[norm_map == 0] = norm_map[norm_map > 0].min()
        result /= norm_map

        return result[
            ...,
            paddings[4]: image_shape[0] + paddings[4],
            paddings[2]: image_shape[1] + paddings[2],
            paddings[0]: image_shape[2] + paddings[0]
        ], labels

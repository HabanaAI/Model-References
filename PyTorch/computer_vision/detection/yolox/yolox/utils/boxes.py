#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
###########################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###########################################################################

import numpy as np

import torch
import torchvision

__all__ = [
    "filter_box",
    "Postprocessor",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]

class Postprocessor(torch.nn.Module):
    def __init__(self, num_classes:int,
                    conf_thre: float = 0.7,
                    nms_thre: float = 0.45,
                    device: str = "cpu"
                ) -> None:
        super(Postprocessor, self).__init__()

        self.num_classes = num_classes
        self.conf_thre = conf_thre
        self.nms_thre = nms_thre
        self.class_agnostic = False
        self.image_indexes = torch.empty((0))
        self.device = device
        self.batch_size = 0
        self.bbox_num = 0

    @torch.jit.export
    def update_image_indexes(self):
        self.image_indexes = torch.tensor([i for i in range(self.batch_size)],
                dtype=torch.float,
                device=torch.device(self.device)
                    ).view(-1, 1
                        ).repeat(1, self.bbox_num)

    @torch.jit.export
    def check_image_indexes(self, batch_size : int, bbox_num : int):
        if self.image_indexes.size(0) == 0 or \
                self.batch_size != batch_size or \
                self.bbox_num != bbox_num:
            self.batch_size = batch_size
            self.bbox_num = bbox_num
            self.update_image_indexes()

    @torch.jit.export
    def get_image_indexes(self, batch_size : int, bbox_num : int):
        self.check_image_indexes(batch_size, bbox_num)
        return self.image_indexes

    def forward(self, prediction):
        bboxes = torch.stack((prediction[:, :, 0], prediction[:, :, 1],
                           prediction[:, :, 0], prediction[:, :, 1]), dim=-1)

        bboxes[:, :, 0] -= prediction[:, :, 2] / 2
        bboxes[:, :, 1] -= prediction[:, :, 3] / 2
        bboxes[:, :, 2] += prediction[:, :, 2] / 2
        bboxes[:, :, 3] += prediction[:, :, 3] / 2

        classifications = prediction[:, :, 5: 5 + self.num_classes]

        class_confidences, classes = torch.max(classifications, -1, keepdim=True)

        confidences = prediction[:, :, 4].unsqueeze(-1) * class_confidences
        detections = torch.cat((bboxes, confidences, classes), -1)

        # also updates self.batch_size and self.bbox_num if needed
        indexes = self.get_image_indexes(int(prediction.size(0)),
                                         int(prediction.size(1)))

        mask = confidences.squeeze(dim=-1) >= self.conf_thre
        detections = detections[mask]
        indexes = indexes[mask]

        output = []
        for batch_number in range(self.batch_size):
            i = int(torch.searchsorted(indexes, batch_number))
            if i == indexes.size(0):
                output.append(torch.empty((0)))
                continue

            j = int(torch.searchsorted(indexes, batch_number, right=True))
            detections_to_nms = detections[i:j,:]

            if self.class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections_to_nms[:, :4],
                    detections_to_nms[:, 4],
                    self.nms_thre,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections_to_nms[:, :4],
                    detections_to_nms[:, 4],
                    detections_to_nms[:, 5],
                    self.nms_thre,
                )

            detections_to_nms = detections_to_nms[nms_out_index]
            output.append(detections_to_nms)

        return output


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    bboxes = torch.zeros_like(prediction[:, :, :4])
    bboxes[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    bboxes[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    bboxes[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    bboxes[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2

    classifications = prediction[:, :, 5: 5 + num_classes]

    class_confidences, classes = torch.max(classifications, -1, keepdim=True)

    confidences = prediction[:, :, 4].unsqueeze(-1) * class_confidences
    batch_detections = torch.cat((bboxes, confidences, classes), -1)

    output = [None for _ in range(len(batch_detections))]
    for i, detections in enumerate(batch_detections):

        # If none are remaining => process next image
        if not detections.size(0):
            continue
        # Get score and class with highest confidence
        conf_mask = (detections[:, 4] >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4],
                detections[:, 5],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy : bool =True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)

    # type convertion for HPU: tl.type() -> tl.dtype
    en = (tl < br).type(tl.dtype).prod(dim=2)

    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes

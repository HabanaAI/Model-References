#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2022, Habana Labs Ltd.  All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, meshgrid

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv
from typing import List, Tuple

@torch.jit.script
def get_l1_target(l1_target, gt, stride, x_shifts, y_shifts, eps : float =1e-8):
    l1_target[:, 0] = gt[:, 0] / stride - x_shifts
    l1_target[:, 1] = gt[:, 1] / stride - y_shifts
    l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
    l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
    return l1_target

@torch.jit.script
def is_in_box(x_centers_per_image, y_centers_per_image, gt_bboxes_per_image):
    gt_bboxes_per_image_l = (
        (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
        .unsqueeze(1)
    )
    gt_bboxes_per_image_r = (
        (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
        .unsqueeze(1)
    )
    gt_bboxes_per_image_t = (
        (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
        .unsqueeze(1)
    )
    gt_bboxes_per_image_b = (
        (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
        .unsqueeze(1)
    )

    b_l = x_centers_per_image - gt_bboxes_per_image_l
    b_r = gt_bboxes_per_image_r - x_centers_per_image
    b_t = y_centers_per_image - gt_bboxes_per_image_t
    b_b = gt_bboxes_per_image_b - y_centers_per_image

    is_in_boxes = torch.min(torch.min(torch.min(b_l, b_t), b_r), b_b) > 0.0
    is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
    return (is_in_boxes, is_in_boxes_all)

@torch.jit.script
def get_in_boxes_info(
    gt_bboxes_per_image,
    expanded_strides,
    x_shifts,
    y_shifts,
):
    expanded_strides_per_image = expanded_strides[0]
    x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
    y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
    x_centers_per_image = (
        (x_shifts_per_image + 0.5 * expanded_strides_per_image)
        .unsqueeze(0)
    )  # [n_anchor] -> [1, n_anchor]
    y_centers_per_image = (
        (y_shifts_per_image + 0.5 * expanded_strides_per_image)
        .unsqueeze(0)
    )

    fut_1 = torch.jit.fork(is_in_box, x_centers_per_image, y_centers_per_image, gt_bboxes_per_image)
    # in fixed center

    center_radius = 2.5

    gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1) - center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1) + center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1) - center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1) + center_radius * expanded_strides_per_image.unsqueeze(0)

    c_l = x_centers_per_image - gt_bboxes_per_image_l
    c_r = gt_bboxes_per_image_r - x_centers_per_image
    c_t = y_centers_per_image - gt_bboxes_per_image_t
    c_b = gt_bboxes_per_image_b - y_centers_per_image

    is_in_centers = torch.min(torch.min(torch.min(c_l, c_t), c_r), c_b) > 0.0
    is_in_centers_all = is_in_centers.sum(dim=0) > 0

    is_in_boxes, is_in_boxes_all = torch.jit.wait(fut_1)

    is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

    is_in_boxes_and_center = (
        is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
    )
    return is_in_boxes_anchor, is_in_boxes_and_center

@torch.jit.script
def dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt : int, fg_mask):
    # Dynamic K
    # ---------------------------------------------------------------
    matching_matrix = torch.zeros(cost.numel(), dtype=torch.uint8)

    ious_in_boxes_matrix = pair_wise_ious
    n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
    topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
    dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
    max_dynamic_ks = torch.max(dynamic_ks)
    mask_dynamic_ks = torch.arange(max_dynamic_ks).repeat(num_gt, 1)
    mask_dynamic_ks = torch.less(mask_dynamic_ks, dynamic_ks.unsqueeze(-1))
    _, pos_idx = torch.topk(
            cost, k=max_dynamic_ks, dim=1, largest=False
        )
    first_pos_idx = pos_idx[:, 0].unsqueeze(-1)
    pos_idx = pos_idx * mask_dynamic_ks + first_pos_idx * ~mask_dynamic_ks
    pos_idx += torch.arange(0, cost.numel(), step=cost.shape[1]).unsqueeze(-1)
    matching_matrix[pos_idx.flatten()] = 1
    matching_matrix = matching_matrix.reshape(-1, cost.shape[1])

    del topk_ious, dynamic_ks, pos_idx

    anchor_matching_gt = matching_matrix.sum(0)
    if (anchor_matching_gt > 1).sum() > 0:
        _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
        matching_matrix[:, anchor_matching_gt > 1] *= 0
        matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
    fg_mask_inboxes = matching_matrix.sum(0) > 0
    num_fg = fg_mask_inboxes.sum().item()

    fg_mask[fg_mask.clone()] = fg_mask_inboxes

    matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
    gt_matched_classes = gt_classes[matched_gt_inds]

    pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
        fg_mask_inboxes
    ]
    return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

@torch.jit.script
def get_assignments(
    batch_idx : int,
    num_gt : int,
    total_num_anchors : int,
    gt_bboxes_per_image,
    gt_classes,
    bboxes_preds_per_image,
    expanded_strides,
    x_shifts,
    y_shifts,
    cls_preds,
    bbox_preds,
    obj_preds,
    labels,
    imgs,
    mode : str ="cpu",
):
    with torch.no_grad():
        device = gt_bboxes_per_image.device

        fg_mask, is_in_boxes_and_center = get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        cls_preds_ = (
                cls_preds_.float().sigmoid_()
                * obj_preds_.float().sigmoid_()
        )
        cls_preds_ = cls_preds_.sqrt_()
        pos_cls_preds_=cls_preds_[:, gt_classes.to(torch.long)]
        pos_cls_preds_=torch.clamp(torch.log(1.0 - pos_cls_preds_), min=-100.0) - torch.clamp(torch.log(pos_cls_preds_),  min=-100.0)
        pos_cls_preds_=pos_cls_preds_.transpose(0,1)
        pair_wise_cls_loss=((torch.clamp(torch.log(1.0 - cls_preds_), min=-100.0).sum(-1)) * -1.0).unsqueeze(0)
        pair_wise_cls_loss = pair_wise_cls_loss + pos_cls_preds_
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.to(device)
            fg_mask = fg_mask.to(device)
            pred_ious_this_matching = pred_ious_this_matching.to(device)
            matched_gt_inds = matched_gt_inds.to(device)

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

@torch.jit.script
def get_gt_loss(batch_idx : int,
    num_gt : int,
    total_num_anchors : int,
    expanded_strides,
    x_shifts,
    y_shifts,
    cls_preds,
    bbox_preds,
    obj_preds,
    labels,
    imgs,
    outputs,
    use_l1 : bool,
    num_classes : int
):
    if num_gt == 0:
        cls_target = outputs.new_zeros((0, num_classes))
        reg_target = outputs.new_zeros((0, 4))
        l1_target = outputs.new_zeros((0, 4))
        obj_target = outputs.new_zeros((total_num_anchors, 1))
        fg_mask = outputs.new_zeros(total_num_anchors, dtype=torch.bool)
        num_fg_img = 0
    else:
        gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
        gt_classes = labels[batch_idx, :num_gt, 0]
        bboxes_preds_per_image = bbox_preds[batch_idx]

        mode = 'cpu'
        (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg_img,
        ) = get_assignments(  # noqa
            batch_idx,
            num_gt,
            total_num_anchors,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            bbox_preds,
            obj_preds,
            labels,
            imgs,
            mode
        )

        cls_target = F.one_hot(
                        gt_matched_classes.to(torch.int64), num_classes
                    ) * pred_ious_this_matching.unsqueeze(-1)
        obj_target = fg_mask.unsqueeze(-1)
        reg_target = gt_bboxes_per_image[matched_gt_inds]

        if use_l1:
            l1_target = get_l1_target(
                outputs.new_zeros((int(num_fg_img), 4)),
                gt_bboxes_per_image[matched_gt_inds],
                expanded_strides[0][fg_mask],
                x_shifts=x_shifts[0][fg_mask],
                y_shifts=y_shifts[0][fg_mask],
            )
        else:
            l1_target = outputs.new_zeros(0, 4)
    return cls_target, reg_target, obj_target, fg_mask, l1_target, int(num_fg_img), num_gt

@torch.jit.script
def gt_compute_task(
    total_num_anchors : int,
    expanded_strides,
    x_shifts,
    y_shifts,
    cls_preds,
    bbox_preds,
    obj_preds,
    labels,
    imgs,
    outputs,
    use_l1 : bool,
    num_classes : int
):
    # calculate targets
    nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

    futures : List[torch.jit.Future[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]]] = []
    for batch_idx in range(outputs.shape[0]):
        num_gt = int(nlabel[batch_idx])
        futures.append(torch.jit.fork(get_gt_loss,batch_idx,
                num_gt,
                total_num_anchors,
                expanded_strides,
                x_shifts,y_shifts,
                cls_preds,
                bbox_preds,
                obj_preds,
                labels,
                imgs,
                outputs,
                use_l1,
                num_classes))
    return futures

class YOLOXHeadScript(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        use_hpu=False,
        use_hmp=False
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False
        self.use_hpu = use_hpu
        self.use_hmp = use_hmp

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, labels.dtype
                )
                # output, grid = self.get_output_and_grid(
                #     output, k, stride_this_level, xin[0].type()
                # )

                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(labels)
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            if self.use_hpu and self.use_hmp:
                from habana_frameworks.torch.hpex import hmp
                with hmp.disable_casts():
                    return self.get_losses(
                        imgs,
                        x_shifts,
                        y_shifts,
                        expanded_strides,
                        labels,
                        torch.cat(outputs, 1),
                        origin_preds,
                        dtype=labels.dtype,
                    )
            else:
                return self.get_losses(
                    imgs,
                    x_shifts,
                    y_shifts,
                    expanded_strides,
                    labels,
                    torch.cat(outputs, 1),
                    origin_preds,
                    dtype=labels.dtype,
                )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].dtype)
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).to(dtype=dtype)

            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)

        # TODO ==== workaround for SW-90060
        # output[..., :2] = (output[..., :2] + grid) * stride
        # output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        output_wo1 = (output[...,:2] + grid) * stride
        output_wo2 = torch.exp(output[..., 2:4]) * stride
        output_wo3 = output[..., 4:]
        output = torch.cat([output_wo1, output_wo2, output_wo3], -1)
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        imgs = imgs.cpu()
        labels = labels.cpu()
        outputs = outputs.cpu()
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1).cpu()  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1).cpu()  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1).cpu()
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1).cpu()

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        futures = gt_compute_task(
                    total_num_anchors,
                    expanded_strides,
                    x_shifts,y_shifts,
                    cls_preds,
                    bbox_preds,
                    obj_preds,
                    labels,
                    imgs,
                    outputs,
                    self.use_l1,
                    self.num_classes)

        results : List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]] = [torch.jit.wait(future) for future in futures]
        for result in results:
            cls_target, reg_target, obj_target, fg_mask, l1_target, num_fg_img, num_gt = result
            num_gts += num_gt
            num_fg += num_fg_img

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )
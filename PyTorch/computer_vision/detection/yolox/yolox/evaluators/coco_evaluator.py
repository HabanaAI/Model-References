#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
###########################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###########################################################################

import contextlib
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

import numpy as np

import torch

from yolox.data.datasets import COCO_CLASSES
from yolox.utils import (
    gather,
    is_main_process,
    Postprocessor,
    synchronize,
    xyxy2xywh
)


def per_class_AR_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_AP: bool = False,
        per_class_AR: bool = False,
        use_hpu: bool = False,
        warmup_steps: int = 0,
        cpu_post_processing: bool = False
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to False.
            per_class_AR: Show per class AR during evalution or not. Default to False.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR
        self.use_hpu = use_hpu
        self.cpu_post_processing = cpu_post_processing
        self.warmup_steps = warmup_steps

        self.post_proc_device = None
        if self.use_hpu:
            self.post_proc_device = "hpu"
        if self.cpu_post_processing:
            self.post_proc_device = "cpu"
        if self.post_proc_device is None:
            raise RuntimeError("Device for post-processing is not defined. "
                               "Please, use `--cpu-post-processing` and your device should be `hpu`.")

        self.postprocessor = Postprocessor(
                                    self.num_classes,
                                    self.confthre,
                                    self.nmsthre,
                                    self.post_proc_device
                            )
        if self.cpu_post_processing:
            self.postprocessor = torch.jit.script(self.postprocessor)

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        if torch.cuda.is_available():
            tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        else:
            tensor_type = torch.float
        model = model.eval()
        if half:
            model = model.half()

        progress_bar = tqdm if is_main_process() else iter

        data_for_evaluation = []
        inference_time = 0
        nms_time = 0
        interence_time_recorded_steps = 0

        if trt_file is not None: # ignore this on cpu or hpu
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        loop_start = time.time()
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                if self.use_hpu:
                    imgs = imgs.to(dtype=tensor_type, device=torch.device("hpu"))
                else:
                    imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_batch_size_full = imgs.size(0) == self.dataloader.batch_size
                need_time_record = self.warmup_steps <= cur_iter and is_batch_size_full
                if need_time_record:
                    interence_time_recorded_steps += 1
                    infer_start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if self.cpu_post_processing:
                    outputs = outputs.to('cpu')

                outputs = self.postprocessor(outputs)

                if need_time_record:
                    inference_time += time.time() - infer_start

            data_for_evaluation.append((outputs, info_imgs, ids))

        total_time = time.time() - loop_start
        if interence_time_recorded_steps < 1:
            logger.warning(
                "Not enough steps have been performed to calculate the inference performance metrics."
                )

        data_list = []
        for outputs, info_imgs, ids in data_for_evaluation:
            data_list.extend(self.convert_to_coco_format(outputs, info_imgs, ids))

        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))

            gloo_group = torch.distributed.new_group(backend="gloo")

            inference_time_list = gather(inference_time, dst=0, group=gloo_group)
            total_time_list = gather(total_time, dst=0, group=gloo_group)
            interence_time_recorded_steps_list = gather(interence_time_recorded_steps, dst=0, group=gloo_group)

            statistics = {
                'inference_time': inference_time_list,
                'interence_time_recorded_steps': interence_time_recorded_steps_list,
                'total_time': total_time_list
            }
        else:
            statistics = {
                'inference_time': [inference_time],
                'interence_time_recorded_steps': [interence_time_recorded_steps],
                'total_time': [total_time]
            }

        eval_results = self.evaluate_prediction(data_list, statistics)

        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None or output.size(0) == 0:
                continue
            output = output.cpu()
            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 5]
            scores = (output[:, 4]).float()
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time_list = statistics['inference_time']
        interence_time_recorded_steps_list = statistics['interence_time_recorded_steps']
        
        total_time = np.average(statistics['total_time'])
        total_throughput = len(self.dataloader.dataset) / total_time

        time_info = f"Total evaluation loop time: {total_time:.2f} (s)" + \
                    f"\nTotal evaluation loop throughput: {total_throughput:.2f} (images/s)"

        average_inference_time = []
        average_inference_tp = []
        is_inference_recorded = False
        for inference_time, interence_time_recorded_steps in zip(inference_time_list, interence_time_recorded_steps_list):
            if interence_time_recorded_steps < 1:
                continue
            is_inference_recorded = True

            average_inference_time.append(1000 * inference_time / interence_time_recorded_steps)
            total_images_recorded = interence_time_recorded_steps * self.dataloader.batch_size
            average_inference_tp.append(total_images_recorded / inference_time)

        if is_inference_recorded:
            average_inference_time = np.average(average_inference_time)
            average_inference_tp = np.average(average_inference_tp)

            time_info += f"\nAverage inference time: {average_inference_time:.2f} (ms)" + \
                        f"\nAverage inference throughput: {average_inference_tp:.2f} (images/s)"

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            if self.per_class_AP:
                AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info

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

import concurrent.futures as futures

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
        post_processing: str = None,
        enable_mediapipe: bool = False,
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
        self.post_processing_device = post_processing
        self.async_post_processing = False
        self.warmup_steps = warmup_steps
        self.enable_mediapipe = enable_mediapipe and use_hpu

        self.post_processor = self.__init_post_processing()


    def __del__(self):
        if self.async_post_processing:
            self.pool.shutdown()

    def __init_post_processing(self):
        if self.post_processing_device is None:
            raise RuntimeError("Device for post-processing is not defined.")

        if self.post_processing_device == 'off':
            return None

        if self.post_processing_device == 'cpu-async':
            self.post_processing_device = 'cpu'
            self.async_post_processing = True

        self.post_processor = Postprocessor(
                                    self.num_classes,
                                    self.confthre,
                                    self.nmsthre,
                                    self.post_processing_device
                            )

        if self.post_processing_device == 'cpu':
            self.post_processor = torch.jit.script(self.post_processor)

            if self.async_post_processing:
                self.pool = futures.ThreadPoolExecutor(max_workers=2)

        return self.post_processor

    def inference(
            self,
            model,
            decoder,
            tensor_type,
            half,
            is_warmup=False
        ):
        data_for_evaluation = []
        inference_time = 0

        loop_start = time.time()
        total_images = 0

        progress_bar = tqdm if is_main_process() else iter
        progress_bar = progress_bar if not is_warmup else iter
        for step, batch_data in enumerate(progress_bar(self.dataloader)):
            if is_warmup and step == self.warmup_steps:
                break

            with torch.no_grad():
                # unpack returned tuple - structure is now the same for PT or mediapipe dataloader
                imgs, _, _, ids = batch_data
                valid_images = imgs.size(0)

                # MediaPipe dataloader pads final partial batch by repeating the last image
                # to avoid graph recompilation, we send in the full batch to model() but read the number of valid images
                #   from 'ids' - see details in YoloxPytorchIterator()
                # for default dataloader, manually pad batch here by replicating last image
                if self.enable_mediapipe:
                    valid_images = ids.size(0)
                elif valid_images < self.dataloader.batch_size:
                    pad_images = self.dataloader.batch_size - valid_images
                    imgs = torch.cat((imgs, imgs[-1:].repeat(pad_images, 1, 1, 1)), dim=0)

                total_images += valid_images
                if self.use_hpu:
                    imgs = imgs.to(dtype=tensor_type, device=torch.device("hpu"))
                else:
                    imgs = imgs.type(tensor_type)

                infer_start = time.time()

                outputs = model(imgs)

                # remove padded images
                if valid_images < self.dataloader.batch_size:
                    outputs = torch.narrow(outputs, 0, 0, valid_images)

                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if self.post_processor:
                    if self.post_processing_device == 'cpu':
                        outputs = outputs.cpu()
                        if half: # "nms_kernel" not implemented for 'Half'
                            outputs = outputs.type(torch.Tensor)

                    if self.async_post_processing:
                        outputs = self.pool.submit(self.post_processor, outputs)
                    else:
                        outputs = self.post_processor(outputs)
                else:
                    outputs = outputs.cpu()

                if self.post_processor and self.post_processing_device != 'cpu':
                    outputs = [output.cpu() for output in outputs]

                inference_time += time.time() - infer_start

            data_for_evaluation.append(outputs)

        if self.async_post_processing:
             futures.wait(data_for_evaluation, timeout=30, return_when=futures.ALL_COMPLETED)

        total_time = time.time() - loop_start

        return data_for_evaluation, inference_time, total_time, total_images

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        performance_test_only=False,
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

        if self.warmup_steps > 0:
            logger.info("Warming-up...")
            self.inference(model, decoder, tensor_type, half, is_warmup=True)

        inference_result = self.inference(model, decoder, tensor_type, half)
        data_for_evaluation, inference_time, total_time, total_images = inference_result

        data_list = []
        if performance_test_only or not self.post_processor:
            logger.info("Skipping convert_to_coco_format...")
        else:
            logger.info("Calling convert_to_coco_format...")
            for outputs, batch in zip(data_for_evaluation, self.dataloader):
                _, _, info_imgs, ids = batch
                if self.async_post_processing:
                    data_list.extend(self.convert_to_coco_format(outputs.result(), info_imgs, ids))
                else:
                    data_list.extend(self.convert_to_coco_format(outputs, info_imgs, ids))

        logger.info("Collecting statistics...")
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))

            gloo_group = torch.distributed.new_group(backend="gloo")

            inference_time_list = gather(inference_time, dst=0, group=gloo_group)
            total_time_list = gather(total_time, dst=0, group=gloo_group)
            interence_steps_list = gather(len(self.dataloader), dst=0, group=gloo_group)
            total_images_list = gather(total_images, dst=0, group=gloo_group)
            statistics = {
                'inference_time': inference_time_list,
                'interence_steps': interence_steps_list,
                'total_time': total_time_list,
                'total_images': total_images_list
            }
        else:
            statistics = {
                'inference_time': [inference_time],
                'interence_steps': [len(self.dataloader)],
                'total_time': [total_time],
                'total_images': [total_images]
            }

        logger.info(f"Calling evaluate prediction (performance_test_only = {performance_test_only})...")
        eval_results = self.evaluate_prediction(data_list, statistics, performance_test_only=performance_test_only)

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
            # scale the bounding boxes back to the original image dimensions
            # NOTE:
            # - MediaPipe path does not pad images so they are always stretched to img_size (e.g. 640x640)
            # - SW path does padding to preserve the aspect ratio (one dimension may be < 640)
            # - this may reduce accuracy slightly when training was done with 'padded' resize and inference is done with 'stretch' resize
            if self.enable_mediapipe:
                scale_w = float(img_w) / self.img_size[1]
                scale_h = float(img_h) / self.img_size[0]

                # bboxes[upper left X, upper left Y, lower right X, lower right Y]
                bboxes[:,0] *= scale_w
                bboxes[:,1] *= scale_h
                bboxes[:,2] *= scale_w
                bboxes[:,3] *= scale_h
            else:
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

    def evaluate_prediction(self, data_dict, statistics, performance_test_only=False):
        if not is_main_process():
            return 0, 0, None, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time_list = statistics['inference_time']

        interence_steps_list = statistics['interence_steps']
        
        num_images = np.sum(statistics['total_images'])
        total_time = np.average(statistics['total_time'])
        total_throughput = num_images / total_time

        time_info = f"Total evaluation loop time:        {total_time: 7.2f} (s)" + \
                    f"\nTotal evaluation loop throughput:  {total_throughput: 7.2f} (images/s)" + \
                    f"\nTotal evaluation loop images:      {num_images: 7d}"

        # TODO: remove duplication
        average_inference_time = []
        average_inference_tp = []
        is_inference_recorded = False
        for inference_time, interence_steps in zip(inference_time_list, interence_steps_list):
            if interence_steps < 1:
                continue
            is_inference_recorded = True
            total_images_recorded = interence_steps * self.dataloader.batch_size      # total images processed on this device

            average_inference_time.append(1000 * inference_time / interence_steps)    # average inference time per batch on this device (in msec)
            average_inference_tp.append(total_images_recorded / inference_time)                     # average inference throughput for this device
            

        if is_inference_recorded:
            average_inference_time = np.average(average_inference_time)                             # average inference time across all devices
            overall_inference_tp = np.sum(average_inference_tp)                                     # overall throughput across all devices

            time_info += f"\nAverage inference time per batch:  {average_inference_time: 7.2f} (ms)" + \
                         f"\nAverage inference throughput:      {overall_inference_tp: 7.2f} (images/s)"

        info = time_info + "\n"

        # return raw perf data for optionally writing to CSV file
        perf_results = [self.dataloader.batch_size, total_throughput, overall_inference_tp, num_images]

        # skip comparison for perf. experiments
        if performance_test_only == True:
            logger.info("Skipping COCOeval comparison...")
            return 0, 0, info, perf_results

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
            return cocoEval.stats[0], cocoEval.stats[1], info, perf_results
        else:
            return 0, 0, info, perf_results

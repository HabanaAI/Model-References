# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - added support for cpu and hpu devices
# - added possibility to limit training time by `max_batches` argument
# - renamed batch_size_per_gpu parameter to batch_size_per_device
# - renamed use_cuda parameter to store_on_device
# - added tensorboard logging possibility by `dump_tb_events` argument
# - added throughput logging in MetricLogger
# - temporary increased number of chunks in knn_classifier from 100 to 200 due to an issue on HPU device

import os
import sys
import argparse

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
import habana_compat


def extract_feature_pipeline(args, create_summary_writer_fn):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = ReturnIndexDataset(os.path.join(args.data_path, "train"), transform=transform)
    dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "val"), transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_device,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_device,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.to(device=args.device)
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    global_batch_size = args.batch_size_per_device * utils.get_world_size()
    train_features = extract_features(model, data_loader_train, global_batch_size, args.device, args.store_on_device, summary_writer=create_summary_writer_fn('train_features'), max_batches=args.max_batches)
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val, global_batch_size, args.device, args.store_on_device, summary_writer=create_summary_writer_fn('test_features'), max_batches=args.max_batches)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        Path(args.dump_features).mkdir(parents=True, exist_ok=True)
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model,
                     data_loader,
                     global_batch_size,
                     device,
                     store_on_device=True,
                     multiscale=False,
                     summary_writer=None,
                     max_batches=None):
    metric_logger = utils.MetricLogger(global_batch_size, delimiter="  ", summary_writer=summary_writer)
    features = None
    habana_compat.mark_step()
    for batch_num, (samples, index) in enumerate(metric_logger.log_every(data_loader, 10)):
        if max_batches is not None and batch_num >= max_batches:
            print("Reached max batches. Stopping...")
            break
        samples = samples.to(device=device, non_blocking=True)
        index = index.to(device=device, non_blocking=True)
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if store_on_device:
                features = features.to(device=device, non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if store_on_device:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
        habana_compat.mark_step()
    return features


@torch.no_grad()
def knn_classifier(train_features,
                   train_labels,
                   test_features,
                   test_labels,
                   k,
                   T,
                   num_classes=1000,
                   summary_writer=None,
                   max_batches=None):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 200
    imgs_per_chunk = num_test_images // num_chunks
    metric_logger = utils.MetricLogger(num_chunks, delimiter="  ", summary_writer=summary_writer)
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in metric_logger.log_every(range(0, num_test_images, imgs_per_chunk), 10):
        if max_batches is not None and idx >= max_batches * imgs_per_chunk:
            print("Reached max batches. Stopping...")
            break
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')

    # HPU
    parser.add_argument('--device', choices=['hpu', 'cuda', 'cpu'], default='hpu',
                        help='Device to be used for computation')

    parser.add_argument('--batch_size_per_device', default=128, type=int,
        help='Per-device batch-size : number of distinct images loaded on one device.')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--store_on_device', default=True, type=utils.bool_flag,
        help="Should we store the features on device? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per device.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument("--max_batches", default=None, type=int,
         help="Limit number of processed batches, both for feature extraction and KNN calculation. "
              "Useful only for testing purposes.")
    parser.add_argument('--dump_tb_events', default=None,
        help='Path where to save tensorboard events, empty for no saving')
    args = parser.parse_args()
    habana_compat.setup_hpu(args)

    def create_summary_writer_fn(subdir):
        if args.dump_tb_events and utils.is_main_process():
            return SummaryWriter(os.path.join(args.dump_tb_events, subdir))
        return None

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.load_features:
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
        train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
    else:
        # need to extract features !
        train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args, create_summary_writer_fn)

    if utils.get_rank() == 0:
        if args.store_on_device:
            train_features = train_features.to(device=args.device)
            test_features = test_features.to(device=args.device)
            train_labels = train_labels.to(device=args.device)
            test_labels = test_labels.to(device=args.device)

        print("Features are ready!\nStart the k-NN classification.")
        for k in args.nb_knn:
            summary_writer = create_summary_writer_fn(f'{k}-NN')
            top1, top5 = knn_classifier(train_features,
                                        train_labels,
                                        test_features,
                                        test_labels,
                                        k,
                                        args.temperature,
                                        summary_writer=summary_writer,
                                        max_batches=args.max_batches)
            if summary_writer is not None:
                summary_writer.add_scalar('top1', top1, 0)
                summary_writer.add_scalar('top5', top5, 0)
                summary_writer.flush()
            print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
    dist.barrier()

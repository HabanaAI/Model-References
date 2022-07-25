###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - Added determinism support via --deterministic/--det. For deterministic training you also need to set the seed using --seed


import os
from argparse import ArgumentParser
from utils import DefaultBoxes, Encoder, COCODetection
from base_model import Loss
from utils import SSDTransformer
from ssd300 import SSD300
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import random
import numpy as np
import logging
from mlperf_logging.mllog import constants as mllog_const
from mlperf_logger import ssd_print, broadcast_seeds
from mlperf_logger import mllogger
import utils_distributed
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.utils.debug as htdebug

_BASE_LR=2.5e-3

def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    parser.add_argument('--data', '-d', type=str, default='/coco',
                        help='path to test and training data files')
    parser.add_argument('--pretrained-backbone', type=str, default=None,
                        help='path to pretrained backbone weights file, '
                             'default is to get it from online torchvision repository')
    parser.add_argument('--epochs', '-e', type=int, default=800,
                        help='number of epochs for training')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='number of examples for each training iteration')
    parser.add_argument('--val-batch-size', type=int, default=None,
                        help='number of examples for each validation iteration (defaults to --batch-size)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--use-hpu', action='store_true',
                        help='use available HPUs')
    parser.add_argument('--dl-worker-type', default='HABANA', type=lambda x: x.upper(),
                        choices = ["MP", "HABANA"], help='select multiprocessing or habana accelerated')
    parser.add_argument('--disable-distributed-validation', action='store_true',
                        help='disable distributed dataloader for validation')
    parser.add_argument('--hpu-lazy-mode', action='store_true',
                        help='enable lazy mode execution on HPUs')
    parser.add_argument('--hmp', dest='is_hmp', action='store_true', help='enable hmp mode')
    parser.add_argument('--hmp-bf16', default='', help='path to bf16 ops list in hmp O1 mode')
    parser.add_argument('--hmp-fp32', default='', help='path to fp32 ops list in hmp O1 mode')
    parser.add_argument('--hmp-opt-level', default='O1', help='choose optimization level for hmp')
    parser.add_argument('--hmp-verbose', action='store_true', help='enable verbose mode for hmp')
    parser.add_argument('--seed', '-s', type=int, default=random.SystemRandom().randint(0, 2**32 - 1),
                        help='manually set random seed for torch')
    parser.add_argument('--deterministic', '--det', action='store_true', help='force deterministic training')  # amorgenstern
    parser.add_argument('--threshold', '-t', type=float, default=0.23,
                        help='stop training early at threshold')
    parser.add_argument('--iteration', type=int, default=0,
                        help='iteration to start from')
    parser.add_argument('--end-iteration', type=int, default=0,
                        help='iteration to end on')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint file')
    parser.add_argument('--no-save', action='store_true',
                        help='save model checkpoints')
    parser.add_argument('--val-interval', type=int, default=5,
                         help='epoch interval for validation in addition to --val-epochs.')
    parser.add_argument('--val-epochs', nargs='*', type=int,
                        default=[],
                        help='epochs at which to evaluate in addition to --val-interval')
    parser.add_argument('--batch-splits', type=int, default=1,
                        help='Split batch to N steps (gradient accumulation)')
    parser.add_argument('--lr-decay-schedule', nargs='*', type=int,
                        default=[40, 50],
                        help='epochs at which to decay the learning rate')
    parser.add_argument('--warmup', type=float, default=None,
                        help='how long the learning rate will be warmed up in fraction of epochs')
    parser.add_argument('--warmup-factor', type=int, default=0,
                        help='mlperf rule parameter for controlling warmup curve')
    parser.add_argument('--lr', type=float, default=_BASE_LR,
                        help='base learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay factor')
    parser.add_argument('--num-cropping-iterations', type=int, default=1,
                        help='cropping retries in augmentation pipeline, '
                             'default 1, other legal value is 50')
    parser.add_argument('--nms-valid-thresh', type=float, default=0.05,
                        help='in eval, filter input boxes to those with score greater '
                             'than nms_valid_thresh.')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for dataloader.')
    # Distributed stuff
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int,
                        help='Used for multi-process training. Can either be manually set '
                             'or automatically set by using \'python -m multiproc\'.')

    return parser.parse_args()


def show_memusage(device=0):
    import gpustat
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))


def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


def coco_eval(model, val_dataloader, cocoGt, encoder, inv_map, threshold,
              epoch, iteration, batch_size, log_interval=100,
              use_cuda=True, use_hpu=False, hpu_device=None, is_hmp=False,
              hpu_lazy_mode=False, nms_valid_thresh=0.05, N_gpu=1,
              local_rank=-1, enable_distributed_validation=True):
    from pycocotools.cocoeval import COCOeval
    print("")

    distributed = False
    if enable_distributed_validation and local_rank >= 0:
        distributed = True

    model.eval()
    if use_cuda:
        model.cuda()
    if use_hpu:
        model.to(hpu_device)
    ret = []

    overlap_threshold = 0.50
    nms_max_detections = 200
    print("nms_valid_thresh is set to {}".format(nms_valid_thresh))

    mllogger.start(
        key=mllog_const.EVAL_START,
        metadata={mllog_const.EPOCH_NUM: epoch})

    start = time.time()
    for nbatch, (img, img_id, img_size, bbox, label) in enumerate(val_dataloader):
        with torch.no_grad():
            if use_cuda:
                img = img.cuda()
            if use_hpu:
                img = img.to(hpu_device, non_blocking=True)
            ploc, plabel = model(img)

            try:
                if use_hpu:
                    if is_hmp:
                        results = encoder.decode_batch(ploc.float(), plabel.float(),
                                                       overlap_threshold,
                                                       nms_max_detections,
                                                       nms_valid_thresh=nms_valid_thresh)
                    else:
                        results = encoder.decode_batch(ploc, plabel,
                                                       overlap_threshold,
                                                       nms_max_detections,
                                                       nms_valid_thresh=nms_valid_thresh)
                else:
                    results = encoder.decode_batch(ploc, plabel,
                                                   overlap_threshold,
                                                   nms_max_detections,
                                                   nms_valid_thresh=nms_valid_thresh)

            except:
                #raise
                print("")
                print("No object detected in batch: {}, rank: {} ".format(nbatch, local_rank))
                continue

            (htot, wtot) = [d.cpu().numpy() for d in img_size]
            img_id = img_id.cpu().numpy()
            # Iterate over batch elements
            for img_id_, wtot_, htot_, result in zip(img_id, wtot, htot, results):
                loc, label, prob = [r.cpu().numpy() for r in result]

                # Iterate over image detections
                for loc_, label_, prob_ in zip(loc, label, prob):
                    ret.append([img_id_, loc_[0]*wtot_, \
                                         loc_[1]*htot_,
                                         (loc_[2] - loc_[0])*wtot_,
                                         (loc_[3] - loc_[1])*htot_,
                                         prob_,
                                         inv_map[label_]])
        if log_interval and not (nbatch+1) % log_interval:
                print("Completed inference on batch: {}".format(nbatch+1))

    ret = np.array(ret).astype(np.float32)

    if use_hpu and distributed:
       ret_copy = torch.tensor(ret).to(hpu_device)
       ret_sizes = [torch.tensor(0).to(hpu_device) for _ in range(N_gpu)]
       torch.distributed.all_gather(ret_sizes, torch.tensor(ret_copy.shape[0]).to(hpu_device))

       max_size = 0
       sizes = []
       for s in ret_sizes:
           max_size = max(max_size, s.item())
           sizes.append(s.item())
       ret_pad = torch.cat([ret_copy, torch.zeros(max_size-ret_copy.shape[0], 7, dtype=torch.float32).to(hpu_device)])
       other_ret = [torch.zeros(max_size, 7, dtype=torch.float32).to(hpu_device) for i in range(8)]
       torch.distributed.all_gather(other_ret, ret_pad)
       cat_tensors = []
       for i in range(N_gpu):
           cat_tensors.append(other_ret[i][:sizes[i]][:])

       final_results = torch.cat(cat_tensors).cpu().numpy()
    else:
       final_results = ret

    if local_rank in [0, -1]:
       print("")
       print("Predicting Ended, total time: {:.2f} s".format(time.time()-start))

    if local_rank in [0, -1]:
       cocoDt = cocoGt.loadRes(np.array(final_results))
       E = COCOeval(cocoGt, cocoDt, iouType='bbox')
       E.evaluate()
       E.accumulate()
       E.summarize()
       print("Current AP: {:.5f} AP goal: {:.5f}".format(E.stats[0], threshold))
    else:
       # put your model back into training mode
       model.train()
       return False

    # put your model back into training mode
    model.train()

    current_accuracy = E.stats[0]

    ssd_print(device=hpu_device, use_hpu=use_hpu, key=mllog_const.EVAL_ACCURACY,
              value=current_accuracy,
              metadata={mllog_const.EPOCH_NUM: epoch},
              sync=False)
    mllogger.end(
        key=mllog_const.EVAL_STOP,
        metadata={mllog_const.EPOCH_NUM: epoch})
    return current_accuracy >= threshold #Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]

def lr_warmup(optim, wb, iter_num, base_lr, args):
	if iter_num < wb:
		# mlperf warmup rule
		warmup_step = base_lr / (wb * (2 ** args.warmup_factor))
		new_lr = base_lr - (wb - iter_num) * warmup_step

		for param_group in optim.param_groups:
			param_group['lr'] = new_lr

#permute the params from filters first (KCRS) to filters last(RSCK) or vice versa.
#and permute from RSCK to KCRS is used for checkpoint saving
def permute_params(model, to_filters_last, lazy_mode=True):
    if htdebug._is_enabled_synapse_layout_handling():
        print("permute_params disabled")
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            if(param.ndim == 4):
                if to_filters_last:
                    param.data = param.data.permute((2, 3, 1, 0))
                else:
                    param.data = param.data.permute((3, 2, 0, 1))  # permute RSCK to KCRS

    if lazy_mode:
        mark_step()

# permute the momentum from filters first (KCRS) to filters last(RSCK) or vice versa.
# and permute from RSCK to KCRS is used for checkpoint saving
# Used for Habana device only


def permute_momentum(optimizer, to_filters_last, lazy_mode=True):
    if htdebug._is_enabled_synapse_layout_handling():
        print("permute_momentum disabled")
        return
    # Permute the momentum buffer before using for checkpoint
    for group in optimizer.param_groups:
        for p in group['params']:
            param_state = optimizer.state[p]
            if 'momentum_buffer' in param_state:
                buf = param_state['momentum_buffer']
                if(buf.ndim == 4):
                    if to_filters_last:
                        buf = buf.permute((2,3,1,0))
                    else:
                        buf = buf.permute((3,2,0,1))
                    param_state['momentum_buffer'] = buf

    if lazy_mode:
        mark_step()

def mark_step(use_hpu=True):
    if not use_hpu:
        return
    htcore.mark_step()

def train300_mlperf_coco(args):
    global torch
    from coco import COCO
    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.use_hpu:
       args.distributed = False
    if use_cuda:
        try:
            from apex.parallel import DistributedDataParallel as DDP
            if 'WORLD_SIZE' in os.environ:
                args.distributed = int(os.environ['WORLD_SIZE']) > 1
                if args.distributed:
                   # necessary pytorch imports
                   import torch.utils.data.distributed
                   import torch.distributed as dist
        except:
            raise ImportError("Please install APEX from https://github.com/nvidia/apex")

    use_hpu = args.use_hpu
    if not args.disable_distributed_validation:
       enable_distributed_validation = True
    else:
       enable_distributed_validation = False
    hpu_lazy_mode = args.hpu_lazy_mode
    is_hmp = args.is_hmp
    device = torch.device('cpu')
    if args.dl_worker_type == "MP":
        data_loader_type = DataLoader
    elif args.dl_worker_type == "HABANA":
        from habana_dataloader import HabanaDataLoader
        data_loader_type = HabanaDataLoader

    val_data_loader_type = DataLoader

    if use_hpu:
        device = torch.device('hpu')
        if hpu_lazy_mode:
            os.environ["PT_HPU_LAZY_MODE"] = "1"
        else:
            os.environ["PT_HPU_LAZY_MODE"] = "2"
        if is_hmp:
            if not args.hmp_bf16:
                raise IOError("Please provide list of BF16 ops")
            if not args.hmp_fp32:
                raise IOError("Please provide list of FP32 ops")
            from habana_frameworks.torch.hpex import hmp
            hmp.convert(opt_level=args.hmp_opt_level, bf16_file_path=args.hmp_bf16,
                        fp32_file_path=args.hmp_fp32, isVerbose=args.hmp_verbose)
        # TODO - add dataloader

    local_seed = args.seed
    if args.distributed:
        # necessary pytorch imports
        import torch.utils.data.distributed
        import torch.distributed as dist
        if use_hpu:
            # set seeds properly
            args.seed = broadcast_seeds(args.seed, device, use_hpu=True)
            local_seed = (args.seed + dist.get_rank()) % 2**32
        elif  args.no_cuda:
            device = torch.device('cpu')
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device('cuda')
            dist.init_process_group(backend='nccl',
                                    init_method='env://')
            # set seeds properly
            args.seed = broadcast_seeds(args.seed, device)
            local_seed = (args.seed + dist.get_rank()) % 2**32
    mllogger.event(key=mllog_const.SEED, value=local_seed)
    torch.manual_seed(local_seed)
    np.random.seed(seed=local_seed)
    random.seed(local_seed)  # amorgenstern
    torch.cuda.manual_seed(local_seed)  # amorgenstern

    args.rank = dist.get_rank() if args.distributed else args.local_rank

    print("args.rank = {}".format(args.rank))
    print("local rank = {}".format(args.local_rank))
    print("distributed={}".format(args.distributed))

    if use_hpu and is_hmp:
        with hmp.disable_casts():
            dboxes = dboxes300_coco()
            encoder = Encoder(dboxes, use_hpu=use_hpu, hpu_device=device)
    else:
        dboxes = dboxes300_coco()
        encoder = Encoder(dboxes, use_hpu=use_hpu, hpu_device=device)

    input_size = 300
    if use_hpu and is_hmp:
        with hmp.disable_casts():
            train_trans = SSDTransformer(dboxes, (input_size, input_size), val=False,
                                         num_cropping_iterations=args.num_cropping_iterations)
            val_trans = SSDTransformer(dboxes, (input_size, input_size), val=True)
    else:
        train_trans = SSDTransformer(dboxes, (input_size, input_size), val=False,
                                     num_cropping_iterations=args.num_cropping_iterations)
        val_trans = SSDTransformer(dboxes, (input_size, input_size), val=True)

    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.data, "val2017")
    train_annotate = os.path.join(args.data, "annotations/instances_train2017.json")
    train_coco_root = os.path.join(args.data, "train2017")

    if use_hpu and is_hmp:
        with hmp.disable_casts():
            cocoGt = COCO(annotation_file=val_annotate)
            train_coco = COCODetection(train_coco_root, train_annotate, train_trans)
            val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    else:
        cocoGt = COCO(annotation_file=val_annotate)
        train_coco = COCODetection(train_coco_root, train_annotate, train_trans)
        val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    mllogger.event(key=mllog_const.TRAIN_SAMPLES, value=len(train_coco))
    mllogger.event(key=mllog_const.EVAL_SAMPLES, value=len(val_coco))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_coco)
        if enable_distributed_validation and use_hpu:
           val_sampler = torch.utils.data.distributed.DistributedSampler(val_coco)
        else:
           val_sampler = None
    else:
        train_sampler = None
        val_sampler = None
    if use_hpu:
        # patch torch cuda functions that are being unconditionally invoked
        # in the multiprocessing data loader
        torch.cuda.current_device = lambda: None
        torch.cuda.set_device = lambda x: None
    if use_hpu:
        train_dataloader = data_loader_type(train_coco,
                                        batch_size=args.batch_size,
                                        shuffle=(train_sampler is None),
                                        sampler=train_sampler,
                                        pin_memory=True,
                                        pin_memory_device='hpu',
                                        num_workers=args.num_workers)
    else:
        train_dataloader = data_loader_type(train_coco,
                                        batch_size=args.batch_size,
                                        shuffle=(train_sampler is None),
                                        sampler=train_sampler,
                                        num_workers=args.num_workers)
    # set shuffle=True in DataLoader
    if (enable_distributed_validation and use_hpu) or args.rank==0:
        val_dataloader = val_data_loader_type(val_coco,
                                              batch_size=args.val_batch_size or args.batch_size,
                                              shuffle=(val_sampler is None),
                                              sampler=val_sampler,
                                              pin_memory=True,
                                              pin_memory_device='hpu',
                                              num_workers=args.num_workers)
    else:
        val_dataloader = None

    ssd300 = SSD300(train_coco.labelnum, model_path=args.pretrained_backbone)
    if args.checkpoint is not None:
        print("loading model checkpoint", args.checkpoint)
        od = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        ssd300.load_state_dict(od["model"])
    ssd300.train()
    if use_cuda:
        ssd300.cuda()
    if use_hpu and is_hmp:
        with hmp.disable_casts():
            loss_func = Loss(dboxes, use_hpu=use_hpu, hpu_device=device)
    else:
        loss_func = Loss(dboxes, use_hpu=use_hpu, hpu_device=device)
    if use_cuda:
        loss_func.cuda()

    if use_hpu:
        ssd300.to(device)
        loss_func.to(device)

    if args.distributed:
        N_gpu = torch.distributed.get_world_size()
    else:
        N_gpu = 1

    global_batch_size = N_gpu * args.batch_size
    mllogger.event(key=mllog_const.GLOBAL_BATCH_SIZE, value=global_batch_size)
    # Reference doesn't support group batch norm, so bn_span==local_batch_size
    mllogger.event(key=mllog_const.MODEL_BN_SPAN, value=args.batch_size)
    current_lr = args.lr * (global_batch_size / 32)

    assert args.batch_size % args.batch_splits == 0, "--batch-size must be divisible by --batch-splits"
    fragment_size = args.batch_size // args.batch_splits
    if args.batch_splits != 1:
        print("using gradient accumulation with fragments of size {}".format(fragment_size))

    current_momentum = 0.9
    sgd_optimizer = torch.optim.SGD
    if use_hpu and hpu_lazy_mode:
        from habana_frameworks.torch.hpex.optimizers import FusedSGD
        sgd_optimizer = FusedSGD
    optim = sgd_optimizer(ssd300.parameters(), lr=current_lr,
                          momentum=current_momentum,
                          weight_decay=args.weight_decay)
    if use_hpu:
        permute_params(model=ssd300, to_filters_last=True, lazy_mode=hpu_lazy_mode)
        permute_momentum(optimizer=optim, to_filters_last=True, lazy_mode=hpu_lazy_mode)

    ssd_print(device=device, use_hpu=use_hpu, key=mllog_const.OPT_BASE_LR, value=current_lr)
    ssd_print(device=device, use_hpu=use_hpu, key=mllog_const.OPT_WEIGHT_DECAY, value=args.weight_decay)

    # parallelize
    if args.distributed:
        if use_hpu:
            ssd300 = torch.nn.parallel.DistributedDataParallel(ssd300, bucket_cap_mb=100, broadcast_buffers=False, gradient_as_bucket_view=True)
        else:
            ssd300 = DDP(ssd300)

    iter_num = args.iteration
    end_iter_num = args.end_iteration
    if end_iter_num:
        print("--end-iteration set to: {}".format(end_iter_num))
        assert end_iter_num > iter_num, "--end-iteration must have a value > --iteration"
    avg_loss = 0.0
    if use_hpu:
        loss_iter = list()
    inv_map = {v:k for k,v in val_coco.label_map.items()}
    success = torch.zeros(1)
    if use_cuda:
        success = success.cuda()
    if use_hpu:
        success = success.to(device)


    if args.warmup:
        nonempty_imgs = len(train_coco)
        wb = int(args.warmup * nonempty_imgs / (N_gpu*args.batch_size))
        ssd_print(device=device, use_hpu=use_hpu, key=mllog_const.OPT_LR_WARMUP_STEPS, value=wb)
        warmup_step = lambda iter_num, current_lr: lr_warmup(optim, wb, iter_num, current_lr, args)
    else:
        warmup_step = lambda iter_num, current_lr: None

    ssd_print(device=device, use_hpu=use_hpu, key=mllog_const.OPT_LR_WARMUP_FACTOR, value=args.warmup_factor)
    ssd_print(device=device, use_hpu=use_hpu, key=mllog_const.OPT_LR_DECAY_BOUNDARY_EPOCHS, value=args.lr_decay_schedule)
    mllogger.start(
        key=mllog_const.BLOCK_START,
        metadata={mllog_const.FIRST_EPOCH_NUM: 1,
                  mllog_const.EPOCH_COUNT: args.epochs})

    optim.zero_grad(set_to_none=True)
    if use_hpu:
        start = time.time()
    for epoch in range(args.epochs):
        mllogger.start(
            key=mllog_const.EPOCH_START,
            metadata={mllog_const.EPOCH_NUM: epoch})
        # set the epoch for the sampler
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if epoch in args.lr_decay_schedule:
            current_lr *= 0.1
            print("")
            print("lr decay step #{num}".format(num=args.lr_decay_schedule.index(epoch) + 1))
            for param_group in optim.param_groups:
                param_group['lr'] = current_lr

        for nbatch, (img, img_id, img_size, bbox, label) in enumerate(train_dataloader):
            current_batch_size = img.shape[0]
            # Split batch for gradient accumulation
            img = torch.split(img, fragment_size)
            bbox = torch.split(bbox, fragment_size)
            label = torch.split(label, fragment_size)

            for (fimg, fbbox, flabel) in zip(img, bbox, label):
                current_fragment_size = fimg.shape[0]
                if not use_hpu:
                    trans_bbox = fbbox.transpose(1,2).contiguous()
                if use_cuda:
                    fimg = fimg.cuda()
                    trans_bbox = trans_bbox.cuda()
                    flabel = flabel.cuda()
                if use_hpu:
                    fimg = fimg.to(device, non_blocking=True)
                    if is_hmp:
                        with hmp.disable_casts():
                            trans_bbox = fbbox.to(device, non_blocking=True).transpose(1,2).contiguous()
                            flabel = flabel.to(device, non_blocking=True)
                    else:
                        trans_bbox = fbbox.to(device, non_blocking=True).transpose(1,2).contiguous()
                        flabel = flabel.to(device, non_blocking=True)
                fimg = Variable(fimg, requires_grad=True)
                ploc, plabel = ssd300(fimg)
                if use_hpu and is_hmp:
                    with hmp.disable_casts():
                        gloc, glabel = Variable(trans_bbox, requires_grad=False), \
                                Variable(flabel, requires_grad=False)
                        loss = loss_func(ploc.float(), plabel.float(), gloc, glabel)
                else:
                    gloc, glabel = Variable(trans_bbox, requires_grad=False), \
                            Variable(flabel, requires_grad=False)
                    loss = loss_func(ploc, plabel, gloc, glabel)
                loss = loss * (current_fragment_size / current_batch_size) # weighted mean
                loss.backward()
                if use_hpu and hpu_lazy_mode:
                    mark_step()

            warmup_step(iter_num, current_lr)
            if use_hpu and is_hmp:
                with hmp.disable_casts():
                    optim.step()
            else:
                optim.step()
            optim.zero_grad(set_to_none=True)
            if use_hpu:
                loss_iter.append(loss.clone().detach())
            else:
                if not np.isinf(loss.item()): avg_loss = 0.999*avg_loss + 0.001*loss.item()
            if use_hpu and hpu_lazy_mode:
                mark_step()
            if use_hpu:
                if args.log_interval and not iter_num % args.log_interval:
                    cur_loss = 0.0
                    for i, x in enumerate(loss_iter):
                        cur_loss = x.cpu().item()
                        if not np.isinf(cur_loss): avg_loss = 0.999*avg_loss + 0.001*cur_loss
                    if args.rank == 0:
                        print("Rank: {:6d}, Iteration: {:6d}, Loss function: {:5.3f}, Average Loss: {:.3f}"\
                            .format(args.rank, iter_num, cur_loss, avg_loss))
                    loss_iter = list()
            else:
                if args.rank == 0 and args.log_interval and not iter_num % args.log_interval:
                    print("Iteration: {:6d}, Loss function: {:5.3f}, Average Loss: {:.3f}"\
                        .format(iter_num, loss.item(), avg_loss))
            iter_num += 1
            if use_hpu and iter_num == 50:
                start = time.time()
            if end_iter_num and iter_num >= end_iter_num:
                if use_hpu:
                    print("Training Ended, total time: {:.2f} s".format(time.time()-start))
                break

        if (args.val_epochs and (epoch+1) in args.val_epochs) or \
           (args.val_interval and not (epoch+1) % args.val_interval):
            if args.distributed:
                world_size = float(dist.get_world_size())
                for bn_name, bn_buf in ssd300.module.named_buffers(recurse=True):
                    if ('running_mean' in bn_name) or ('running_var' in bn_name):
                        dist.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                        bn_buf /= world_size
                        if not use_hpu:
                           ssd_print(device=device, use_hpu=use_hpu, key=mllog_const.MODEL_BN_SPAN,
                               value=bn_buf)
            if args.rank == 0 or (enable_distributed_validation and use_hpu):
                if args.rank == 0:
                    print("Training Ended, total time: {:.2f} s".format(time.time()-start))
                    if not args.no_save:
                       print("")
                       print("saving model...")
                       if use_hpu:
                          permute_params(model=ssd300, to_filters_last=False, lazy_mode=hpu_lazy_mode)
                          ssd300_copy = SSD300(train_coco.labelnum, model_path=args.pretrained_backbone)
                          if args.distributed:
                             ssd300_copy.load_state_dict(ssd300.module.state_dict())
                          else:
                             ssd300_copy.load_state_dict(ssd300.state_dict())
                          torch.save({"model" : ssd300_copy.state_dict(), "label_map": train_coco.label_info},
                                   "./models/iter_{}.pt".format(iter_num))
                          permute_params(model=ssd300, to_filters_last=True, lazy_mode=hpu_lazy_mode)
                       else:
                          torch.save({"model" : ssd300.state_dict(), "label_map": train_coco.label_info},
                                   "./models/iter_{}.pt".format(iter_num))

                if coco_eval(ssd300, val_dataloader, cocoGt, encoder, inv_map,
                             args.threshold, epoch + 1, iter_num,
                             args.val_batch_size,
                             log_interval=args.log_interval,
                             use_cuda=use_cuda,
                             use_hpu=use_hpu,
                             hpu_device=device,
                             is_hmp=is_hmp,
                             hpu_lazy_mode=hpu_lazy_mode,
                             nms_valid_thresh=args.nms_valid_thresh,
                             N_gpu=N_gpu,
                             local_rank=args.local_rank if args.distributed else -1,
                             enable_distributed_validation=enable_distributed_validation):
                    success = torch.ones(1)
                    if use_cuda:
                        success = success.cuda()
                    if use_hpu:
                        success = success.to(device)
            if args.distributed:
                if use_hpu:
                   dist.barrier()
                dist.broadcast(success, 0)
            if success[0]:
                    return True
            mllogger.end(
                key=mllog_const.EPOCH_STOP,
                metadata={mllog_const.EPOCH_NUM: epoch})
    mllogger.end(
        key=mllog_const.BLOCK_STOP,
        metadata={mllog_const.FIRST_EPOCH_NUM: 1,
                  mllog_const.EPOCH_COUNT: args.epochs})

    return False

def main():
    mllogger.start(key=mllog_const.INIT_START)
    args = parse_args()

    utils_distributed.init_distributed_mode(args)

    if args.local_rank == 0:
        if not os.path.isdir('./models'):
            os.mkdir('./models')

    torch.backends.cudnn.benchmark = True
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    mllogger.end(key=mllog_const.INIT_STOP)
    mllogger.start(key=mllog_const.RUN_START)

    success = train300_mlperf_coco(args)

    # end timing here
    mllogger.end(key=mllog_const.RUN_STOP, value={"success": success})


if __name__ == "__main__":
    main()

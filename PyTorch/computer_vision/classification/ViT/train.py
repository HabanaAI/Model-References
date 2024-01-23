# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta
import time

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from models.modeling import VisionTransformer, CONFIGS

from vit_utils import scheduler
from vit_utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from vit_utils.data_utils import get_loader
from vit_utils.dist_util import get_world_size

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.utils.debug as htdebug

logger = logging.getLogger(__name__)

def init_distributed_mode(args):
    world_size = 0

    from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu
    world_size, rank, args.local_rank = initialize_distributed_hpu()
    print('| distributed init (rank {})'.format(args.local_rank), flush=True)

    process_per_node = 8 #make this configurable for multi-hls using args
    if world_size  > 1:
        # extend the default HCL timeout
        os.environ["MAX_WAIT_ATTEMPTS"] = "50"
        dist.init_process_group('hccl', rank=rank, world_size=world_size)
    else:
        print("single card run ")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model, optimizer, scheduler):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(
        args.output_dir, "%s_checkpoint.pth" % args.name)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)
    if (str(args.device) == 'hpu'):
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to('cpu')

        config = CONFIGS[args.model_type]
        copy_model = VisionTransformer(
            config, args.img_size, zero_head=True, num_classes=1000)
        copy_model.load_state_dict(model_to_save.state_dict())
        save_state = {'model': copy_model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict(),
                      'config': config}
        torch.save(save_state, model_checkpoint)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to('hpu')
    else:
        torch.save(model_to_save.state_dict(), model_checkpoint)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "imagenet1K":
        num_classes = 1000
    else:
        num_classes = 100

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    if os.path.exists(args.pretrained_dir) or args.support_inaccurate_perf_test == False :
        model.load_from(np.load(args.pretrained_dir))
    else:
        logger.info("bypassed loading pre-trained weights - results will not be correct - internal perf test only")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)

    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.is_autocast):
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            with torch.no_grad():
                logits = model(x)[0]

                eval_loss = loss_fct(logits, y)
                eval_losses.update(eval_loss.item())

                preds = torch.argmax(logits, dim=-1)

            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                    )
                all_label[0] = np.append(
                    all_label[0], y.detach().cpu().numpy(), axis=0
                    )
            epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    logger.info("========    args    ========= \n")
    logger.info(args)
    logger.info("========            ========= \n")
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    if (str(args.device) == 'hpu' and args.run_lazy_mode):
        # use fused SGD for better performance
        from habana_frameworks.torch.hpex.optimizers import FusedSGD
        optimizer = FusedSGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False, gradient_as_bucket_view=False)

    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps* (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    # Train!
    logger.info("***** Running training *****")
    logger.info(args.device)
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size (per CPU/HPU) = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    end = time.time()
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.is_autocast):
                batch = tuple(t.to(args.device) for t in batch)
                x, y = batch
                loss = model(x, y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # WA -MarkStep added for SFG support- will be removed later on
            if (str(args.device) == 'hpu') and args.run_lazy_mode:
                htcore.mark_step()

            loss.backward()

            if (str(args.device) == 'hpu') and args.run_lazy_mode:
                htcore.mark_step()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if (str(args.device) == 'hpu'):
                    optimizer.step()
                    if args.run_lazy_mode:
                        htcore.mark_step()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()	# moved after optimizer step call (see https://pytorch.org/docs/stable/optim.html)

                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f images/sec=%2.5f)" % (global_step, t_total, losses.val, total_batch_size / (time.time()-end))
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy = valid(args, model, writer, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model, optimizer, scheduler)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
                end = time.time()
            if str(args.device) == 'hpu' and args.local_rank != -1:
                dist.barrier()
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet1K"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=384, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=6e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--internal-perf-measure',dest='support_inaccurate_perf_test',  action='store_true',
                        help="allow inaccurate run (skip pretained weights load) for internal perf test")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--data_path", default="dataset_path", type=str,
                        help="The data path to non CIFAR10, CIFAR100 dataset  e.g. imagenet1K directory")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",)
    parser.add_argument('--use_hpu', type=int,  default=1, help='use hpu for run')
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--run_lazy_mode', default='True', type=lambda x: x.lower() == 'true',
                        help='run model in lazy execution mode(enabled by default).'
                        'Any value other than True(case insensitive) disables lazy mode')
    parser.add_argument('--autocast', dest='is_autocast', action='store_true', help='enable autocast mode on Gaudi')
    args = parser.parse_args()

    if args.use_hpu == 0:
        logger.info("**************  setting device to be CPU !")
        device = torch.device("cpu")
    else:
        logger.info("**************  setting device to be HPU !")
        device = torch.device("hpu")
    args.n_gpu = 0
    args.device = device

    ############# Setup & Function Add For HPU Mode
    if (str(args.device) == 'hpu'):
        if args.run_lazy_mode:
            assert os.getenv('PT_HPU_LAZY_MODE') == '1' or os.getenv('PT_HPU_LAZY_MODE') == None, f"run_lazy_mode == True, but PT_HPU_LAZY_MODE={os.getenv('PT_HPU_LAZY_MODE')}. For run lazy mode, set PT_HPU_LAZY_MODE to 1"

    if args.use_hpu:
        init_distributed_mode(args)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1)))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    model.to(device)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()

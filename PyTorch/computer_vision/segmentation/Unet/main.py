
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

def get_canonical_path_str(name):
    return os.fspath(Path(os.path.expandvars(os.path.expanduser(name))).resolve())

def positive_int(value):
    ivalue = int(value)
    assert ivalue > 0, f"Argparse error. Expected positive integer but got {value}"
    return ivalue


def non_negative_int(value):
    ivalue = int(value)
    assert ivalue >= 0, f"Argparse error. Expected positive integer but got {value}"
    return ivalue


def float_0_1(value):
    ivalue = float(value)
    assert 0 <= ivalue <= 1, f"Argparse error. Expected float to be in range (0, 1), but got {value}"
    return ivalue

def get_main_args(strings=None):
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument

    arg("--framework", type=str, choices=["pytorch", "pytorch-lightning"],
        default="pytorch-lightning",
        help='Launch model training on pytorch or pytorch-lightning framework'
    )
    arg(
        "--exec_mode",
        type=str,
        choices=["train", "evaluate", "predict"],
        default="train",
        help="Execution mode to run the model",
    )
    arg("--data", type=str, default="/data", help="Path to data directory")
    arg("--results", type=str, default="/results", help="Path to results directory")
    arg("--logname", type=str, default=None, help="Name of dlloger output")
    arg("--task", type=str, help="Task number. MSD uses numbers 01-10")
    arg("--gpus", type=non_negative_int, default=0, help="Number of gpus")
    arg("--hpus", type=non_negative_int, default=0, help="Number of hpus")
    arg("--learning_rate", type=float, default=0.001, help="Learning rate")
    arg("--gradient_clip_val", type=float, default=0, help="Gradient clipping norm value")
    arg("--negative_slope", type=float, default=0.01, help="Negative slope for LeakyReLU")
    arg("--tta", action="store_true", help="Enable test time augmentation")

    arg("--gradient_clip", action="store_true", help="Enable gradient_clip to improve training stability")
    # For Gradient clip norm, the default value is 12. refer to the original model: https://github.com/MIC-DKFZ/nnUNet
    arg("--gradient_clip_norm", type=float, default=12, help="Gradient clipping norm value for NPT ONLY")

    arg("--amp", action="store_true", help="Enable automatic mixed precision")
    arg("--benchmark", action="store_true", help="Run model benchmarking")
    arg("--deep_supervision", action="store_true", help="Enable deep supervision")
    arg("--drop_block", action="store_true", help="Enable drop block")
    arg("--attention", action="store_true", help="Enable attention in decoder")
    arg("--residual", action="store_true", help="Enable residual block in encoder")
    arg("--focal", action="store_true", help="Use focal loss instead of cross entropy")
    arg("--sync_batchnorm", action="store_true", help="Enable synchronized batchnorm")
    arg("--save_ckpt", action="store_true", help="Enable saving checkpoint")
    arg("--nfolds", type=positive_int, default=5, help="Number of cross-validation folds")
    arg("--seed", type=non_negative_int, default=1, help="Random seed")
    arg("--skip_first_n_eval", type=non_negative_int, default=0, help="Skip the evaluation for the first n epochs.")
    arg("--ckpt_path", type=str, default=None, help="Path to checkpoint")
    arg("--fold", type=non_negative_int, default=0, help="Fold number")
    arg("--patience", type=positive_int, default=100, help="Early stopping patience")
    arg("--lr_patience", type=positive_int, default=70, help="Patience for ReduceLROnPlateau scheduler")
    arg("--batch_size", type=positive_int, default=2, help="Batch size")
    arg("--val_batch_size", type=positive_int, default=4, help="Validation batch size")
    arg("--steps", nargs="+", type=positive_int, required=False, help="Steps for multistep scheduler")
    arg("--profile", action="store_true", help="Run dlprof/PT profiling")
    arg("--profile_steps", type=str, default="90:95", help="PT profiling steps range separated by colon like 90:95")
    arg("--momentum", type=float, default=0.99, help="Momentum factor")
    arg("--weight_decay", type=float, default=0.0001, help="Weight decay (L2 penalty)")
    arg("--save_preds", action="store_true", help="Enable prediction saving")
    arg("--dim", type=int, choices=[2, 3], default=3, help="UNet dimension")
    arg("--resume_training", action="store_true", help="Resume training from the last checkpoint")
    arg("--factor", type=float, default=0.3, help="Scheduler factor")
    arg("--num_workers", type=non_negative_int, default=8, help="Number of subprocesses to use for data loading")
    arg("--min_epochs", type=non_negative_int, default=30, help="Force training for at least these many epochs")
    arg("--max_epochs", type=non_negative_int, default=10000, help="Stop training after this number of epochs")
    arg("--warmup", type=non_negative_int, default=5, help="Warmup iterations before collecting statistics")
    arg("--norm", type=str, choices=["instance", "batch", "group"], default="instance", help="Normalization layer")
    arg("--nvol", type=positive_int, default=1, help="Number of volumes which come into single batch size for 2D model")
    arg('--run-lazy-mode', default='True', type=lambda x: x.lower() == 'true',
         help='run model in lazy execution mode(enabled by default)'
         'Any value other than True(case insensitive) disables lazy mode')
    arg("--inference_mode", type=str, default="graphs", choices=["lazy", "graphs"], help="inference mode to run")
    arg('--autocast', dest='is_autocast', action='store_true', help='Enable autocast mode on Gaudi')
    arg('--habana_loader', action='store_true', help='Enable Habana Media Loader')
    arg("--bucket_cap_mb", type=positive_int, default=130, help="Size in MB for the gradient reduction bucket size")
    arg(
        "--data2d_dim",
        choices=[2, 3],
        type=int,
        default=3,
        help="Input data dimension for 2d model",
    )
    arg(
        "--oversampling",
        type=float_0_1,
        default=0.33,
        help="Probability of crop to have some region with positive label",
    )
    arg(
        "--overlap",
        type=float_0_1,
        default=0.5,
        help="Amount of overlap between scans during sliding window inference",
    )
    arg(
        "--affinity",
        type=str,
        default="disabled",
        choices=[
            "socket",
            "single",
            "single_unique",
            "socket_unique_interleaved",
            "socket_unique_continuous",
            "disabled",
        ],
        help="type of CPU affinity",
    )
    arg(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "multistep", "cosine", "plateau"],
        help="Learning rate scheduler",
    )
    arg(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["sgd", "radam", "adam", "adamw", "fusedadamw"],
        help="Optimizer",
    )
    arg(
        "--blend",
        type=str,
        choices=["gaussian", "constant"],
        default="gaussian",
        help="How to blend output of overlapping windows",
    )
    arg(
        "--train_batches",
        type=non_negative_int,
        default=0,
        help="Limit number of batches for training (used for benchmarking mode only)",
    )
    arg(
        "--test_batches",
        type=non_negative_int,
        default=0,
        help="Limit number of batches for inference (used for benchmarking mode only)",
    )
    arg("--progress_bar_refresh_rate", type=non_negative_int, default=25, help="set progress_bar_refresh_rate")
    arg('--set_aug_seed', dest='set_aug_seed', action='store_true', help='Set seed in data augmentation functions')
    arg('--no-augment', dest='augment', action='store_false')
    arg("--measurement_type", type=str, choices=["throughput", "latency"], default="throughput", help="Measurement mode for inference benchmark")
    arg("--use_torch_compile", action="store_true", help="Enable torch.compile")
    parser.set_defaults(augment=True)

    if strings is not None:
        arg(
            "strings",
            metavar="STRING",
            nargs="*",
            help="String for searching",
        )
        args = parser.parse_args(strings.split())
    else:
        args = parser.parse_args()

    if args.hpus and args.gpus:
        assert False, 'Cannot use both gpus and hpus'

    # Enable hpu dynamic shape
    if args.hpus:
        try:
            import habana_frameworks.torch.hpu as hthpu
            hthpu.enable_dynamic_shape()
        except ImportError:
            print("habana_frameworks could Not be loaded")
    if not args.hpus:
        args.run_lazy_mode = False
        if args.optimizer.lower() == 'fusedadamw':
            raise NotImplementedError("FusedAdamW is only supported for hpu devices.")

    return args



def main():
    args = get_main_args()
    if args.framework == 'pytorch-lightning':
        os.environ['framework'] = "PTL"
        from lightning_trainer.ptl import ptlrun
        ptlrun(args)
    elif args.framework == "pytorch":
        os.environ['framework'] ="NPT"
        from pytorch.npt import nptrun
        nptrun(args)
    else:
        print(f"please specify which framework would you like to run the model")

if __name__ == '__main__':
    main()


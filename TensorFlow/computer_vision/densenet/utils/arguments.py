###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse


class DenseNetArgumentParser(argparse.ArgumentParser):
    def __init__(self, description):
        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                         description=description, allow_abbrev=False)

        self.add_argument('--dataset_dir', type=str, default='/data/imagenet/tf_records',
                          help='path to dataset')
        self.add_argument('--model_dir', type=str, default='./',
                          help='directory for storing saved model and logs')
        self.add_argument('--dtype', type=str, choices=['fp32', 'bf16'], default='bf16',
                          help='data type used during training')
        self.add_argument('--dropout_rate', type=float, default=0.0,
                          help='(default: %(default)f)')
        self.add_argument('--optimizer', type=str, default='sgd',
                          choices=['sgd', 'adam', 'rmsprop'])
        self.add_argument('--batch_size', type=int, default=256,
                          help='(default: %(default)d)')
        self.add_argument('--initial_lr', type=float, default=1e-1,
                          help='(default: %(default)f)')
        self.add_argument('--weight_decay', type=float, default=1e-4,
                          help='(default: %(default)f)')
        self.add_argument('--epochs', type=int, default=90,
                          help='total number of epochs for training')
        self.add_argument('--steps_per_epoch', type=int, default=None,
                          help='number of steps per epoch')
        self.add_argument('--validation_steps', type=int, default=None,
                          help='number of validation steps, set to 0 to disable validation')
        self.add_argument('--model', type=str, default='densenet121',
                          choices=['densenet121', 'densenet161', 'densenet169'])
        self.add_argument('--train_subset', type=str, default='train')
        self.add_argument('--val_subset', type=str, default='validation')
        self.add_argument('--resume_from_checkpoint_path',
                          type=str, default=None,
                          help='path to checkpoint from which to resume training')
        self.add_argument('--resume_from_epoch', type=int, default=0,
                          help='from which epoch to resume training (used in '
                               'conjunction with resume_from_checkpoint_path argument)')
        self.add_argument('--evaluate_checkpoint_path', type=str, default=None,
                          help='checkpoint path for evaluating the model on --val_subset ')
        self.add_argument('--seed', type=int, default=None,
                          help='seed for randomness')
        self.add_argument('--warmup_epochs', type=int, default=5,
                          help='number of epochs with learning rate warmup')
        self.add_argument('--save_summary_steps', type=int, default=None,
                          help='steps between saving summaries to TensorBoard; '
                               'when None, logging to TensorBoard is disabled. '
                               '(enabling this option might affect the performance)')
        self.add_argument('--run_on_hpu', action='store_true',
                          help='whether to use HPU for training')
        self.add_argument('--use_hpu_strategy', action='store_true',
                          help='enables HPU strategy for distributed training')
        self.add_argument('--dump_config',
                          help='Side-by-side config file. Internal, do not use.')
        self.add_argument('--deterministic', action='store_true')
        self.add_argument('--inputs', type=str,
                          help="--inputs <Path to inputs>. required for deterministic mode")
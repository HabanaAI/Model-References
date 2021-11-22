###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import sys
import argparse
from pathlib import Path

def main():
    description = "This script is a single chip launcher for mobilenetv2 training and evaluation.\n "

    parser = argparse.ArgumentParser(description=description, add_help=False, usage=argparse.SUPPRESS)
    parser.add_argument('--computation_type',
                         type=str,
                         default="train")
    args, unknown_args = parser.parse_known_args()

    if args.computation_type == 'validation':
       script_to_run = "eval_image_classifier.py"
    else:
       script_to_run = "train_image_classifier.py"

    if '--help' in unknown_args or '-h' in unknown_args:
       print("""\nThis script is a single chip launcher for mobilenetv2 train (python demo_mobilenetv2.py) and evaluation (python demo_mobilenetv2.py --computation_type validation).""")
       if args.computation_type == 'validation':
           print("""\nvalidation examples:\n
           python demo_mobilenetv2.py --computation_type validation --dataset_dir /software/data/tf/data/mobilenetv2/validation
           python demo_resnet_keras.py --computation_type validation --dataset_dir /software/data/tf/data/mobilenetv2/validation --dtype bf16
           \nIn order to see all possible arguments to eval_image_classifier.py, run "python eval_image_classifier.py --helpfull"
           \nvalidation optional arguments:
           \nusage validation: python demo_mobilenetv2.py --computation_type validation [arguments]

           --max_num_batches <max_num_batches>                          Maxinum of batch number per evaluation step, defaults to 400
           --dataset_dir <dataset_dir>                                  Dataset dir, defaults to `/software/data/tf/data/mobilenetv2/validation`.
                                                                                     Needs to be specified if the above does not exists.
           --checkpoint_path <checkpoint_path>                          Checkpoint path for the trained models, defaults to /tmp/tfmodel (be the same as model_dir set in training)
           --eval_dir <eval_dir>                                        Evaluation directory. Default is /tmp/tfmodel/eval/
           """)
       else:
           print("""\ntrain examples:\n
           python demo_mobilenetv2.py --dataset_dir /software/data/tf/data/mobilenetv2/train
           python demo_resnet_keras.py --dataset_dir /software/data/tf/data/mobilenetv2/train --dtype bf16
           \nIn order to see all possible arguments to train_image_classifier.py, run "python train_image_classifier.py --helpfull"
           \ntrain optional arguments:
           \nusage train: python demo_mobilenetv2.py [arguments]
           --dataset_dir <dataset_dir>                                  Dataset dir, defaults to `/software/data/tf/data/mobilenetv2/train`.
                                                                                     Needs to be specified if the above does not exists.
           --learning_rate <learning_rate>                              Start learning rate, defaults to 0.045 for batch size 96
           --model_dir <model_dir>                                      The directory to save the trained models, defaults to /tmp/tfmodel
           --learning_rate <learning_rate>                              Start learning rate, defaults to 0.045 for batch size 96
           --clone_on_cpu <clone_on_cpu>                                Clone on CPU or not, defaults to False
           --num_clones <num_clones>                                    The number of clone of the model, defaults to 1
           --learning_rate_decay_factor <learning_rate_decay_factor>    Hyperparameter: default value is 0.98
           --max_number_of_steps <max_number_of_steps>                  The maxinum number of training steps, defaults to 500
           """)
       print("""
           --dtype <data_type>                                          Data type, possible values: fp32, bf16. Defaults to fp32
           --batch_size <batch_size>                                    Batch size, defaults to 96
           --dataset_name <dataset_name>                                Dataset name, defaults to imagenet
           --model_name <model_name>                                    The model name. Default is mobilenet_v2
           --preprocessing_name <preprocessing_name>                    Preprocessing name, defaults to inception_v2
           --label_smoothing <label_smoothing>                          Hyperparameter: default value is 0.1
           --moving_average_decay <moving_average_decay>                Hyperparameter: default value is 0.9999
           --num_epochs_per_decay <num_epochs_per_decay>                Hyperparameter: default value is 2.5 for batch size 96
       """)
       exit(0)

    command_to_run = [sys.executable, script_to_run]
    command_to_run.extend(unknown_args)
    command_str = ' '.join(command_to_run)

    print(f"Running: {command_str}", flush=True)
    os.system(command_str)

if __name__ == "__main__":
        main()

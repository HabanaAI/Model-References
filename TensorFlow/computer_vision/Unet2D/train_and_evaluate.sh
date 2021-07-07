###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
#
# This script runs 5-fold cross-validation of UNet2D topology for 6400 iterations
# Usage:
# bash examples/train_and_evaluate.sh <path/to/dataset> <path/for/results> <batch size> <data type> <number of HPUs>
python3 unet2d_demo.py --data_dir $1 --model_dir $2 --log_dir $2/fold_0 --max_steps 6400 --batch_size $3 --dtype $4 --exec_mode train_and_evaluate --fold 0 --hvd_workers $5 --tensorboard_logging --log_all_workers > $2/${4}_${5}hpu_fold0.log
python3 unet2d_demo.py --data_dir $1 --model_dir $2 --log_dir $2/fold_1 --max_steps 6400 --batch_size $3 --dtype $4 --exec_mode train_and_evaluate --fold 1 --hvd_workers $5 --tensorboard_logging --log_all_workers > $2/${4}_${5}hpu_fold1.log
python3 unet2d_demo.py --data_dir $1 --model_dir $2 --log_dir $2/fold_2 --max_steps 6400 --batch_size $3 --dtype $4 --exec_mode train_and_evaluate --fold 2 --hvd_workers $5 --tensorboard_logging --log_all_workers > $2/${4}_${5}hpu_fold2.log
python3 unet2d_demo.py --data_dir $1 --model_dir $2 --log_dir $2/fold_3 --max_steps 6400 --batch_size $3 --dtype $4 --exec_mode train_and_evaluate --fold 3 --hvd_workers $5 --tensorboard_logging --log_all_workers > $2/${4}_${5}hpu_fold3.log
python3 unet2d_demo.py --data_dir $1 --model_dir $2 --log_dir $2/fold_4 --max_steps 6400 --batch_size $3 --dtype $4 --exec_mode train_and_evaluate --fold 4 --hvd_workers $5 --tensorboard_logging --log_all_workers > $2/${4}_${5}hpu_fold4.log
python3 runtime/parse_results.py --model_dir $2 --env ${4}_${5}hpu

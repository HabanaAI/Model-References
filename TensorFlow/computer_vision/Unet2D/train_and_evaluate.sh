###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
#
# This script runs 5-fold cross-validation of UNet2D topology for 6400 iterations
# Usage:
# bash train_and_evaluate.sh <path/to/dataset> <path/for/results> <batch size> <data type> <number of HPUs>

if [ ! -d $2 ] # If the results path doesn't exist, create
then
  mkdir $2
fi

if [ $5 == 1 ] # Single card training
then
  $PYTHON unet2d.py --data_dir $1 --model_dir $2 --log_dir $2/fold_0 --max_steps 6400 --batch_size $3 --dtype $4 --exec_mode train_and_evaluate --fold 0 --tensorboard_logging > $2/${4}_${5}hpu_fold0.log
  $PYTHON unet2d.py --data_dir $1 --model_dir $2 --log_dir $2/fold_1 --max_steps 6400 --batch_size $3 --dtype $4 --exec_mode train_and_evaluate --fold 1 --tensorboard_logging > $2/${4}_${5}hpu_fold1.log
  $PYTHON unet2d.py --data_dir $1 --model_dir $2 --log_dir $2/fold_2 --max_steps 6400 --batch_size $3 --dtype $4 --exec_mode train_and_evaluate --fold 2 --tensorboard_logging > $2/${4}_${5}hpu_fold2.log
  $PYTHON unet2d.py --data_dir $1 --model_dir $2 --log_dir $2/fold_3 --max_steps 6400 --batch_size $3 --dtype $4 --exec_mode train_and_evaluate --fold 3 --tensorboard_logging > $2/${4}_${5}hpu_fold3.log
  $PYTHON unet2d.py --data_dir $1 --model_dir $2 --log_dir $2/fold_4 --max_steps 6400 --batch_size $3 --dtype $4 --exec_mode train_and_evaluate --fold 4 --tensorboard_logging > $2/${4}_${5}hpu_fold4.log
else # Multi card training
  mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --bind-to core --map-by socket:PE=6 -np $5 $PYTHON unet2d.py \
    --data_dir $1 --model_dir $2 --log_dir $2/fold_0 --max_steps 6400 --batch_size $3 --dtype $4 --exec_mode train_and_evaluate \
    --fold 0 --tensorboard_logging --log_all_workers --use_horovod > $2/${4}_${5}hpu_fold0.log
  mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --bind-to core --map-by socket:PE=6 -np $5 $PYTHON unet2d.py \
    --data_dir $1 --model_dir $2 --log_dir $2/fold_1 --max_steps 6400 --batch_size $3 --dtype $4 --exec_mode train_and_evaluate \
    --fold 1 --tensorboard_logging --log_all_workers --use_horovod > $2/${4}_${5}hpu_fold1.log
  mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --bind-to core --map-by socket:PE=6 -np $5 $PYTHON unet2d.py \
    --data_dir $1 --model_dir $2 --log_dir $2/fold_2 --max_steps 6400 --batch_size $3 --dtype $4 --exec_mode train_and_evaluate \
    --fold 2 --tensorboard_logging --log_all_workers --use_horovod > $2/${4}_${5}hpu_fold2.log
  mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --bind-to core --map-by socket:PE=6 -np $5 $PYTHON unet2d.py \
    --data_dir $1 --model_dir $2 --log_dir $2/fold_3 --max_steps 6400 --batch_size $3 --dtype $4 --exec_mode train_and_evaluate \
    --fold 3 --tensorboard_logging --log_all_workers --use_horovod > $2/${4}_${5}hpu_fold3.log
  mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --bind-to core --map-by socket:PE=6 -np $5 $PYTHON unet2d.py \
    --data_dir $1 --model_dir $2 --log_dir $2/fold_4 --max_steps 6400 --batch_size $3 --dtype $4 --exec_mode train_and_evaluate \
    --fold 4 --tensorboard_logging --log_all_workers --use_horovod > $2/${4}_${5}hpu_fold4.log
fi
$PYTHON runtime/parse_results.py --model_dir $2 --env ${4}_${5}hpu

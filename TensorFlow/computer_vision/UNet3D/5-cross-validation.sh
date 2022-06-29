###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################
#
# This script runs 5-fold cross-validation of UNet3D topology for 16000 iterations on single HLS
# Usage:
# bash 5-cross-validation.sh <path/to/dataset> <path/for/results> <batch size> <data type>
mpirun --allow-run-as-root --bind-to core --map-by socket:PE=4 --np 8 $PYTHON main.py --use_horovod --data_dir $1 --model_dir $2/fold_0 --log_dir $2/fold_0 --batch_size $3 --dtype $4 --fold 0 --tensorboard_logging --log_all_workers > $2/result_$4_fold0.log
mpirun --allow-run-as-root --bind-to core --map-by socket:PE=4 --np 8 $PYTHON main.py --use_horovod --data_dir $1 --model_dir $2/fold_1 --log_dir $2/fold_1 --batch_size $3 --dtype $4 --fold 1 --tensorboard_logging --log_all_workers > $2/result_$4_fold1.log
mpirun --allow-run-as-root --bind-to core --map-by socket:PE=4 --np 8 $PYTHON main.py --use_horovod --data_dir $1 --model_dir $2/fold_2 --log_dir $2/fold_2 --batch_size $3 --dtype $4 --fold 2 --tensorboard_logging --log_all_workers > $2/result_$4_fold2.log
mpirun --allow-run-as-root --bind-to core --map-by socket:PE=4 --np 8 $PYTHON main.py --use_horovod --data_dir $1 --model_dir $2/fold_3 --log_dir $2/fold_3 --batch_size $3 --dtype $4 --fold 3 --tensorboard_logging --log_all_workers > $2/result_$4_fold3.log
mpirun --allow-run-as-root --bind-to core --map-by socket:PE=4 --np 8 $PYTHON main.py --use_horovod --data_dir $1 --model_dir $2/fold_4 --log_dir $2/fold_4 --batch_size $3 --dtype $4 --fold 4 --tensorboard_logging --log_all_workers > $2/result_$4_fold4.log

$PYTHON runtime/parse_results.py --model_dir $2 --env result_$4

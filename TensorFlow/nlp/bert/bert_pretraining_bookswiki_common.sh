###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
###############################################################################

if [ "$USE_HOROVOD" == "true" ]; then
    # HCL Streams:ON by default
    export HABANA_USE_STREAMS_FOR_HCL=${HABANA_USE_STREAMS_FOR_HCL:-true}
    echo HABANA_USE_STREAMS_FOR_HCL=${HABANA_USE_STREAMS_FOR_HCL}

    # ART:ON by default
    export HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE=${HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE:-false}
    echo HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE=${HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE}

    # SAO:ON by default
    export TF_DISABLE_SCOPED_ALLOCATOR=${TF_DISABLE_SCOPED_ALLOCATOR:-false}
    echo TF_DISABLE_SCOPED_ALLOCATOR=$TF_DISABLE_SCOPED_ALLOCATOR

    export FORCE_WEIGHT_SYNC=${FORCE_WEIGHT_SYNC:-true}
    echo FORCE_WEIGHT_SYNC=$FORCE_WEIGHT_SYNC
fi

if [[ "$*" == --no_steps_accumulation ]]
then
    num_acc_steps_phase1=1
    num_acc_steps_phase2=1
    learning_rate_phase1=7.5e-4
    learning_rate_phase2=5.0e-4
else
    if [ "$FAST_PERF_ONLY" == "1" ]; then
        local global_batch_size1=1600
        local global_batch_size2=200
    else
        local global_batch_size1=65536
        local global_batch_size2=32768
    fi
    echo global_batch_sizes = $global_batch_size1, $global_batch_size2
    num_acc_steps_phase1=$(expr $global_batch_size1 \/ $NUM_WORKERS_TOTAL \/ $P1_BATCH)
    num_acc_steps_phase2=$(expr $global_batch_size2 \/ $NUM_WORKERS_TOTAL \/ $P2_BATCH)
    # Default value of learning rate argument (which is then scaled in run_pretraining.py
    # with the formula: effective_learning_rate = learning_rate * number_of_workers) for:
    # - 1st phase with global batch size = 64Ki and 8 workers is 7.5e-4,
    # - 2nd phase with global batch size = 32Ki and 8 workers is 5.0e-4.
    # According to global_batch_size/learning_rate = const, to compute learning rate of
    # number of workers and global batch size, we first multiply default value by
    # (8 / $NUM_WORKERS_TOTAL) and then by ($global_batch_size / 65536) for 1st phase, and,
    # respectively for second phase.
    learning_rate_phase1=$(echo "0.00075 * ( 8 / $NUM_WORKERS_TOTAL ) * ( $global_batch_size1 / 65536 )" | bc -l)
    learning_rate_phase2=$(echo "0.0005  * ( 8 / $NUM_WORKERS_TOTAL ) * ( $global_batch_size2 / 32768 )" | bc -l)
fi

export RESULTS_DIR=${OUTPUT_DIR:-$HOME/tmp/pretraining}

readonly TOTAL_DATASET_WORD_COUNT=1973248217 # with full bookcorpus should be around 3.3 billion

source ${SCRIPT_DIR}/pretraining/scripts/run_pretraining_lamb.sh \
    $P1_BATCH \
    $P2_BATCH \
    8 \
    $learning_rate_phase1 \
    $learning_rate_phase2 \
    fp32 \
    false \
    $NUM_WORKERS_PER_HLS \
    $P1_WARMUP \
    $P2_WARMUP \
    $P1_STEPS \
    $P2_STEPS \
    100 \
    $num_acc_steps_phase1 \
    $num_acc_steps_phase2 \
    base

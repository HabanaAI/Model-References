###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

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

source ${SCRIPT_DIR}/scripts/run_pretraining_lamb.sh \
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
    $SAVE_CKPT_STEPS \
    $num_acc_steps_phase1 \
    $num_acc_steps_phase2 \
    base

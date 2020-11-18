
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
fi

export RESULTS_DIR=$HOME/tmp/pretraining

readonly TOTAL_DATASET_WORD_COUNT=1973248217 # with full bookcorpus should be around 3.3 billion

export HABANA_INITIAL_WORKSPACE_SIZE_MB=6400
source ${SCRIPT_DIR}/pretraining/scripts/run_pretraining_lamb.sh \
    $P1_BATCH \
    $P2_BATCH \
    8 \
    7.5e-4 \
    5e-4 \
    fp32 \
    false \
    $NUM_WORKERS \
    $P1_WARMUP \
    $P2_WARMUP \
    $P1_STEPS \
    $P2_STEPS \
    100 \
    1 \
    1 \
    base

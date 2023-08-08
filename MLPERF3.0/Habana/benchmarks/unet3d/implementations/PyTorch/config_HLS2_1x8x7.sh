# Config parameters for 1xHLS2 (8xGaudi2) with batch_size=7

# System config params
export NUM_WORKERS_PER_HLS=8

# Hyper-parameters
export BATCH_SIZE=7
export START_EVAL_AT=1000
export EVALUATE_EVERY=20
export LR=1.5
export LR_WARMUP_EPOCHS=1200
export LR_DECAY_EPOCHS=1700
export LR_DECAY_FACTOR=0.5

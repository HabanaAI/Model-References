LEARNING_RATE=2e-5
DATASET_PATH="tensorflow_datasets/bert/MRPC"
export LOG_LEVEL_ALL=${LOG_LEVEL_ALL:-6}
echo LOG_LEVEL_ALL=$LOG_LEVEL_ALL
DEMODIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
source $DEMODIR/../../../common/common.sh
setup_libjemalloc

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR=$HOME/tmp/mrpc_output/
fi

function download_dataset()
{
    if [[ ! -d "${DATASET_PATH}" ]]; then
        printf "*** Downloading dataset...\n\n"
        mkdir -p "${DATASET_PATH}"
        python3 ${SCRIPT_DIR}/download/download_glue_data.py --data_dir "${DATASET_PATH}/.."  --tasks MRPC
    fi
}

run_per_ip download_dataset

function clean_temp_files()
{
    printf "*** Cleaning temp files in  ${OUTPUT_DIR}...\n\n"
    rm -rf ${OUTPUT_DIR}
}

run_per_ip clean_temp_files

printf "*** Running BERT training...\n\n"
export HABANA_INITIAL_WORKSPACE_SIZE_MB=15437
time $MPIRUN_CMD python3 ${SCRIPT_DIR}/run_classifier.py \
    --task_name=MRPC \
    --do_train=true \
    --do_eval=true \
    --data_dir=${DATASET_PATH} \
    --vocab_file=${PRETRAINED_MODEL}/vocab.txt \
    --bert_config_file=${PRETRAINED_MODEL}/bert_config.json \
    --init_checkpoint=${PRETRAINED_MODEL}/bert_model.ckpt \
    --max_seq_length=$MAX_SEQ_LENGTH \
    --train_batch_size=$TRAIN_BATCH \
    --learning_rate=$LEARNING_RATE \
    --num_train_epochs=$TRAIN_EPOCHS \
    --output_dir=$OUTPUT_DIR \
    --use_horovod=$USE_HOROVOD

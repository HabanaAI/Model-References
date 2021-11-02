LEARNING_RATE=3e-5
export LOG_LEVEL_ALL=${LOG_LEVEL_ALL:-6}
echo LOG_LEVEL_ALL=$LOG_LEVEL_ALL
DEMODIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
source $DEMODIR/../../../common/common.sh
run_per_ip setup_libjemalloc

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR=$HOME/tmp/squad_$MODEL_TYPE/
fi

function prepare_output_dir()
{
    # tf_record generation by run_squad.py script takes ~5 minutes for base model, so if possible, let's keep this file
    # solution should also work large model in the future, but make sure of it here
    if [[ -d "${OUTPUT_DIR}" ]]; then
        if [[ ! -f "${OUTPUT_DIR}/last_config_${TRAIN_BATCH}_${MAX_SEQ_LENGTH}" ]]; then
            printf "*** Cleaning temp directory content in ${OUTPUT_DIR}... \n\n"
            ls ${OUTPUT_DIR}/* | xargs rm -rf
        else
            printf "*** Cleaning temp directory content in ${OUTPUT_DIR}... (except *.tf_record files) \n\n"
            ls ${OUTPUT_DIR}/* | grep -v .tf_record | xargs rm -rf
        fi
    else
        mkdir -p ${OUTPUT_DIR}
    fi

    touch "${OUTPUT_DIR}/last_config_${TRAIN_BATCH}_${MAX_SEQ_LENGTH}"
}

run_per_ip prepare_output_dir

printf "*** Running BERT training...\n\n"
SQUAD_DIR=${SCRIPT_DIR}

if [ "${TF_BF16_CONVERSION:=0}" == "0" ]; then
    if [ "${MODEL_TYPE}" == "large" ]; then
        export HABANA_INITIAL_WORKSPACE_SIZE_MB=17257
    else
        export HABANA_INITIAL_WORKSPACE_SIZE_MB=13271
    fi
else
    if [ "${MODEL_TYPE}" == "large" ]; then
        export HABANA_INITIAL_WORKSPACE_SIZE_MB=21393
    else
        export HABANA_INITIAL_WORKSPACE_SIZE_MB=17371
    fi
fi
time $MPIRUN_CMD python3 ${SCRIPT_DIR}/run_squad.py \
  --vocab_file=${PRETRAINED_MODEL}/vocab.txt \
  --bert_config_file=${PRETRAINED_MODEL}/bert_config.json \
  --init_checkpoint=${PRETRAINED_MODEL}/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/data/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/data/dev-v1.1.json \
  --do_eval=True \
  --train_batch_size=$TRAIN_BATCH \
  --learning_rate=$LEARNING_RATE \
  --num_train_epochs=$TRAIN_EPOCHS \
  --max_seq_length=$MAX_SEQ_LENGTH \
  --doc_stride=128 \
  --output_dir=$OUTPUT_DIR \
  --use_horovod=$USE_HOROVOD

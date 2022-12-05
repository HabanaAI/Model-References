###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

# script for this section: https://github.com/HabanaAI/Model-References/tree/master/MLPERF/Habana/benchmarks#training-data-for-bert

# Inputs:
# $1 is WORK_DIR: Location where temp files might be downloaded, and where final tf records will be created

# sample run command:
# ./prep_mlperf_bert_data.sh workspace_bert

# Outputs:
# workspace_bert/Model-References/MLPERF/Habana/benchmarks/bert/data/

# Do not set this env var, during actual run, this is only for fast testing purposes
# NUM_LOOPS=i (i is expected to be in [0-499] inclusive)
# See its use in step 4/5

if [[ ! $NUM_LOOPS ]]; then
    NUM_LOOPS=499
else
    if [ $NUM_LOOPS -gt 499 ]; then
        echo "NUM_LOOPS expected to be between 0-499 inclusive, but got $NUM_LOOPS"
        exit 1
    fi
    if [ $NUM_LOOPS -lt 0 ]; then
        echo "NUM_LOOPS expected to be between 0-499 inclusive, but got $NUM_LOOPS"
        exit 1
    fi
fi

if [ "$#" -ne 1 ]; then
    echo "Expected 1 argument, a workspace folder, got $# arguments instead"
    echo "Sample command line: ./prep_mlperf_bert_data.sh workspace_bert"
    exit 1
fi

WORK_DIR=$1


if [ -d "$WORK_DIR" ]; then
    echo "Working dir $WORK_DIR already exists. Please delete it or give path to a non-existing directory"
    exit 1
fi
mkdir -p $WORK_DIR
python3.6 -m venv $WORK_DIR/venv
source $WORK_DIR/venv/bin/activate
pip3 install --upgrade pip setuptools

pip install gdown # to download from google drive

# step 1 #########################
echo "step 1"
CURRENT=`pwd`
git clone https://github.com/mlcommons/training.git $WORK_DIR/mlcommons_training

# step 2 #########################
echo "step 2"
cd $WORK_DIR/mlcommons_training/language_model/tensorflow/bert/cleanup_scripts
mkdir tfrecord_dir
mkdir eval_intermediate
mkdir mlperf_bert


# step 3 #########################
echo "step 3"
google_drive_download()
{
    # Other alternatives of downloading (without gdown dependency). But gdown is good for large file downloads
    # https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99
    # https://drive.google.com/file/d/1fbGClQMi2CoMv7fwrwTC5YYPooQBdcFW/view?usp=sharing
    # wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fbGClQMi2CoMv7fwrwTC5YYPooQBdcFW' -O bert_config.json
    echo "Downloading from google drive https://docs.google.com/uc?export=download&id=$1"
    gdown "https://docs.google.com/uc?export=download&id=$1"
}

google_drive_download 1fbGClQMi2CoMv7fwrwTC5YYPooQBdcFW #bert_config.json
google_drive_download 1USK108J6hMM_d27xCHi738qBL8_BT1u1 #vocab.txt


# step 4 #########################
echo "step 4"
google_drive_download 1dy_Jt3s6CYy6SqAJeIveI1P2msSNlhBH #results_text.zip

if [ $NUM_LOOPS -eq 499 ]; then
    # Full data extract
    unzip results_text.zip
else
    # Fast path, just extract the zip partially
    j=0
    mkdir results4
    for H in {0..4}
    do
        for T in {0..9}
        do
            for O in {0..9}
            do
                if [ $j -le $NUM_LOOPS ]; then
                    unzip -p results_text.zip results4/part-00${H}${T}${O}-of-00500 > results4/part-00${H}${T}${O}-of-00500
                    ((j=j+1))
                fi
            done
        done
    done
    unzip -p results_text.zip results4/eval.txt > results4/eval.txt
fi

# step 5 #########################
echo "step 5"
RESULTS=`pwd`/results4
TFRECORD_DIR=`pwd`/tfrecord_dir
EVAL=`pwd`/eval_10k
VOCAB=`pwd`/vocab.txt
BERT_CONFIG=`pwd`/bert_config.json
pip install absl-py
pip install tensorflow

i=0
# TODO: right now running 2 jobs in parallel. Can we fire more?
for H in {0..4}
do
    for T in {0..9}
    do
        for O in {0..4}
        do
            if [ $i -le $NUM_LOOPS ]; then # Set NUM_LOOPS to a small number for a quick test run
                # There are 500 create_pretraining_data that need to be fired. firing 2 of then parallely
                echo "Launch $((100*H+10*T+2*O))"
                python3 create_pretraining_data.py --input_file=$RESULTS/part-00${H}${T}$((2*O))-of-00500 \
                --output_file=$TFRECORD_DIR/part-00${H}${T}$((2*O))-of-00500 \
                --vocab_file=$VOCAB \
                --do_lower_case=True \
                --max_seq_length=512 \
                --max_predictions_per_seq=76 \
                --masked_lm_prob=0.15 \
                --random_seed=12345 \
                --dupe_factor=10&
                ((i=i+1))

                if [ $i -le $NUM_LOOPS ]; then
                    echo "Launch $((100*H+10*T+2*O+1))"
                    python3 create_pretraining_data.py --input_file=$RESULTS/part-00${H}${T}$((2*O+1))-of-00500 \
                    --output_file=$TFRECORD_DIR/part-00${H}${T}$((2*O+1))-of-00500 \
                    --vocab_file=$VOCAB \
                    --do_lower_case=True \
                    --max_seq_length=512 \
                    --max_predictions_per_seq=76 \
                    --masked_lm_prob=0.15 \
                    --random_seed=12345 \
                    --dupe_factor=10&
                    ((i=i+1))
                fi
                wait
            fi
        done
    done
done


# step 6 #########################
echo "step 6"
python3 create_pretraining_data.py --input_file=results4/eval.txt \
    --output_file=eval_intermediate/eval_10k --vocab_file=vocab.txt \
    --do_lower_case=True --max_seq_length=512 --max_predictions_per_seq=76 \
    --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=10

if [ $NUM_LOOPS -eq 499 ]; then
    PICK=10000
else
    PICK=100
fi
python3 pick_eval_samples.py --input_tfrecord=eval_intermediate/eval_10k \
    --output_tfrecord=eval_10k --num_examples_to_pick=$PICK


# step 7 #########################
echo "step 7"
cd $CURRENT
git clone https://github.com/HabanaAI/Model-References.git $WORK_DIR/Model-References
cd $WORK_DIR/Model-References/MLPERF/Habana/benchmarks/bert
export DATA=`pwd`/data
echo "DATA dir: $DATA"
mkdir -p $DATA/bert_pretraining $DATA/train/packed_data_500 \
  $DATA/mlperf_bert_eval_dataset $DATA/MLPerf_BERT_checkpoint \
  mlperf_result
cp $BERT_CONFIG $DATA/MLPerf_BERT_checkpoint/bert_config.json

# step 8 #########################
echo "step 8"
pip install pandas
pip install scipy
pip install matplotlib
python3 scripts/pack_pretraining_data_tfrec.py \
  --input-glob "$TFRECORD_DIR/" \
  --output-dir $DATA/train/packed_data_500 \
  --max-files $NUM_LOOPS

# step 9 #########################
echo "step 9"
mv $TFRECORD_DIR $DATA/bert_pretraining/training
mv $EVAL $DATA/mlperf_bert_eval_dataset/eval_10k


# step 10 #########################
echo "step 10"
cd $DATA/MLPerf_BERT_checkpoint
# tf 2
#google_drive_download 11VyBm4xTPutIuO---NgpOg0Mu-eSyXHs #License.txt
#google_drive_download 1pJhVkACK3p_7Uc-1pAzRaOXodNeeHZ7F #model.ckpt-28252.data-00000-of-00001
#google_drive_download 1oVBgtSxkXC9rH2SXJv85RXR9-WrMPy-Q #model.ckpt-28252.index

# tf 1
google_drive_download 1QTLMFQn4HSMNQnoD3Bgs_IHkY_jE2ujZ #License.txt
google_drive_download 1chiTBljF0Eh1U5pKs6ureVHgSbtU8OG_ #model.ckpt-28252.data-00000-of-00001
google_drive_download 1Q47V3K3jFRkbJ2zGCrKkKk-n0fvMZsa0 #model.ckpt-28252.index
google_drive_download 1vAcVmXSLsLeQ1q7gvHnQUSth5W_f_pwv #model.ckpt-28252.meta


deactivate

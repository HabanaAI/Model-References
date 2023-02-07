#!/bin/bash
# Copyright (c) 2019-2022 NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
# Copyright (c) 2022, Habana Labs Ltd.  All rights reserved.
###############################################################################

function usage()
{
   cat << HEREDOC

   Usage: $progname [-o|--outputdir PATH] [-h|--help TIME_STR]

   optional arguments:
     -h, --help            show this help message and exit
     -o, --outputdir PATH  pass in a localization of resulting dataset
     -s, --skip-download   skip downloading raw files from GDrive (assuming it already has been done)
     -p, --shards          number of resulting shards. For small scales (less than 256 nodes) use 2048. For sacles >256 4320 is recommended (default 4320)

HEREDOC
}

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

#if no arguments passed
DATADIR=/workspace/bert_data
SKIP=0
SHARDS=4320

#parse passed arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -h|--help)
      usage
      exit 0
      ;;
    -o|--outputdir)
      DATADIR="$2"
      shift # past argument
      shift # past value
      ;;
    -p|--shards)
      SHARDS="$2"
      shift # past argument
      shift # past value
      ;;
    -s|--skip-download)
      SKIP=1
      shift
      ;;
    *)    # unknown option
      usage
      exit 1
      ;;
  esac
done


echo "Preparing Mlperf BERT dataset in ${DATADIR}"
mkdir -p ${DATADIR}

if (( SKIP==0 )) ; then
    
    mkdir -p ${DATADIR}/phase1 && cd ${DATADIR}/phase1
    ### Download 
    # bert_config.json
    gdown https://drive.google.com/uc?id=1fbGClQMi2CoMv7fwrwTC5YYPooQBdcFW    
    # vocab.txt
    gdown https://drive.google.com/uc?id=1USK108J6hMM_d27xCHi738qBL8_BT1u1
    
    ### Download dataset
    mkdir -p ${DATADIR}/download && cd ${DATADIR}/download
    # md5 sums
    gdown https://drive.google.com/uc?id=1tmMgLwoBvbEJEHXh77sqrXYw5RpqT8R_
    # processed chunks
    gdown https://drive.google.com/uc?id=14xV2OUGSQDG_yDBrmbSdcDC-QGeqpfs_
    # unpack results and verify md5sums
    tar -xzf results_text.tar.gz && (cd results4 && md5sum --check ../bert_reference_results_text_md5.txt)
    
    
    ### Download TF1 checkpoint
    mkdir -p ${DATADIR}/phase1 && cd ${DATADIR}/phase1
    # model.ckpt-28252.data-00000-of-00001
    gdown https://drive.google.com/uc?id=1chiTBljF0Eh1U5pKs6ureVHgSbtU8OG_
    # model.ckpt-28252.index
    gdown https://drive.google.com/uc?id=1Q47V3K3jFRkbJ2zGCrKkKk-n0fvMZsa0
    # model.ckpt-28252.meta
    gdown https://drive.google.com/uc?id=1vAcVmXSLsLeQ1q7gvHnQUSth5W_f_pwv
    
    cd ${DATADIR}
    
fi
### Create HDF5 files for training
mkdir -p ${DATADIR}/hdf5/training
bash ${SCRIPT_DIR}/parallel_create_hdf5.sh -i ${DATADIR}/download/results4 -o ${DATADIR}/hdf5/training -v ${DATADIR}/phase1/vocab.txt

### Chop HDF5 files into chunks
ulimit -n 10000 # handles potential OSError Too many open files
python3 ${SCRIPT_DIR}/chop_hdf5_files.py \
 --num_shards ${SHARDS} \
 --input_hdf5_dir ${DATADIR}/hdf5/training \
 --output_hdf5_dir ${DATADIR}/hdf5/training-${SHARDS}

### Convert fixed length to variable length format
mkdir -p ${DATADIR}/hdf5/training-${SHARDS}/hdf5_${SHARDS}_shards_varlength
CPUS=$( ls -d /sys/devices/system/cpu/cpu[[:digit:]]* | wc -w )
CPUS=$((CPUS / 2))
ls -1 ${DATADIR}/hdf5/training-${SHARDS}/hdf5_${SHARDS}_shards_uncompressed | \
  xargs --max-args=1 --max-procs=${CPUS} -I{} python3 ${SCRIPT_DIR}/convert_fixed2variable.py \
  --input_hdf5_file ${DATADIR}/hdf5/training-${SHARDS}/hdf5_${SHARDS}_shards_uncompressed/{} \
  --output_hdf5_file ${DATADIR}/hdf5/training-${SHARDS}/hdf5_${SHARDS}_shards_varlength/{}

#### Create full HDF5 files for evaluation
mkdir -p ${DATADIR}/hdf5/eval
python3 ${SCRIPT_DIR}/create_pretraining_data.py \
 --input_file=${DATADIR}/download/results4/eval.txt \
 --output_file=${DATADIR}/hdf5/eval/eval_all \
 --vocab_file=${DATADIR}/phase1/vocab.txt \
 --do_lower_case=True \
 --max_seq_length=512 \
 --max_predictions_per_seq=76 \
 --masked_lm_prob=0.15 \
 --random_seed=12345 \
 --dupe_factor=10

#### pick 10k samples for evaluation
python3 ${SCRIPT_DIR}/pick_eval_samples.py \
 --input_hdf5_file=${DATADIR}/hdf5/eval/eval_all.hdf5 \
 --output_hdf5_file=${DATADIR}/hdf5/eval/part_eval_10k \
 --num_examples_to_pick=10000

#### Convert fixed length to variable length format
mkdir -p ${DATADIR}/hdf5/eval_varlength
python3 ${SCRIPT_DIR}/convert_fixed2variable.py --input_hdf5_file ${DATADIR}/hdf5/eval/part_eval_10k.hdf5 \
  --output_hdf5_file ${DATADIR}/hdf5/eval_varlength/part_eval_10k.hdf5

#### Convert Tensorflow checkpoint to Pytorch one
python3 ${SCRIPT_DIR}/../convert_tf_checkpoint.py \
  --tf_checkpoint ${DATADIR}/phase1/model.ckpt-28252 \
  --bert_config_path ${DATADIR}/phase1/bert_config.json \
  --output_checkpoint ${DATADIR}/phase1/model.ckpt-28252.pt

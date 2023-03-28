#!/bin/bash
###############################################################################
# Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
###############################################################################

BASE_DIR=`dirname -- "$0"`/..

FILE=$1
shift

OPTIONS_RE='([-0-9a-z]+m)\.([0-9]+)b\.([0-9]+)\.([a-z0-9]+)'
[[ $FILE =~ $OPTIONS_RE ]]
    
MODEL=${BASH_REMATCH[1]}
BEAMS=${BASH_REMATCH[2]}
MAX_LEN=${BASH_REMATCH[3]}
DTYPE=${BASH_REMATCH[4]}

${BASE_DIR}/bloom.py --input_file ${BASE_DIR}/samples/wmt14.json --reference_file ${FILE} --model ${MODEL} --beams ${BEAMS} --max_length ${MAX_LEN} --dtype ${DTYPE} $*

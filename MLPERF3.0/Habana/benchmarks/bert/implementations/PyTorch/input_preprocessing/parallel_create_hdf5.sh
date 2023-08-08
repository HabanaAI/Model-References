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

CPUS=$( ls -d /sys/devices/system/cpu/cpu[[:digit:]]* | wc -w )
CPUS=$((CPUS / 2))
echo "Using ${CPUS} CPU cores"

function usage()
{
   cat << HEREDOC

   Usage: $progname [-i|--inputdir PATH -o|--outputdir PATH -v|--vocab VOCAB-PATH] [-h|--help TIME_STR]

   optional arguments:
     -h, --help            show this help message and exit
     -o, --outputdir PATH  pass in a localization of resulting dataset
     -i, --inputdir PATH   pass in a localization of resulting hdf5 files
     -v, --vocab PATH pass in exact path to vocabulary file

HEREDOC
}


#parse passed arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -h|--help)
      usage
      exit 0
      ;;
    -o|--outputdir)
      OUTPUTDIR="$2"
      shift # past argument
      shift # past value
      ;;
    -i|--inputdir)
      INPUTDIR="$2"
      shift
      shift
      ;;
    -v|--vocab)
      VOCAB="$2"
      shift
      shift
      ;;
    *)    # unknown option
      usage
      exit 1
      ;;
  esac
done

# get script reference directory 
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"


mkdir -p ${OUTPUTDIR}
find -L ${INPUTDIR} -name "part-00*" | xargs --max-args=1 --max-procs=${CPUS} -I{}  ${SCRIPT_DIR}/create_pretraining_data_wrapper.sh {} ${OUTPUTDIR} ${VOCAB}

### If continue, you can try instead of line above something like line below to pick only the files not yet computed 
# comm -3 <(ls -1 ${INPUTDIR}/) <(ls -1 ${OUTPUTDIR} | sed 's/\.hdf5$//') | grep -e "^part" | xargs --max-args=1 --max-procs=${CPUS} -I{}  ${SCRIPT_DIR}/create_pretraining_data_wrapper.sh ${INPUTDIR}/{} ${OUTPUTDIR} ${VOCAB}

#!/bin/bash

###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

[[ $0 != $BASH_SOURCE ]] || echo "This script must be sourced!"

export MLPERF_INFERENCE_CODE_DIR=$(realpath $(dirname $BASH_SOURCE) )
if [ -f "$MLPERF_INFERENCE_CODE_DIR/tests.sh" ]; then
      source "$MLPERF_INFERENCE_CODE_DIR/tests.sh"
fi
function mlperf_inference_usage()
{
    echo -e "\n usage: build_mlperf_inference [options]\n"
    echo -e "options:\n"
    echo -e "  --output-dir                Path to save logs, results and summary; optional"
    echo -e "  --skip-reqs                 Skip installing requirements, downloading MLCommons Inference and building loadgen; optional"
    echo -e "  --compliance                Create a submission package compliant with MLCommons submission checker; optional"
    echo -e "  --submission                List of scenarios to run; optional"
    echo -e "  -h,  --help                 Prints this help"

}

build_mlperf_inference()
{
    set -e
    output_dir=$(pwd)/results
    submission_args=""
    compliance=false
    skip_reqs=false

    while [ -n "$1" ];
    do
        case $1 in

            -h  | --help )
                mlperf_inference_usage
                return 0
            ;;
            --output-dir )
                output_dir=$2
                shift 2
            ;;
            --compliance )
                compliance=true
                shift 1
            ;;
            --skip-reqs )
                shift
                skip_reqs=true
            ;;
            --submission )
                shift
                submission_args=$@
                break
            ;;
        esac
    done

    if [ "$skip_reqs" == "false" ]; then
        model_name=""
        if [ -n "$submission_args" ]; then
            pushd $MLPERF_INFERENCE_CODE_DIR
            model_name=$(python -c 'import yaml,sys;sc = sys.argv[1].split("_");print(yaml.full_load(open("scenarios.yaml"))["scenarios"][sc[0]]["code_dir"])' $submission_args)
            pushd $MLPERF_INFERENCE_CODE_DIR/$model_name/
            pip install -r requirements.txt
            popd
            popd
            pip list
        fi

        BUILD_DIR=$(mktemp -d -t mlperf.XXXX)
        pushd $BUILD_DIR
        git clone --depth 1 --recurse-submodules https://github.com/mlcommons/inference.git mlcommons_inference
        cd mlcommons_inference/loadgen
        CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel
        cd ..; pip install --force-reinstall loadgen/dist/`ls -r loadgen/dist/ | head -n1` ; cd -
        popd
    fi

    if [ ! -z "$submission_args" ]; then
        pushd $MLPERF_INFERENCE_CODE_DIR
        if [ "$compliance" == "true"  ]; then
            python run_mlperf_scenarios.py $submission_args --output-dir $output_dir --mlperf-path $BUILD_DIR/mlcommons_inference --compliance
            python prepare_and_check_submission.py $submission_args --output-dir $output_dir --mlperf-path $BUILD_DIR/mlcommons_inference --systems-dir-path $MLPERF_INFERENCE_CODE_DIR/../systems --measurements-dir-path $MLPERF_INFERENCE_CODE_DIR/../measurements
        else
            python run_mlperf_scenarios.py $submission_args --output-dir $output_dir
        fi
        popd

    fi

    rm -rf $BUILD_DIR
}

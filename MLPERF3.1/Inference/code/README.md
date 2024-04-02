# Habana MLPerf™ inference submission
This directory provides instructions to reproduce Habana's results for MLPerf™ inference submission.\
MLPerf™ is a trademark and service mark of MLCommons Association in the United States and other countries.\
All rights reserved. Unauthorized use is strictly prohibited.

- [Habana MLPerf™ inference submission](#habana-mlperf-inference-submission)
  - [Setup](#setup)
    - [Prepare MLPerf Directory](#prepare-mlperf-directory)
    - [Build and Deploy HabanaLabs Container](#build-and-deploy-habanalabs-container)
    - [Download Checkpoint](#download-checkpoint)
    - [Download Dataset](#download-dataset)
  - [Reproduce Results](#reproduce-results)
    - [99 and 99.9 Accuracy](#99-and-999-accuracy)
    - [Get Started](#get-started)
    - [Generate Results](#generate-results)
  - [Performance Optimization with FP8 Flow](#performance-optimization-with-fp8-flow)
    - [Environment Variables](#environment-variables)
  - [Supported Configurations](#supported-configurations)
  - [Changelog](#changelog)

## Setup

Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the environment.

### Prepare MLPerf Directory

Perform the following:

1. Follow the instructions provided in the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the benchmarks on Gaudi.

2. Clone Model-References repository and switch to the branch that matches your SynapseAI version. You can run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.

    ```bash
    export MLPERF_ROOT=/path/to/mlperf/root
    cd $MLPERF_ROOT
    git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
    export MLPERF_DIR=$MLPERF_ROOT/Model-References/MLPERF3.1/Inference
    ```

### Build and Deploy HabanaLabs Container

To build MLPerf inference 3.1 container, perform the following:

1. Set the environment variables for the docker command.
   * To find a docker image, go to [gaudi-docker](https://vault.habana.ai/ui/repos/tree/General/gaudi-docker).
   * Open gaudi-docker directory, and select the folder that matches the SynapseAI version (determined by running [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)).
   * Navigate to subdirectories, choose system and framework version.
   * Choose the docker build version. Most often 'latest' will be used.
   * Navigate to "Docker Info" tab and note "Title" string.
   * Set `DOCKER_IMAGE` to "Title" string with `vault.habana.ai/gaudi-docker/` prefix. See the examples below.
      * Example on PyTorch Container:
          ```bash
          # NOTE: The below is only an example value. Replace [SynapseAI version] and [PT version] to match your setup and Supported Configuration.
          export DOCKER_IMAGE=vault.habana.ai/gaudi-docker/[SynapseAI version]/ubuntu20.04/habanalabs/pytorch-installer-[PT Version]:latest
          ```


2. Create `mlperf-habana container` by running the following command.

```bash
docker run --privileged --security-opt seccomp=unconfined \
           --name mlperf-habana -td                \
           -v /dev:/dev                            \
           --device=/dev:/dev                      \
           -v /sys/kernel/debug:/sys/kernel/debug  \
           -v /tmp:/tmp                            \
           -v $MLPERF_DIR:/root/Habana/            \
           --cap-add=sys_nice --cap-add=SYS_PTRACE \
           --user root --workdir=/root --net=host  \
           --ulimit memlock=-1:-1 ${DOCKER_IMAGE}
```

3. Start the docker.
```bash
docker exec -it mlperf-habana bash
```

### Download Checkpoint
```bash
mkdir -p /mnt/weka/data/pytorch/
pushd /mnt/weka/data/pytorch/
wget https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download  --output-document checkpoint.zip
unzip -q checkpoint.zip && rm checkpoint.zip
popd
```

### Download Dataset
```bash
pushd /root/Habana/code/gptj-99.9/gpt-j
python download_cnndm.py
cp data/cnn_eval.json /mnt/weka/data/pytorch/gpt-j/cnn_eval.json
popd
```

##  Reproduce Results
### 99 and 99.9 Accuracy
The same script was submitted for both 99 and 99.9 benchmarks - no additional improvements were made for low accuracy (99), and 99.9 results were used for 99 as well.

### Get Started
Install the requirements and build the latest loadgen.

```bash
cd /root/Habana/code
source functions.sh
build_mlperf_inference
```
### Generate Results
**To generate full submission results, run the following command:**
```bash
build_mlperf_inference --output-dir <path_to_output_dir> --submission gptj-99.9-fp8
```
The command produces results from accuracy and performance runs for both Offline and Server scenarios.
Logs can be found under /output_dir/logs/model/, e.g. /results/logs/gptj-99.9-fp8/


**To generate results for Offline and Server scenarios separately, run the following commands:**
```bash
build_mlperf_inference --output-dir <path_to_output_dir> --submission gptj-99.9-fp8_Offline
```

```bash
build_mlperf_inference --output-dir <path_to_output_dir> --submission gptj-99.9-fp8_Server
```
Logs can be found under /output_dir/logs/model/scenario/, e.g. /results/logs/gptj-99.9-fp8/Offline/

**To generate results for accuracy and performance separately, add ```--mode``` flag as in one of the following commands:**
```bash
build_mlperf_inference --output-dir <path_to_output_dir> --submission gptj-99.9-fp8_Server --mode acc
```
```bash
build_mlperf_inference --output-dir <path_to_output_dir> --submission gptj-99.9-fp8_Offline --mode perf
```

Logs can be found under /output_dir/logs/model/scenario/mode/, e.g. /results/logs/gptj-99.9-fp8/Offline/accuracy/

## Performance Optimization with FP8 Flow
To optimize performance, we set heavy-performance ops to operate in fp8-143.

All fp8 ops are working with a fixed fp8 exponent bias = 7 and no scaling is required.

### Environment Variables
The following outlines custom ENV variables used in the GPT-J submission script:

| Enviroment Variable                                                   	| Effect                                                                                                                                                                  	|
|-------------------------------------------------------------------------	|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| PT_USE_FP8_143=1                                                        	| Sets PT backend fp8 flavor to fp8_143                                                                                                                                   	|
| UPDATE_MME_OUTPUT_PRECISION_FILTER="v_proj,matmul_av"                   	| Allows the specified MME layer to output fp8 for performance optimization.                                                                                              	|
| SCALES_FILE_PATH=quantization/measurements/per_tensor_scales_gpt_j.json 	| Loads per-tensor scales required for fp8 quantization. If not provided, no scaling is applied.                                                                          	|
| ENABLE_EXPERIMENTAL_FLAGS=true                                          	| Enables the above flags                                                                                                                                                     	|

## Supported Configurations

| Validated on | SynapseAI Version | Framework Version(s) |   Mode   |
| :----------: | :---------------: | :------------------: | :------: |
|    Gaudi2    |      1.15.1       |    PyTorch 2.2.0     | Inference |

## Changelog
### 1.13.0
- Published MLPerf™ inference 3.1 GPT-J script
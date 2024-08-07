# Running Intel-HabanaLabs MLPerf™ Llama-70B LoRA Benchmark

This directory provides instructions to reproduce Intel-HabanaLabs's results for [MLPerf Training v4.0](https://habana.ai/since-habanas-last-mlperf-submission/) Llama-70B LoRA benchmark on single server with 8 Gaudi2 cards.

For more information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/)

MLPerf™ is a trademark and service mark of MLCommons Association in the United States and other countries. All rights reserved. Unauthorized use is strictly prohibited.

- [Running Intel-HabanaLabs MLPerf™ Llama-70B LoRA Benchmark](#running-intel-habanalabs-mlperf-llama-70b-lora-benchmark)
  - [Setup](#setup)
    - [Prepare MLPerf Directory](#prepare-mlperf-directory)
    - [Build and Deploy Intel-HabanaLabs MLPerf Training 4.0 Container](#build-and-deploy-intel-habanalabs-mlperf-training-40-container)
    - [Download Data and Model](#download-data-and-model)
  - [Finetuning Llama2 70B with LoRA](#finetuning-llama2-70b-with-lora)

## Setup

Make sure to have requested permission for donwloading Llama2 weights on the Hugging Face Hub: https://huggingface.co/meta-llama/Llama-2-7b-hf

### Prepare MLPerf Directory

On each compute node, perform the following:

1. Follow the instructions provided in the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the benchmarks on Gaudi.

1. Create directories for dataset and logs:
    ```
    export MLPERF_DIR=/path/to/mlperf/root
    export DATASETS_DIR=/path/to/datasets
    export MODEL_DIR=/path/to/model
    mkdir -p $MLPERF_DIR/Intel-HabanaLabs $MODEL_DIR $DATASETS_DIR
    ```

2. This README is located in `benchmarks/llm_finetune` directory corresponding to Intel-HabanaLabs's Llama-70B LoRA submission.
Download this whole `benchmarks` folder along with all subfolders and copy it under `$MLPERF_DIR/Intel-HabanaLabs`

### Build and Deploy Intel-HabanaLabs MLPerf Training 4.0 Container

1. Create `mlperf4.0` container by running the following command.

- TODO: update `DOCKER_IMAGE` once it is known and published.

    ```bash
    export CONTAINER_NAME=mlperf4.0
    export DOCKER_IMAGE=vault.habana.ai/gaudi-docker-mlperf/ver4.0/pytorch-installer-2.2.0:1.16.98-46
    docker run --privileged --security-opt seccomp=unconfined \
      --name $CONTAINER_NAME -td                              \
      -v /dev:/dev                                            \
      --device=/dev:/dev                                      \
      -e LOG_LEVEL_ALL=6                                      \
      -v /sys/kernel/debug:/sys/kernel/debug                  \
      -v /tmp:/tmp                                            \
      -v $MLPERF_DIR:/root/MLPERF                             \
      -v $DATASETS_DIR:/root/datasets                         \
      -v $MODEL_DIR:/root/model                               \
      --cap-add=sys_nice --cap-add=SYS_PTRACE                 \
      --user root --workdir=/root --net=host                  \
      --ulimit memlock=-1:-1 ${DOCKER_IMAGE}
    ```

2. Start the docker.

    ```bash
    docker exec $CONTAINER_NAME bash -c "service ssh start"
    docker exec -it $CONTAINER_NAME bash
    ```

### Download Data and Model

MLCommons hosts the model for download exclusively by MLCommons Members. You must first agree to the [confidentiality notice](https://docs.google.com/forms/d/e/1FAIpQLSc_8VIvRmXM3I8KQaYnKf7gy27Z63BBoI_I1u02f4lw6rBp3g/viewform), then follow the [link[(https://drive.google.com/drive/folders/11tBZvvrh0FCm3XuR5E849K42TqftYdUF)] to a directory containing [Rclone download instructions](https://docs.google.com/document/d/1Yp2T_TsVfg8uEoEv0wa-dGP4R7r1EOHucTvDNWznWzE/edit#heading=h.at8a3matgbrk). Follow steps 1-3 to install and activate Rclone. Finally, download the model to the desired download directory (default ./models):
Log into mlperf4.0 container and run:
```bash
rclone copy mlc-llama2:Llama2-70b-fused-qkv-mlperf /root/model/Llama2-70b-fused-qkv-mlperf -P
```
Similarly download the data to the desired download directory (default ./dataset):
```bash
rclone copy mlc-llama2:training/scrolls_gov_report_8k /root/datasets/scrolls_gov_report_8k -P
```


## Finetuning Llama2 70B with LoRA

1. Inside the mlperf4.0 container, install requirements:
```bash
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.15.0
pip install git+https://github.com/HabanaAI/optimum-habana-fork.git@cef6209
pip install -r  /root/MLPERF/Intel-HabanaLabs/benchmarks/llm_finetune/requirements.txt
huggingface-cli login
```
2. Create device warmup data:
```bash
cd /root/datasets/scrolls_gov_report_8k
python /root/MLPERF/Intel-HabanaLabs/benchmarks/llm_finetune/scripts/create_warmup_data.py
```

3. Run the training.
```bash
cd /root/MLPERF/Intel-HabanaLabs/benchmarks/llm_finetune/
cp /root/MLPERF/Intel-HabanaLabs/benchmarks/llm_finetune/config.json /root/model/Llama2-70b-fused-qkv-mlperf/
./run_llama_70B_fp8_submission.sh
```

# Llama 70B for Intel® Gaudi® MLPerf™ Inference Submission
This directory provides instructions to reproduce Intel Gaudi's results for MLPerf™ Inference submission.\
MLPerf™ is a trademark and service mark of MLCommons Association in the United States and other countries.\
All rights reserved. Unauthorized use is strictly prohibited.

## Setup

Please follow the instructions provided in the [Intel® Gaudi® Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the environment.

### CPU Performance Mode

Set CPU to performance mode:
```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

In order to verify the CPU mode, run:
```bash
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Huge Pages
It is recommended to set the number of huge pages as provided below:
```bash
#set current hugepages
sudo sysctl -w vm.nr_hugepages=100000
#Remove old entry if exists in sysctl.conf
sudo sed --in-place '/nr_hugepages/d' /etc/sysctl.conf
#Insert huge pages settings to persist
echo "vm.nr_hugepages=100000" | sudo tee -a /etc/sysctl.conf
```

### Clone Intel Gaudi Model-References
Clone this repository and switch to the branch that matches your Intel Gaudi software version.
You can run the [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the Intel Gaudi software version.
```bash
git clone --recurse-submodules -b [Intel Gaudi software version] https://github.com/HabanaAI/Model-References
```

### Prepare `Intel-HabanaLabs` MLPerf Inference Container
```bash
export INTEL_HABANALABS_DIR=$PWD/Model-References/MLPERF4.0/Inference
```

```bash
docker run --privileged --security-opt seccomp=unconfined   \
           --name mlperf-intel-habanalabs -td               \
           -v /dev:/dev                                     \
           --device=/dev:/dev                               \
           -v /sys/kernel/debug:/sys/kernel/debug           \
           -v /tmp:/tmp                                     \
           -v $INTEL_HABANALABS_DIR:/root/Intel-HabanaLabs/ \
           --cap-add=sys_nice --cap-add=SYS_PTRACE          \
           --user root --workdir=/root --net=host           \
           --ulimit memlock=-1:-1 vault.habana.ai/gaudi-docker-mlperf/ver4.0/pytorch-installer-2.1.1:1.14.98-33
```
```bash
docker exec -it mlperf-intel-habanalabs bash
```
### Get the Model
Choose one of the two available paths for downloading the model.

#### MLCommons Members Download
MLCommons hosts the model and preprocessed dataset for download exclusively by MLCommons Members.
You must first agree to the [confidentiality notice](https://docs.google.com/forms/d/e/1FAIpQLSc_8VIvRmXM3I8KQaYnKf7gy27Z63BBoI_I1u02f4lw6rBp3g/viewform),
then follow the link to a directory containing Rclone download instructions.

#### External Download
Using an e-mail registered for HuggingFace (if you do not have an account, you will need to create one),
go to [llama2-request-link](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and make a request.
Having it accepted, go to [HuggingFace Llama2 page](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) and ask for an access to the model.
Please note your HuggingFace authentication credentials as you may be required to provide them when cloning the below.
The download requires Git Large Files Storage:
```bash
mkdir -p /mnt/weka/data/pytorch/llama2/
pushd /mnt/weka/data/pytorch/llama2/
apt-get update
apt-get install git-lfs
git clone https://huggingface.co/meta-llama/Llama-2-70b-chat-hf Llama-2-70b-chat-hf
popd
```

### Get the Dataset
```bash
pushd /root/Intel-HabanaLabs/llama
export EXPORT_DIR=${PWD}/open_orca
mkdir -p /mnt/weka/data/mlperf_inference/llama2/
export DATASET_PATH=/mnt/weka/data/mlperf_inference/llama2/processed-data.pkl

curl https://rclone.org/install.sh | bash
rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b \
secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b \
endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
rclone copy mlc-inference:mlcommons-inference-wg-public/open_orca ./open_orca -P

pushd open_orca
gzip -d open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz
popd

mv ${EXPORT_DIR}/open_orca_gpt4_tokenized_llama.sampled_24576.pkl ${DATASET_PATH}
md5sum ${DATASET_PATH}
popd
```
The md5sum of generated dataset file should be 5fe8be0a7ce5c3c9a028674fd24b00d5.

##  Reproduce Results
### 99 and 99.9 Accuracy
The same script was submitted for both 99 and 99.9 benchmarks - no additional improvements were made for low accuracy (99), and 99.9 results were used for 99 as well.

### Get Started
Source the necessary files:

```bash
cd /root/Intel-HabanaLabs
source functions.sh
```

### Generate Results
**To generate full submission results, run the following command:**
```bash
build_mlperf_inference --output-dir <path_to_output_dir> --submission llama-99.9-fp8
```
The command produces results from accuracy and performance runs for both Offline and Server scenarios.
Logs can be found under /output_dir/logs/model/, e.g. /results/logs/llama-99.9-fp8/


**To generate results for Offline and Server scenarios separately, run the following commands:**
```bash
build_mlperf_inference --output-dir <path_to_output_dir> --submission llama-99.9-fp8_Offline
```

```bash
build_mlperf_inference --output-dir <path_to_output_dir> --submission llama-99.9-fp8_Server
```
Logs can be found under /output_dir/logs/model/scenario/, e.g. /results/logs/llama-99.9-fp8/Offline/

**To generate results for accuracy and performance separately, add ```--mode``` flag as in one of the following commands:**
```bash
build_mlperf_inference --output-dir <path_to_output_dir> --submission llama-99.9-fp8_Server --mode acc
```
```bash
build_mlperf_inference --output-dir <path_to_output_dir> --submission llama-99.9-fp8_Offline --mode perf
```

Logs can be found under /output_dir/logs/model/scenario/mode/, e.g. /results/logs/llama-99.9-fp8/Offline/accuracy/

## Performance Optimization with FP8 Flow
All linear operators' input activations and weights (linear and matmul operators) are quantized to FP8-143.
The weights are pre-quantized to FP8-143.

### Calibration
The submission contains measurement files required for FP8 quantization.

The following procedure was used to generate them:
```bash
cd /root/Intel-HabanaLabs/llama
export QUANT_CONFIG=hqt/llama2-70b-8x/config_meas_maxabs.json
deepspeed --num_gpus 8 llama_greedy.py --model_name_or_path /mnt/weka/data/pytorch/llama2/Llama-2-70b-chat-hf/ \
  --bf16 --attn_softmax_bf16 --use_hpu_graphs --use_kv_cache --batch_size 128 --reuse_cache                    \
  --trim_logits --limit_hpu_graphs --dataset $EXPORT_DIR/open_orca_gpt4_tokenized_llama.calibration_1000.pkl
```

The Quantization Toolkit is described in the [Intel Gaudi documentation](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html#fp8-inference-using-hqt).

## Supported Configurations

| Validated on | Intel Gaudi Software Version | Framework Version(s) |   Mode   |
| :----------: | :--------------------------: | :------------------: | :------: |
|    Gaudi 2   |      1.17.1                  |    PyTorch 2.3.1     | Inference |

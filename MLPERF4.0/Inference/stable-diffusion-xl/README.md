# Stable Diffusion XL for Intel® Gaudi® MLPerf™ Inference Submission
This directory provides instructions to reproduce Intel Gaudi's results for MLPerf™ inference submission.\
MLPerf™ is a trademark and service mark of MLCommons Association in the United States and other countries.\
All rights reserved. Unauthorized use is strictly prohibited.

## Setup

Please follow the instructions provided in the [Intel Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the environment.

### Prepare `Intel-HabanaLabs` MLPerf Inference Container

```bash
mkdir -p /path/to/Intel-HabanaLabs
export INTEL_HABANALABS_DIR=/path/to/Intel-HabanaLabs
```

This README is located in [code](./) directory corresponding to Intel-HabanaLabs submission. Download the whole [code](./) folder along with all subfolders under $INTEL_HABANALABS_DIR:

```bash
cd /path/to/Intel-HabanaLabs
git clone https://github.com/HabanaAI/Model-References.git
```

NOTE: The below is only an example value. Replace [Intel_Gaudi_software_version] and [PT_version] to match your setup and Supported Configuration.
```bash
export CONTAINER_NAME=mlperf4.0
export DOCKER_IMAGE=vault.habana.ai/gaudi-docker/[Intel_Gaudi_software_version]/ubuntu22.04/habanalabs/pytorch-installer-[PT_Version]:latest
docker run --privileged --security-opt seccomp=unconfined \
           --name $CONTAINER_NAME -td              \
           -v /dev:/dev                            \
           --device=/dev:/dev                      \
           -v /sys/kernel/debug:/sys/kernel/debug  \
           -v /tmp:/tmp                            \
           -v $INTEL_HABANALABS_DIR:/root/Intel-HabanaLabs/  \
           --cap-add=sys_nice --cap-add=SYS_PTRACE \
           --user root --workdir=/root --net=host  \
           --ulimit memlock=-1:-1 ${DOCKER_IMAGE}
```
```bash
docker exec $CONTAINER_NAME bash -c "service ssh start"
docker exec -it $CONTAINER_NAME bash
```
### Download Checkpoint
```bash
mkdir -p /mnt/weka/data/mlperf_inference/stable-diffusion-xl/stable_diffusion_fp32
pushd /mnt/weka/data/mlperf_inference/stable-diffusion-xl
curl https://rclone.org/install.sh | bash
rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
rclone copy mlc-inference:mlcommons-inference-wg-public/stable_diffusion_fp32 ./stable_diffusion_fp32 -P
popd
```
Alternatively, the required checkpoints/datasets can be downloaded offline and copied to the
required path before running Docker run command.

### Download Statistics File for Calculating FID
To download statistics file:
```bash
pushd /root/Intel-HabanaLabs/Model-References/MLPERF4.0/Inference/stable-diffusion-xl/tools
wget -L https://github.com/mlcommons/inference/raw/master/text_to_image/tools/val2014.npz
popd
```

### Download Dataset (Optional)
To download dataset, run the below command:
```bash
pushd /root/Intel-HabanaLabs/Model-References/MLPERF4.0/Inference/stable-diffusion-xl/tools
./download-coco-2014.sh -n 1
popd
```
build_mlperf_inference covers the same functionality.

##  Reproduce Results
### Get Started
Installation of the requirements and loadgen are part of the build_mlperf_inference script.

```bash
cd /root/Intel-HabanaLabs/Model-References/MLPERF4.0/Inference/
source functions.sh
```

### Generate Results
**To generate full submission results, run the following command:**
```bash
build_mlperf_inference --output-dir <path_to_output_dir> --submission sd-xl-fp8
```
The command produces results from accuracy and performance runs for both Offline and Server scenarios.
Logs can be found under /path_to_output_dir/logs/model/, e.g. /results/logs/sd-xl-fp8/

For future runs you can skip installing requirements and laodgen by passing --skip-reqs in build_mlperf_inference


**To generate results for Offline and Server scenarios separately, run the following commands:**
```bash
source functions.sh
build_mlperf_inference --output-dir <path_to_output_dir> --submission sd-xl-fp8_Offline
```

```bash
source functions.sh
build_mlperf_inference --output-dir <path_to_output_dir> --submission sd-xl-fp8_Server
```
Logs can be found under /path_to_output_dir/logs/model/scenario/, e.g. /results/logs/sd-xl-fp8/Offline/

**To generate results for accuracy and performance separately, add ```--mode``` flag as in one of the following commands:**
```bash
source functions.sh
build_mlperf_inference --output-dir <path_to_output_dir> --submission sd-xl-fp8_Offline --mode acc
```
```bash
source functions.sh
build_mlperf_inference --output-dir <path_to_output_dir> --submission sd-xl-fp8_Offline --mode perf
```

Logs can be found under /path_to_output_dir/logs/model/scenario/mode/, e.g. /results/logs/sd-xl-fp8/Offline/accuracy/

### Calibration Steps (Optional)
The below command recreates the measurements on the calibration dataset which we later use to determine the scales:
```bash
pushd /root/Intel-HabanaLabs/code/stable-diffusion-xl/stable-diffusion-xl
bash ./tools/measure.sh
popd
```

## Performance Optimization with FP8 Flow
To optimize performance, we set heavy-performance ops to operate in FP8-143.
In the conversion to FP8-143 we use various values of exponent bias which are determined using a calibration dataset.
For each input processed, the UNET block is iteratively invoked 20 times. A more aggressive form is used for the first 18 steps; and a less aggressive one for the final 2 steps.

### Environment Variables
All necessary environmental variables are enabled by default.

## Supported Configurations

| Validated on | Intel Gaudi Software Version | Framework Version(s) |   Mode   |
| :----------: | :--------------------------: | :------------------: | :------: |
|    Gaudi 2   |      1.18.0                   |    PyTorch 2.4.0     | Inference |

## Changelog

### Script Modifications
### 1.18.0
* Added INC (Intel Neural Compressor) Support, HQT replaced with INC

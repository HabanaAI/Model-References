###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

export SDXL_DIR=$PWD

# download calbiration captions
cd $SDXL_DIR/tools

bash download-coco-2014-calibration.sh -n 1 --calibration-dir $SDXL_DIR/coco2014/calibration

bash download-coco-2014.sh -n 1

# copy latents to calibration dir and move captions, because main script expects this structure
cd $SDXL_DIR
cp -r $SDXL_DIR/coco2014/latents $SDXL_DIR/coco2014/calibration
mkdir -p $SDXL_DIR/coco2014/calibration/captions
mv $SDXL_DIR/coco2014/calibration/captions.tsv $SDXL_DIR/coco2014/calibration/captions/captions.tsv
mkdir -p $SDXL_DIR/tools/quantize/measure_all
# generate scales
export QUANT_CONFIG=$SDXL_DIR/tools/quantize/measure_config.json

python3 main.py \
--dataset "coco-1024" --dataset-path coco2014/calibration \
--profile stable-diffusion-xl-pytorch --model-path /mnt/weka/data/mlperf_inference/stable-diffusion-xl/stable_diffusion_fp32 \
--max-batchsize 1 --dtype bf16 --scenario Server --hpus 1 --device cuda --measure --count 50 --user_conf configs/user.conf

# format weights
input_file=$SDXL_DIR/tools/quantize/measure_all/fp8_hooks_maxabs_-1_1.npz
prefix="${input_file%_-1_1.npz}"
for i in {0..7}; do
  output_file="${prefix}_${i}_8.npz"
  cp "$input_file" "$output_file"
  echo "Created: $output_file"
done

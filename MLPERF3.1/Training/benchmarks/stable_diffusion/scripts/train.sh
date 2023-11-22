HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
DIFFUSERS_OFFLINE=1

python  main.py -m train \
    --logdir /tmp/results \
    --autocast \
    -b ./configs/train_512.yaml

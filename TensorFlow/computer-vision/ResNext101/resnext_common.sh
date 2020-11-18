TRAINING_COMMAND="python imagenet-resnet.py
  --mode resnext32x4d
  --data /datasets/imagenet/
  -d 101
  --batch 32"

$TRAINING_COMMAND

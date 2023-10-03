#!/bin/bash

deepspeed --num_nodes=1 --num_gpus=8 cifar10_deepspeed.py --deepspeed --deepspeed_config ds_config.json --use_hpu $@
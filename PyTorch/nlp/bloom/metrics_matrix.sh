for num_dev in 1 8;
    do
    for DT in fp32 fp16 bf16;
    do
        for model_size in 560m 1b7 3b 7b1;
            do
            deepspeed --num_gpus ${num_dev} bloom_metrics.py --weights /datasets/model_weights/bloom-${model_size} --model bloom-${model_size} --device=cuda --use_graphs=False --dtype=${DT}
            done
    done
done

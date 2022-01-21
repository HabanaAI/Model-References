## Habana PyTorch Hello World Usage

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).
This mnist based hello world model is forked from github repo [pytorch/examples](https://github.com/pytorch/examples/tree/master/mnist).

## This hello world example can run on single Gaudi and 8 Gaudi in eager mode and lazy mode with FP32 data type and BF16 mixed data type.

### Single Gaudi run commands
Single Gaudi FP32 eager mode run command:
```bash
python3 demo_mnist.py --hpu
```

Single Gaudi BF16 eager mode run command:
```bash
python3 demo_mnist.py --hpu --hmp
```

Single Gaudi FP32 lazy mode run command:
```bash
python3 demo_mnist.py --hpu --use_lazy_mode
```

Single Gaudi BF16 lazy mode run command:
```bash
python3 demo_mnist.py --hpu --hmp --use_lazy_mode
```

### Multinode run commands

8 Gaudi FP32 eager mode run command:
```bash
python3 demo_mnist.py --hpu --data_type fp32 --world_size 8
```

8 Gaudi BF16 eager mode run command:
```bash
python3 demo_mnist.py --hpu --data_type bf16 --world_size 8
```

8 Gaudi FP32 lazy mode run command:
```bash
python3 demo_mnist.py --hpu --data_type fp32 --use_lazy_mode --world_size 8
```

8 Gaudi BF16 lazy mode run command:
```bash
python3 demo_mnist.py --hpu --data_type bf16 --use_lazy_mode --world_size 8
```

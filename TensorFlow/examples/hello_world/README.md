# A simple Hello World example for new Gaudi users

This simple example provides basic intructions to new Gaudi users on how to port their own models to Gaudi and launch runs on single Gaudi card, 8 Gaudi cards, and 16 Gaudi cards. For 16 Gaudi cards, users can choose using Gaudi NIC and host NIC.

## Table Of Contents
* [Model and system setup](#model-system-setup)
* [Hello world usage](#hello-world-usage)
* [Example output of single card training](#example-output-of-single-card-training)

## Model and system setup

Users can refer to this website https://github.com/HabanaAI/Setup_and_Install to set up their Gaudi system and running environment and lauch the docker image.

Clone the repository and go to resnet directory:
```
git clone https://github.com/HabanaAI/Model-References.git
cd Model-References/TensorFlow/examples/hello_world
```

Note: If the repository is not in the PYTHONPATH, make sure you update it.
```
export PYTHONPATH=</path/to/Model-References>/Model-References:$PYTHONPATH
```

## Hello World Usage
### This simple hello world example can run on single Gaudi, 8 Gaudi, and 16 Gaudi.

Multicard runs require openmpi 4.0.2 installed on the system.

To run on single Gaudi, the run command is `./run_single_gaudi.sh`

To run on 8 Gaudi, the run command is `./run_hvd_8gaudi.sh`

To run on 16 Gaudi cards, one can choose using Gaudi nic or host nic.

To run on 16 Gaudi using Gaudi nic, the run command is `./run_hvd_16gaudi.sh 192.168.0.1 192.168.0.2`

To run on 16 Gaudi cards using host nic, the run command is `./run_hvd_16gaudi_hostnic.sh 192.168.0.1 192.168.0.2`

## Example output of training on a single Gaudi card

Instructions for updating:
If using Keras pass *_constraint arguments to layers.
Train on 60000 samples
Epoch 1/5
60000/60000 [==============================] - 2s 31us/sample - loss: 1.2641 - accuracy: 0.7205
Epoch 2/5
60000/60000 [==============================] - 2s 26us/sample - loss: 0.7140 - accuracy: 0.8434
Epoch 3/5
60000/60000 [==============================] - 2s 25us/sample - loss: 0.5862 - accuracy: 0.8615
Epoch 4/5
60000/60000 [==============================] - 2s 26us/sample - loss: 0.5247 - accuracy: 0.8707
Epoch 5/5
60000/60000 [==============================] - 2s 26us/sample - loss: 0.4872 - accuracy: 0.8765

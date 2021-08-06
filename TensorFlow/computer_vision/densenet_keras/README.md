# Densenet-121
## Table Of Contents
* [Model overview](#Model-overview)
* [Setup](#Setup)
* [Training the Model](#Training-the-model)

## Model overview
This code demonstrates training of the Densenet-121 model on ILSVRC2012 dataset (aka ImageNet).
The model architecture is described in this paper: https://arxiv.org/abs/1608.06993.
The implementation employs Keras.Application model (https://www.tensorflow.org/api_docs/python/tf/keras/applications/densenet).

Please visit [this page](../../../README.md#tensorflow-model-performance) for performance information.

### Changes to the original model:
1. A generic Keras-based training script was used from here: https://github.com/jkjung-avt/keras_imagenet.
As this script was originally developed in TF-1.12, it was migrated to TF-2.2 using the automatic migration tool
provided by TensorFlow (https://www.tensorflow.org/guide/upgrade).
2. Updates on the usage of tf.contrib module were made, as these are not available in TensorFlow 2.2.
An additional change is the addition of multi card support, using
[Mirrored strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy). This change only applies to non-HPU hardware.
3. `set_learning_phase(True)` - Sets the learning phase to a fixed value.
4. Setting batch size before calling `model.fit`

## Setup

Please follow the instructions given in the following link for setting up the environment: [Gaudi Setup and Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please answer the questions in the guide according to your preferences. This guide will walk you through the process of setting up your system to run the model on Gaudi.

### Data

The training script requires Imagenet data (ILSVRC2012) to be preprocessed into TFRecords.

**Note: Preprocessing requires use of TensorFlow version 1.12.2**

1.	Create a folder named `dataset_preparation`
	The `dataset_preparation` folder structure is described below:

	* `dataset_preparation`

        •	`keras_imagenet`  (a github repository for converting the files in train and validation folders  to TF records described in step #5)

        •	`tfrecords`  (The folder in which the training and validation TF record files will reside. This folder is the –dataset_dir folder used by the densnet training script)

        •	`train`

        •	`validation`

        •	`ILSVRC2012_img_train.tar`   (imagenet train file obtained by step #2)

        •	`ILSVRC2012_img_val.tar`	   (imagenet validation file obtained by step #2)

2.	Download imagenet train (`ILSVRC2012_img_train.tar`) and validation (`ILSVRC2012_img_val.tar`) files from http://image-net.org/download-images  to `dataset_preparation` directory.
3.	Imagent training files  creation of 1000 folder classes

    a.	`mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train`

    b.	`tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar`

    c.	`find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done`

    d.	`cd ..`

4.	Imagenet validation files creation in validation folder

    a.	`mkdir validation && mv ILSVRC2012_img_val.tar validation/ && cd validation`

    b.	`tar xvf ILSVRC2012_img_val.tar`

5.	Converting the validation and training images to TF records

    a.  In `dataset_preparation` folder `git clone  https://github.com/jkjung-avt/keras_imagenet.git`

    b.	`cd keras_imagenet/data`

    c.	`python3 preprocess_imagenet_validation_data.py ../../validation  imagenet_2012_validation_synset_labels.txt`

    d. `mkdir ../../tfrecords`

    d.	`python3 build_imagenet_data.py --output_directory ../../tfrecords --train_directory ../../train --validation_directory ../../validation`


After step #5 the `tfrecords` directory will contain 1024 training and 128 validation TFrecord files. The `tfrecords` directory is the value of the `--dataset_dir` command line argument for the densenet training script train.py.



### Model Setup
Clone the repository and go to densenet_keras directory:
```
$ git clone https://github.com/HabanaAI/Model-References.git /root/Model-References
$ cd Model-References/TensorFlow/computer_vision/densenet_keras
```

Note: If the repository is not in the PYTHONPATH, make sure you update it.

```
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

## Training the Model

1. `python3 train.py --dataset_dir <path TFRecords dataset>`
2. `python3 train.py --dataset_dir <path TFRecords dataset> --only_eval <path to saved model>`

Step 1 will save the trained model after each epoch under `saves`

The following params are default:
1. `--dropout_rate`: 0.0 (no dropout)
2. `--weight_decay`: 1e-4
3. `--optimizer`: sgd
4. `--batch_size`: 64
5. `--lr_sched`: steps
6. `--initial_lr`: 5e-2
7. `--final_lr`: 1e-5
8. `--bfloat16`: True
9. `--epochs`: 90

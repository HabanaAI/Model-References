# Hello World Example for TensorFlow

This directory provides example training scripts to run Hello World on 1 HPU and 8 HPUs of 1 server, 16 HPUs of 2 servers, and multiple Tenants scenario which has 2-4 HPU workloads running in parallel.

For further information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../README.md)
* [Example Overview](#example-overview)
* [Setup](#setup)
* [Training Examples](#training-examples)

## Example Overview

The TensorFlow Hello World example is based on the MNIST example from TensorFlow tutorial: [Training a Neural Network on MNIST with Keras](https://www.tensorflow.org/datasets/keras_example).

## Setup

Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to set up the environment including the `$PYTHON` environment variable. The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References

In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. You can run the [`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References /root/Model-References
```

**Note:** If Model-References repository path is not in the PYTHONPATH, make sure you update it:
```bash
export PYTHONPATH=$PYTHONPATH:/root/Model-References
```

## Training Examples

The sections below provide examples in Bash scripts and Python scripts.

### Single Card and Multi-Card Training Examples

#### Examples in Bash Scripts

**Run training on 1 HPU:**

```bash
bash run_single_gaudi.sh
```

**Run training on 8 HPUs:**

- 8 HPUs on Horovod:
    ```bash
    bash run_hvd_8gaudi.sh
    ```
- Or, 8 HPUs on Horovod with `tf.function`:

    ```bash
    bash run_hvd_8gaudi_tf_func.sh
    ```

#### Examples in Python Scripts

The `example.py` presents a basic TensorFlow code example. For more details on the additional migration Python examples, refer to [TensorFlow Migration Guide](https://docs.habana.ai/en/latest/TensorFlow/Migration_Guide/Porting_Simple_TensorFlow_Model_to_Gaudi.html#creating-a-tensorflow-example).

**Run training on 1 HPU:**

- `example.py` file is an example showing with only Keras APIs:
    ```python
    $PYTHON example.py
    ```

- `example_tf_func.py` file is an example with `tf.function`. To run `example_tf_func.py` script, run:
    ```python
    $PYTHON example_tf_func.py
    ```

- `example_tf_session.py` file is an example with TF1 which uses `tf.Session`. To run `example_tf_session.py` script, run:
    ```python
    $PYTHON example_tf_session.py
    ```

**Note**:
* `input_data.py` file will download and process the dataset.
* `example_tf_session.py` script requires downloading the dataset from [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/). If you experience issues, you can download the files manually and save them to **MNIST_data** directory.

**Run training on 8 HPUs:**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

`example_tf_func_hvd.py` file uses `tf.function` and runs on multiple cards. To run `example_tf_func_hvd.py` on 8 cards, run:
    ```bash
    mpirun --allow-run-as-root -np 8 python3 example_tf_func_hvd.py
    ```

### Multi-Server Training Examples

**Run training on 16 HPUs**

When training on 16 HPUs, you can choose using either Gaudi NICs or host NICs.

- To run training on 16 HPUs using Gaudi NICs, run the following command:

    ```bash
    ./run_hvd_16gaudi.sh <ip_address_1> <ip_address_2>
    ```

- To run training on 16 HPUs using host NICs, run the following command:

    ```bash
    ./run_hvd_16gaudi_hostnic.sh <ip_address_1> <ip_address_2>
    ```

&nbsp;

### Partial Run with Multiple Gaudi Cards

**NOTE:** Multi-card trainings require Open MPI installed on the system.

Running part of the Gaudi cards installed on the system is allowed. It also makes it possible to run
multiple workloads in parallel on the same machine. For further information, refer to [Multiple Tenants on HPU](https://docs.habana.ai/en/latest/Orchestration/Multiple_Tenants_on_HPU/index.html).

When implementing partial runs on multiple cards, make sure to set the following:

- The number of Gaudi cards involved must be power of 2, such as 2, 4, or 8.
- The environment variable ``HABANA_VISIBLE_MODULES`` must be set to properly configure the used Gaudi cards.

**Run training on 4 HPUs:**

- To run 2 workloads with 4 HPUs each on a single server simultaneously, run the following command:

    ```bash
    bash run_multi_hvd_4_4.sh
    ```

The environment variable ``HABANA_VISIBLE_MODULES`` should be set to "0,1,2,3" and "4,5,6,7" to the 2 workloads respectively.

- To run 3 workloads with 4 Gaudi cards on 1 workload and 2 Gaudi cards on the other two, you can set ``HABANA_VISIBLE_MODULES`` to "0,1", "2,3", "4,5,6,7".

**NOTE:** For workloads running with 2 Gaudi cards, it is recommended to set "0,1", "2,3", "4,5", "6,7" to the environment variable ``HABANA_VISIBLE_MODULES``.
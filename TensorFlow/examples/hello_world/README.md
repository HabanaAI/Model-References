## Hello World Usage

For further information about training deep learning models on Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Setup
Follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the examples on Gaudi.

### Examples

#### Examples in Bash Scripts

This simple hello world example can run on single Gaudi, 8 Gaudis and 16 Gaudis.
Multicard runs require Open MPI installed on the system.

To run on single Gaudi, run the following command:

```bash
bash run_single_gaudi.sh
```

To run on 8 Gaudis, run the following command:

```bash
bash run_hvd_8gaudi.sh
```
or

```bash
bash run_hvd_8gaudi_tf_func.sh
```

To run on 16 Gaudi cards, one can choose using Gaudi nic or host nic.

To run on 16 Gaudi cards using Gaudi nic, run the following command: 

```bash
./run_hvd_16gaudi.sh <ip_address_1> <ip_address_2>
```

To run on 16 Gaudi cards using host nic, run the following command:

```bash
./run_hvd_16gaudi_hostnic.sh <ip_address_1> <ip_address_2>
```

To run 2 workloads with 4 Gaudi cards each on single server simultaneously,run the following command:

```bash
bash run_multi_hvd_4_4.sh
```
&nbsp;

#### Examples in Python Scripts

The following scripts are examples in Python programming language.
If the Model-References repository is not in the PYTHONPATH, make sure to update it.

```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

##### example.py, example\_tf\_func.py, example\_tf\_session.py and example\_tf\_func\_hvd.py

The file `example.py` is an example showing with only keras APIs.
```python
$PYTHON example.py
```

The file `example_tf_func.py` is an example with `tf.function`. The command to run `example_tf_func.py` is
```python
$PYTHON example_tf_func.py
```

The file `example_tf_session.py` is an example with TF1 which uses `tf.Session`. The command to run `example_tf_session.py` is
```python
$PYTHON example_tf_session.py
```
The file input_data.py will download and process the dataset.

**Note**: The script `example_tf_session.py` needs to download dataset from website http://yann.lecun.com/exdb/mnist/. If there are issues downloading the data in the script, users can manually download the files and save them to **MNIST_data** directory.

The file `example_tf_func_hvd.py` is an example which uses `tf.function` and runs on multiple Gaudi. The command to run `example_tf_func_hvd.py` on 8 Gaudi is
```bash
mpirun --allow-run-as-root -np 8 python3 example_tf_func_hvd.py
```
&nbsp;

#### Examples of Partial Run with Multiple Gaudi Cards

Running part of the Gaudi cards installed on the system is allowed. It also makes it possible to run
multiple workloads in parallel on the same machine. However, it needs to follow some requirements.

- The number of Gaudi cards involved must be power of 2, such as 2, 4, or 8.

- The environment variable ``HABANA_VISIBLE_MODULES`` must be set to properly configure the used Gaudi cards.

The script `./run_multi_hvd_4_4.sh` demonstrates an example to run 2 workloads with 4 Gaudi cards each in parallel. 

The environment variable ``HABANA_VISIBLE_MODULES`` is set to "0,1,2,3" and "4,5,6,7" to the 2 workloads respectively.

For workloads running with 2 Gaudi cards, it is recommended to set "0,1", "2,3", "4,5", "6,7" to the environment variable "HABANA_VISIBLE_MODULES". For example, if you plan to run 3 workloads with 4 Gaudi cards on 1 workload and 2 Gaudi cards on the other two, you can set the environment variable "HABANA_VISIBLE_MODULES" to "0,1", "2,3", "4,5,6,7" respectively.

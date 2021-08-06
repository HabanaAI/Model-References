## Hello World Usage

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

### This simple hello world example can run on single Gaudi, 8 Gaudi, and 16 Gaudi.

Multicard runs require openmpi 4.0.2 installed on the system.

To run on single Gaudi, the run command is `./run_single_gaudi.sh`

To run on 8 Gaudi, the run command is `./run_hvd_8gaudi.sh` or `./run_hvd_8gaudi_tf_func.sh`.

To run on 16 Gaudi cards, one can choose using Gaudi nic or host nic.

To run on 16 Gaudi using Gaudi nic, the run command is `./run_hvd_16gaudi.sh 192.168.0.1 192.168.0.2`

To run on 16 Gaudi cards using host nic, the run command is `./run_hvd_16gaudi_hostnic.sh 192.168.0.1 192.168.0.2`

### example.py and example_hvd.py

The file `example.py` is an example showing with only keras APIs

The file `example_hvd.py` is an example which runs on multiple Gaudi.

### example\_tf\_func.py, example\_tf\_session.py and example\_tf\_func\_hvd.py

The file `example\_tf\_func.py` is an example with `tf.function`. The command to run `example\_tf\_func.py` is `python3 example\_tf\_func.py`.

The file `example\_tf\_session.py` is an example with TF1 which uses `tf.Session`. The command to run `example\_tf\_session.py` is `python3 example\_tf\_session.py`. The file input_data.py will download and process the dataset.
**Note**: The script `example\_tf\_session.py` needs to download dataset from website http://yann.lecun.com/exdb/mnist/. If there are issues downloading the data in the script, users can manually download the files and save them to **MNIST_data** directory.

The file `example\_tf\_func\_hvd.py` is an example which uses `tf.function` and runs on multiple Gaudi. The command to run `example\_tf\_func\_hvd.py` on 8 Gaudi is `mpirun --allow-run-as-root -np 8 python3 example\_tf\_func\_hvd.py`.

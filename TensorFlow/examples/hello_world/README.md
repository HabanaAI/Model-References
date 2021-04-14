## Hello World Usage

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

### This simple hello world example can run on single Gaudi, 8 Gaudi, and 16 Gaudi.

Multicard runs require openmpi 4.0.2 installed on the system.

To run on single Gaudi, the run command is `./run_single_gaudi.sh`

To run on 8 Gaudi, the run command is `./run_hvd_8gaudi.sh`

To run on 16 Gaudi cards, one can choose using Gaudi nic or host nic.

To run on 16 Gaudi using Gaudi nic, the run command is `./run_hvd_16gaudi.sh 192.168.0.1 192.168.0.2`

To run on 16 Gaudi cards using host nic, the run command is `./run_hvd_16gaudi_hostnic.sh 192.168.0.1 192.168.0.2`


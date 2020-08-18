
# Packaging scripts for gathering information about the model and Habana training session for Habana analysis and debug

The purpose of these packaging scripts is to provide an automated way for the Habana Model-References user to gather information relevant for Habana engineers to:

- Replicate any errors, crashes or other problems they are seeing with their models
- Further analyze, debug and resolve these problems

There are two packaging scripts provided, and they can be run in any order:

- **gather_info_docker.py**

  This is run in the Habana docker container and gathers data about the training command-line, the environment variables that are set, python packages, contents of $HOME of the container which contains Python code to run training, model related artifacts such as datasets, output directories from training, etc.

- **gather_info_host.py**

  This is run outside of the Habana docker container, in an xterm on the same host machine. It takes the docker container ID as input and gathers information about the container such as the Habana TensorFlow docker image it is running, "docker stats" for memory usage, "docker inspect" on details on lower level resources controlled by docker, etc.

## Run gather_info_docker.py in the Habana docker container running training

The gather_info_docker.py script options are best described by its help message:

```
$ cd Model-References/TensorFlow/utils/packaging_for_habana_debug
$ python3 gather_info_docker.py --help

usage: gather_info_docker.py [-h] [-o OUTDIR] [-c] [--lite] -s STDOUT
                             [-e STDERR] [-y YAML_CONFIG] -cmd CMD_NAME
                             [--copydirs COPYDIRS [COPYDIRS ...]]

optional arguments:
  -h, --help            show this help message and exit
  -o OUTDIR, --outdir OUTDIR
                        The output directory where all the files and other
                        information will be stored. The output will be stored
                        as an archive as well as the actual directory where
                        all the contents are copied. Key assumptions: 1) The
                        output directory specified here should be mapped to
                        the host via the docker run command-line. It should
                        have sufficient disk space to include the contents of
                        the docker container's $HOME - this would include
                        datasets, TensorFlow events and checkpoints, etc.
                        E.g.: sudo docker run -it -v <host outdir
                        path>:<docker container outdir path> 2) This script
                        assumes that the user invokes this script from a Unix
                        shell (and not from a Python interpreter shell)
                        running within a Habana TensorFlow/PyTorch Docker
                        container. 3) The script also assumes that the user
                        account used to run the docker container is either
                        root or a user account with sudo access.
  -c, --clear           Delete the existing contents of the output directory
                        before copying over files from this run.
  --lite                Run a lite version of the packaging script. This will
                        not collect the files and directories under $HOME,
                        i.e., it will not gather model artifacts and training
                        output directories. Please use the --copydirs option
                        to specify any artifacts in the docker's file-system
                        that you want to be collected by this script, as a
                        space-separated list of file/directory names following
                        this option.
  -s STDOUT, --stdout STDOUT
                        This is the fully path-qualified filename for the
                        stdout for the training run.
  -e STDERR, --stderr STDERR
                        This is the fully path-qualified filename for the
                        stderr for the training run, if different from the
                        stdout.
  -y YAML_CONFIG, --yaml_config YAML_CONFIG
                        This is the fully path-qualified filename for the yaml
                        config file used for the training run.
  -cmd CMD_NAME, --cmd_name CMD_NAME
                        This is the command name that the script should search
                        for in the history of the current shell to get the
                        full training command-line invocation that the user
                        ran. Caveats: This script should be invoked from a
                        Habana TensorFlow/PyTorch Docker container bash shell
                        and not from within a Python interpreter. This
                        functionality depends on the availability of
                        $HOME/.bash_history in the docker container shell. If
                        this file does not exist, it can be created by running
                        'history -a' in the container.
  --copydirs COPYDIRS [COPYDIRS ...]
                        Space-separated list of names of directories and files
                        that are in the docker container, to be copied to the
                        output directory. For e.g. this could be
                        /software/data/bert_checkpoints. Unless you are using
                        the --lite option, this list would typically not
                        include files or directories that are under $HOME in
                        the Habana docker container.
```

### Example usage of gather_info_docker.py

```
$ git clone https://github.com/HabanaAI/Model-References
```

Suppose you are working with the TensorFlow NLP BERT model that is released in Model-References. Run the training as described in the README document for TensorFlow/nlp/bert:

```
$ cd Model-References/TensorFlow/nlp/bert/
$ python3 ../../habana_model_runner.py --model bert --hb_config bert_base_pretraining_overfit.yaml >& ~/hlogs/bert_logs/bert_base_pretraining_overfit.txt
```

Now, to gather all aspects of the training run, including the model training Python code, the datasets and training output directories, run the gather_info_docker.py packaging script as follows. Let's assume that the BERT overfit pretraining dataset has been mapped to "/software/data/bert_checkpoints" in the container.

A couple of important notes:

- Please make sure that the directory specified for the **--outdir** option is mapped from the host system to the docker container
- Please make sure that $HOME/.bash_history exists in the container shell, and if it doesn't, run "history -a" before invoking gather_info_docker.py so that it can correctly capture the training command-line invoked in the container shell

```
$ python3 ../../utils/packaging_for_habana_debug/gather_info_docker.py --outdir ~/hlogs/bert_issues/package_dir --stdout ~/hlogs/bert_logs/bert_base_pretraining_overfit.txt --yaml_config bert_base_pretraining_overfit.yaml -cmd habana_model_runner.py --copydirs /software/data/bert_checkpoints
```

This will generate the following directories under the outdir **$HOME/hlogs/bert_issues/package_dir**:

```
$ cd ~/hlogs/bert_issues/package_dir
$ ls -lt
-rwxr-xr-x 1 root root 11643911026 Feb  8 15:00 gather_info_docker.tar.gz
drwxr-xr-x 5 root root        4096 Feb  8 15:00 gather_info_docker
$ cd gather_info_docker/
$ ls -lt
-rwxr-xr-x 1 root root    960 Feb  8 15:00 machine_cpumode.txt
-rwxr-xr-x 1 root root 142343 Feb  8 15:00 machine_cpuinfo.txt
-rwxr-xr-x 1 root root  24178 Feb  8 15:00 machine_lspci.txt
-rwxr-xr-x 1 root root  16465 Feb  8 15:00 machine_hlsmi.txt
-rwxr-xr-x 1 root root    856 Feb  8 15:00 machine_ipaddr.txt
-rwxr-xr-x 1 root root     14 Feb  8 15:00 machine_hostname.txt
-rwxr-xr-x 1 root root     11 Feb  8 15:00 docker_container_disk_usage.txt
-rwxr-xr-x 1 root root   4876 Feb  8 15:00 docker_container_python_pkgs.txt
drwxr-xr-x 3 root root   4096 Feb  8 15:00 docker_container_additional_dirs
drwxr-xr-x 9 root root   4096 Feb  8 15:00 docker_container_homedir_and_model_artifacts
drwxr-xr-x 2 root root   4096 Feb  8 15:00 habana_logs
-rwxr-xr-x 1 root root   2169 Feb  8 15:00 env_vars.txt
-rwxr-xr-x 1 root root    119 Feb  8 15:00 cmdline_invocation.txt
-rwxr-xr-x 1 root root    276 Feb  8 15:00 execution-command-line-gather_info_docker.py.txt
-rwxr-xr-x 1 root root 206149 Feb  8 15:00 cmd_stdout.txt
-rwxr-xr-x 1 root root   3158 Feb  8 15:00 model_yaml_config.txt
-rwxr-xr-x 1 root root    271 Feb  8 15:00 docker_container_history.txt
```

This is a summary of the information that gather_info_docker.py has gathered from your docker training session:

- **machine_cpumode.txt**
  CPU scaling governor settings (powersave, performance).
- **machine_cpuinfo.txt**
  Output of /proc/cpuinfo
- **machine_lspci.txt**
  Output of lspci
- **machine_hlsmi.txt**
  Output of running "hl-smi -q" that generates detailed information about the Habana system, # Gaudi cards available, their firmware versions, etc.
- **machine_ipaddr.txt**
  The machine's IP address, as returned by "hostname -I"
- **machine_hostname.txt**
  The machine's hostname, as returned by "hostname"
- **docker_container_disk_usage.txt**
  The disk usage in $HOME of the container
- **docker_container_python_pkgs.txt**
  The Python packages installed in the container, along with their version numbers
- **docker_container_additional_dirs**
  The files and directories specified following the --copydirs option
- **docker_container_homedir_and_model_artifacts**
  Everything under $HOME, excluding the highest-level directory in the path specified to --outdir (and its contents)
- **habana_logs**
  The contents of the directory referred to by the $HABANA_LOGS environment variable
- **env_vars.txt**
  The environment variables set in the docker container shell
- **cmdline_invocation.txt**
  The latest invocation of the command name specified to the -cmd or --cmd_name option (e.g. habana_model_runner.py) in the container shell, including all the options it was run with. For this to work correctly, please make sure to run "history -a" in the container shell before running gather_info_docker.py.
- **execution-command-line-gather_info_docker.py.txt**
  gather_info_docker.py's invocation command-line, along with the command-line options used
- **cmd_stdout.txt**
  The content(s) of the file(s) specified to the --stdout and --stderr (if any) option(s) to gather_info_docker.py
- **model_yaml_config.txt**
  The content of the file specified to the --yaml_config option to gather_info_docker.py
- **docker_container_history.txt**
  The results of running "history -a" in the docker container shell


## Run gather_info_host.py in an xterm on the same host system, outside of the Habana docker container:

The gather_info_host.py script options are best described by its help message:

```
$ cd Model-References/TensorFlow/utils/packaging_for_habana_debug
$ python3 gather_info_docker.py --help

usage: gather_info_host.py [-h] [-o OUTDIR] [-c] -i CONTAINER_ID

optional arguments:
  -h, --help            show this help message and exit
  -o OUTDIR, --outdir OUTDIR
                        The output directory where all the files and other
                        information will be stored. The output will be stored
                        as an archive as well as the actual directory where
                        all the contents are copied.
  -c, --clear           Delete the existing contents of the output directory
                        before copying over files from this run.
  -i CONTAINER_ID, --container_id CONTAINER_ID
                        This is the Docker container ID for the running docker
                        session, obtained by "docker ps" and selecting the
                        CONTAINER ID for the session for which we want to run
                        the packaging script.
```

### Example usage of gather_info_host.py

On the host that is running the docker training session, open an xterm and get the container ID of the docker session in which you ran the training session for which data is being packaged for Habana debug. In other words, get the container ID of the docker session in which you ran the "gather_info_docker.py" above.
```
$ sudo docker ps
```

Note the container ID of this docker session, and run gather_info_host.py with that container ID. You can pass the same --outdir argument as you did for gather_info_docker.py. gather_info_host.py will create a separate sub-directory called "gather_info_host" under this outdir.

```
$ cd Model-References/TensorFlow/utils/packaging_for_habana_debug
$ sudo python3 gather_info_host.py --outdir ~/hlogs/bert_issues/package_dir --container <container ID>
```

This will generate the following directories under the outdir **$HOME/hlogs/bert_issues/package_dir**:

```
$ cd ~/hlogs/bert_issues/package_dir
$ ls -lt
-rwxr-xr-x 1 root root        4253 Feb  8 15:00 gather_info_host.tar.gz
drwxr-xr-x 2 root root        4096 Feb  8 15:00 gather_info_host
-rwxr-xr-x 1 root root 11643911026 Feb  8 15:00 gather_info_docker.tar.gz
drwxr-xr-x 5 root root        4096 Feb  8 15:00 gather_info_docker
$ cd gather_info_host
$ ls -lt
-rwxr-xr-x 1 root root   428 Feb  8 15:00 docker_ps_cmd.txt
-rwxr-xr-x 1 root root   385 Feb  8 15:00 docker_stats.txt
-rwxr-xr-x 1 root root 13334 Feb  8 15:00 docker_inspect.txt
-rwxr-xr-x 1 root root   146 Feb  8 15:00 execution-command-line-gather_info_host.py.txt
```

This is a summary of the information that gather_info_host.py has gathered about the docker container that ran BERT pretraining:

- **docker_ps_cmd.txt**
  The details of the container ID, including the IMAGE, so that Habana knows the Habana docker image you are running
- **docker_stats.txt**
  The output of running "docker stats" with the given container ID, that includes container resource usage statistics
- **docker_inspect.txt**
  The output of running "docker inspect" with the given container ID, that includes additional system-level details about the docker container
- **execution-command-line-gather_info_host.py.txt**
  gather_info_host.py's invocation command-line, along with the command-line options used

## Upload to Habana
Specific mechanisms for sharing the **gather_info_docker.tar.gz** and **gather_info_host.tar.gz** files with Habana are being finalized.

## Important note about security guidelines
The packaging scripts are only intended as a convenience for Habana users. Please make sure that the content that the packaging scripts have gathered in the output directories "**gather_info_docker**" and "**gather_info_host**" is collateral that is NDA-protected and meets any security and IP protection requirements of that NDA. If needed, you can delete specific content from these directories and manually re-generate the **gather_info_docker.tar.gz** / **gather_info_host.tar.gz** files before sharing them with Habana.

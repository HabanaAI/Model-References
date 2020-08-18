###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
###############################################################################

"""Packaging script for analysis/debug of models in Habana model_garden

This python script is called for collecting info related to the training session
from the Habana docker container in which the user has run training.
The docker container should be run as root or with sudo.
"""

import os
import sys
import argparse
import subprocess
import tarfile
from helper_functions import get_canonical_path, create_output_dir

ACTUAL_CMD = os.path.realpath(sys.argv[0])
INFO_TARBALL = "{}".format(os.path.splitext(os.path.basename(ACTUAL_CMD))[0])
EXECCMDLINEFILE = "execution-command-line-{}.txt".format(os.path.basename(ACTUAL_CMD))
TMPDIR = INFO_TARBALL
TMPFILENAME = "tmpfile"

class GatherInfoDockerArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(GatherInfoDockerArgParser, self).__init__()
        self.add_argument('-o', '--outdir', type=str, default = './', help="""
            The output directory where all the files and other information will be stored.
            The output will be stored as an archive as well as the actual directory where
            all the contents are copied.
            Key assumptions:
            1) The output directory specified here should be mapped to the host via the
               docker run command-line. It should have sufficient disk space to include
               the contents of the docker container's $HOME - this would include datasets,
               TensorFlow events and checkpoints, etc.
               E.g.: sudo docker run -it -v <host outdir path>:<docker container outdir path>
            2) This script assumes that the user invokes this script from a
               Unix shell (and not from a Python interpreter shell) running within a Habana
               TensorFlow/PyTorch Docker container.
            3) The script also assumes that the user account used to run the docker container
               is either root or a user account with sudo access.
            """
        )
        self.add_argument('-c', '--clear', action="store_true", help="""
            Delete the existing contents of the output directory before copying over
            files from this run.
            """
        )
        self.add_argument('--lite', action="store_true", help="""
            Run a lite version of the packaging script. This will not collect
            the files and directories under $HOME, i.e., it will not gather
            model artifacts and training output directories. Please use the --copydirs
            option to specify any artifacts in the docker's file-system that you want
            to be collected by this script, as a space-separated list of file/directory
            names following this option.
            """
        )
        self.add_argument('-s', '--stdout', type=str, required=True, help="""
            This is the fully path-qualified filename for the stdout for the training run.
            """
        )
        self.add_argument('-e', '--stderr', type=str, required=False, help="""
            This is the fully path-qualified filename for the stderr for the training run,
            if different from the stdout.
            """
        )
        self.add_argument('-y', '--yaml_config', type=str, required=False, help="""
            This is the fully path-qualified filename for the yaml config file used for the training run.
            """
        )
        self.add_argument('-cmd', '--cmd_name', type=str, required=True, help="""
            This is the command name that the script should search for in the history of the current shell
            to get the full training command-line invocation that the user ran.
            Caveats:
            This script should be invoked from a Habana TensorFlow/PyTorch Docker container
            bash shell and not from within a Python interpreter.
            This functionality depends on the availability of $HOME/.bash_history in the docker container
            shell. If this file does not exist, it can be created by running \'history -a\' in the
            container.
            """
        )
        self.add_argument('--copydirs', nargs="+", default = [], help="""
            Space-separated list of names of directories and files that are in the docker container, to be copied
            to the output directory. For e.g. this could be /software/data/bert_checkpoints. Unless you are using
            the --lite option, this list would typically not include files or directories that are under $HOME
            in the Habana docker container.
            """
        )


    def parse_args(self):
        args = super(GatherInfoDockerArgParser, self).parse_args()
        args.outdir = str(get_canonical_path(args.outdir).joinpath(TMPDIR))
        if os.geteuid() != 0:
            print("*** Rerun this script as user 'root' or as sudo ***\n\n")
            self.print_help()
            sys.exit(1)
        return args

class PackagingScriptDocker():
    STANDARD_FILE_NAMES = {"stdout" : "cmd_stdout.txt",
                           "stderr" : "cmd_stderr.txt",
                           "cmdline" : "cmdline_invocation.txt",
                           "yamlconfig" : "model_yaml_config.txt",
                           "envvars" : "env_vars.txt",
                           "habanalogs" : "habana_logs",
                           "dockerc_pypkgs" : "docker_container_python_pkgs.txt",
                           "dockerc_history" : "docker_container_history.txt",
                           "dockerc_diskusage" : "docker_container_disk_usage.txt",
                           "dockerc_homedir" : "docker_container_homedir_and_model_artifacts",
                           "dockerc_copydir" : "docker_container_additional_dirs",
                           "machine_hostname" : "machine_hostname.txt",
                           "machine_ipaddr" : "machine_ipaddr.txt",
                           "machine_hlsmi" : "machine_hlsmi.txt",
                           "machine_lspci" : "machine_lspci.txt",
                           "machine_cpuinfo" : "machine_cpuinfo.txt",
                           "machine_cpumode" : "machine_cpumode.txt",
                           "machine_nicstatus" : "machine_Gaudi_NIC_status.txt"}
    STANDARD_INFO_FILE_NAMES = {}

    def __init__(self, args, outdir_path):
        self.args = args
        self.outdir_path = outdir_path

    def get_outdir_filename(self, filename):
        return str(self.outdir_path.joinpath(filename))

    def run_cmd(self, cmd=str):
        print(cmd)
        with subprocess.Popen(cmd, shell=True, executable='/bin/bash') as proc:
            proc.wait()

    # Save the file specified to the output directory
    def saveFile(self, info_file_name, info_file_type=None):
        if info_file_type is not None:
            save_cmd = f"cp -r -f -L --preserve=timestamps {info_file_name} " + self.get_outdir_filename(PackagingScriptDocker.STANDARD_FILE_NAMES[info_file_type])
        else:
            save_cmd = f"cp -r -f -L --preserve=timestamps {info_file_name} " + self.get_outdir_filename(info_file_name)
        self.run_cmd(save_cmd)

    # Save the file specified to the output directory - without dereferencing symbolic links
    def saveFileNoSymlink(self, info_file_name, info_file_type=None):
        if info_file_type is not None:
            save_cmd = f"cp -r -f --preserve=timestamps {info_file_name} " + self.get_outdir_filename(PackagingScriptDocker.STANDARD_FILE_NAMES[info_file_type])
        else:
            save_cmd = f"cp -r -f --preserve=timestamps {info_file_name} " + self.get_outdir_filename(info_file_name)
        self.run_cmd(save_cmd)

    # Save the TMPFILENAME
    def saveTmpFile(self, info_file_type):
        save_cmd = f"mv {self.args.outdir}/{TMPFILENAME} " + self.get_outdir_filename(PackagingScriptDocker.STANDARD_FILE_NAMES[info_file_type])
        self.run_cmd(save_cmd)

    # Save the string to the appropriate file name in the output directory
    def saveInfo(self, info, info_key):
        save_cmd = f"echo {info} > " + self.get_outdir_filename(STANDARD_INFO_FILE_NAMES[info_key])
        self.run_cmd(save_cmd)

    # Save stdout & stderr from the run
    def saveCmdOutputs(self):
        try:
            if os.path.exists(self.args.stdout):
                self.saveFile(self.args.stdout, "stdout")
            else:
                print(f"The filename specified by {self.args.stdout} does not exist")
                raise Exception(f"{self.args.stdout} is not a valid filename")
            if self.args.stderr is not None:
                if os.path.exists(self.args.stderr):
                    self.saveFile(self.args.stderr, "stderr")
                else:
                    print(f"The filename specified by {self.args.stderr} does not exist")
                    raise Exception(f"{self.args.stderr} is not a valid filename")
        except Exception as exc:
            raise RuntimeError("Error in saveCmdOutputs") from exc

    # Save exact command line the user ran, along with command line options and environment flags used to invoke the training
    # Save yaml config if available
    def saveCmdlineAndOptions(self):
        try:
            self.run_cmd("history -a")
            this_filename = os.path.basename(__file__)
            cmd = f"grep {self.args.cmd_name} $HOME/.bash_history | grep -v {this_filename} | tail -1 > " + self.get_outdir_filename(PackagingScriptDocker.STANDARD_FILE_NAMES["cmdline"])
            self.run_cmd(cmd)
            self.saveFile(self.args.yaml_config, "yamlconfig")
        except Exception as exc:
            raise RuntimeError("Error in saveCmdlineAndOptions") from exc

    # Save the names and values of all environment variables set in the environment
    def saveEnvVars(self):
        try:
            if os.path.exists(self.get_outdir_filename(TMPFILENAME)):
                os.remove(self.get_outdir_filename(TMPFILENAME))
            cmd = f"env > " + self.get_outdir_filename(PackagingScriptDocker.STANDARD_FILE_NAMES["envvars"])
            self.run_cmd(cmd)
        except Exception as exc:
            raise RuntimeError("Error in saveEnvVars") from exc

    # Save Synapse, HCL logs, etc. Habana log files
    def saveHabanaLogs(self):
        try:
            os.makedirs(self.get_outdir_filename(PackagingScriptDocker.STANDARD_FILE_NAMES["habanalogs"]), mode=0o777, exist_ok=True)
            logs_path = os.getenv('HABANA_LOGS')
            self.saveFile(f"{logs_path}/*", "habanalogs")
        except Exception as exc:
            raise RuntimeError("Error in saveHabanaLogs") from exc

    def isValidDirToCopy(self, start_path, outdir_basename):
        with os.scandir(start_path) as it:
            for entry in it:
                if entry.is_dir():
                    if (os.path.basename(entry.path) == outdir_basename) or not self.isValidDirToCopy(entry.path, outdir_basename):
                        return False
        return True

    # Get the files in the home directory to be copied to the outdir: these are any directories that do not contain the outdir
    def getHomeDirContentToSave(self):
        start_path = get_canonical_path("$HOME")
        sv_files = []
        outdir_basename = os.path.basename(self.outdir_path.parent)
        with os.scandir(start_path) as it:
            for entry in it:
                if entry.is_dir() and (os.path.basename(entry.path) != outdir_basename) and self.isValidDirToCopy(entry.path, outdir_basename):
                    sv_files.append(entry.path)
                    print(entry.path)
                elif entry.is_file():
                    sv_files.append(entry.path)
                    print(entry.path)
        return sv_files

    # Model artifacts: dataset, python code, python package dependencies
    # Data generated from training, including TF events and checkpoints
    # To keep things simple, copy all non-outdir content in $HOME to the outdir
    def saveModelSpecificArtifacts(self):
        if self.args.lite:
            print("Skipping copying model artifacts, datasets, training run output directories, and other contents of $HOME since --lite option is specified...")
            return
        try:
            os.makedirs(self.get_outdir_filename(PackagingScriptDocker.STANDARD_FILE_NAMES["dockerc_homedir"]), mode=0o777, exist_ok=True)
            files_dirs_to_save = self.getHomeDirContentToSave()
            print('files_dirs_to_save: ', files_dirs_to_save)
            for sv_file in files_dirs_to_save:
                self.saveFile(f"{sv_file}", "dockerc_homedir")
        except Exception as exc:
            raise RuntimeError("Error in saveModelSpecificArtifacts") from exc

    # Save additional directories not in $HOME that the user has asked to be copied
    def saveAdditionalCopyDirs(self):
        try:
            if self.args.copydirs != []:
                print(f"copydirs = {self.args.copydirs}")
                os.makedirs(self.get_outdir_filename(PackagingScriptDocker.STANDARD_FILE_NAMES["dockerc_copydir"]), mode=0o777, exist_ok=True)
                for sv_file in self.args.copydirs:
                    if os.path.exists(sv_file):
                        self.saveFileNoSymlink(f"{sv_file}", "dockerc_copydir")
                    else:
                        print(f"Skipping {sv_file} from copydirs since this path does not exist")
        except Exception as exc:
            raise RuntimeError("Error in saveAdditionalCopyDirs") from exc

    # Save docker container information:
    # - Save the container command history for settings made within the container
    # - Python packages/versions list
    # - Amount of disk space used in the container
    # - TBD: save the docker image?
    def saveDockerRunParameters(self):
        try:
            self.run_cmd("history -a")
            self.saveFile("$HOME/.bash_history", "dockerc_history")
            cmd = f"pip list > " + self.get_outdir_filename(PackagingScriptDocker.STANDARD_FILE_NAMES["dockerc_pypkgs"])
            self.run_cmd(cmd)
            cmd = f"du -s -h ~/ > " + self.get_outdir_filename(PackagingScriptDocker.STANDARD_FILE_NAMES["dockerc_diskusage"])
            self.run_cmd(cmd)
        except Exception as exc:
            raise RuntimeError("Error in saveDockerRunParameters") from exc

    # Machine-specific information:
    # - Host name, IP address
    # - Number and status of Gaudi cards, configuration (HLS1/HL200), firmware version, NIC ports status
    # - CPU scaling_governor
    def saveMachineStatus(self):
        try:
            cmd = f"hostname > " + self.get_outdir_filename(PackagingScriptDocker.STANDARD_FILE_NAMES["machine_hostname"])
            self.run_cmd(cmd)
            cmd = f"hostname -I > " + self.get_outdir_filename(PackagingScriptDocker.STANDARD_FILE_NAMES["machine_ipaddr"])
            self.run_cmd(cmd)
            cmd = f"hl-smi -q > " + self.get_outdir_filename(PackagingScriptDocker.STANDARD_FILE_NAMES["machine_hlsmi"])
            self.run_cmd(cmd)
            cmd = f"lspci > " + self.get_outdir_filename(PackagingScriptDocker.STANDARD_FILE_NAMES["machine_lspci"])
            self.run_cmd(cmd)
            cmd = f"cat /proc/cpuinfo > " + self.get_outdir_filename(PackagingScriptDocker.STANDARD_FILE_NAMES["machine_cpuinfo"])
            self.run_cmd(cmd)
            cmd = f"cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > " + self.get_outdir_filename(PackagingScriptDocker.STANDARD_FILE_NAMES["machine_cpumode"])
            self.run_cmd(cmd)
            # Todo: NIC ports status
        except Exception as exc:
            raise RuntimeError("Error in saveMachineStatus") from exc


    # This generates a <args.outdir>.tar.gz in the parent directory of args.outdir
    # This generates a <args.outdir>.tar.gz in the parent directory of args.outdir
    def generateTarball(self):
        try:
            cmd = f"chmod -R 755 {str(self.outdir_path)}"
            self.run_cmd(cmd)
            parent_dir = os.path.dirname(self.outdir_path)
            tardir_name = os.path.basename(os.path.normpath(self.outdir_path))
            tarfile_name = f"{tardir_name}.tar.gz"
            os.chdir(parent_dir)
            print(f"Creating {tarfile_name}...")
            tar = tarfile.open(tarfile_name, 'x:gz')
            tar.add(tardir_name)
            tar.close()
            cmd = f"chmod -R 755 {tarfile_name}"
            self.run_cmd(cmd)
        except Exception as exc:
            raise RuntimeError("Error in generateTarball") from exc

    def uploadTarball(self): pass

    def run(self):
        self.saveCmdOutputs()
        self.saveCmdlineAndOptions()
        self.saveEnvVars()
        self.saveHabanaLogs()
        self.saveModelSpecificArtifacts()
        self.saveAdditionalCopyDirs()
        self.saveDockerRunParameters()
        self.saveMachineStatus()
        self.generateTarball()
        self.uploadTarball()
        print("Packaging script completed successfully (from docker container)")

"""
   Script will gather and save the following information from the training run, to the output directory:
   - stdout & stderr from the run
   - The exact command line the user ran, along with command line options and environment flags used to invoke the training
   - yaml config file used for the run
   - Values of all environment variables set in the environment
   - Synapse, HCL logs

   - Docker container information:
     - Docker run command-line, including the docker image version used from Artifactory
     - Settings made within the container
     - Python packages/versions list
     - Amount of disk space used in the container
     - Optional - save the docker image

   - Machine-specific information:
     - Host name, IP address
     - Number and status of Gaudi cards, configuration (HLS1/HL200), firmware version, NIC ports status
     - CPU scaling_governor

   - gdb and Synapse profiler information:
     - TBD

   - AWS instance-specific information:
     - TBD

   Save the output directory as a tar.gz and upload to <TBD>
"""
def main():
    argparser = GatherInfoDockerArgParser()
    if len(sys.argv) == 1:
        argparser.print_help()
        sys.exit(1)
    args = argparser.parse_args()

    outdir_path = create_output_dir(args.outdir, args.clear)

    with open(outdir_path.joinpath(EXECCMDLINEFILE), "w") as fdout:
        fdout.write("Command executed as: python3 {}\n".format(" ".join(sys.argv)))

    pkg = PackagingScriptDocker(args, outdir_path)
    pkg.run()

if "__main__" == __name__:
    main()

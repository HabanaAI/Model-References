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

This python script is called for collecting docker related info from the host:
  docker inspect <container_ID>
  docker stats --no-stream --no-trunc <container_ID>
  docker ps <container_ID> to get the Habana docker image being run
It needs to be run as root or with sudo from a shell on the same machine on
which the user is running the docker training session.
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

class GatherInfoHostArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(GatherInfoHostArgParser, self).__init__()
        self.add_argument('-o', '--outdir', type = str, default = './', help="""
            The output directory where all the files and other information will be stored.
            The output will be stored as an archive as well as the actual directory where
            all the contents are copied.
            """
        )
        self.add_argument('-c', '--clear', action="store_true", help="""
            Delete the existing contents of the output directory before copying over
            files from this run.
            """
        )
        self.add_argument('-i', '--container_id', type = str, required=True, help="""
            This is the Docker container ID for the running docker session, obtained
            by \"docker ps\" and selecting the CONTAINER ID for the session for
            which we want to run the packaging script.
            """
        )

    def parse_args(self):
        args = super(GatherInfoHostArgParser, self).parse_args()
        args.outdir = str(get_canonical_path(args.outdir).joinpath(TMPDIR))
        if os.getuid() != 0:
            print("*** Rerun this script as user 'root' or as sudo ***\n\n")
            self.print_help()
            sys.exit(1)
        return args

class PackagingScriptHost():
    STANDARD_FILE_NAMES = {"docker_inspect" : "docker_inspect.txt", "docker_stats" : "docker_stats.txt", "docker_ps_cmd" : "docker_ps_cmd.txt"}

    def __init__(self, args, outdir_path):
        self.args = args
        self.outdir_path = outdir_path

    def get_outdir_filename(self, filename):
        return str(self.outdir_path.joinpath(filename))

    def run_cmd(self, cmd=str):
        print(cmd)
        with subprocess.Popen(cmd, shell=True, executable='/bin/bash') as proc:
            proc.wait()

    def saveDockerContainerInfoFromHost(self):
        try:
            cmd = f"docker inspect {self.args.container_id} > " + self.get_outdir_filename(PackagingScriptHost.STANDARD_FILE_NAMES["docker_inspect"])
            self.run_cmd(cmd)
            cmd = f"docker stats --no-stream --no-trunc {self.args.container_id} > " + self.get_outdir_filename(PackagingScriptHost.STANDARD_FILE_NAMES["docker_stats"])
            self.run_cmd(cmd)
            query_str = f"\'CONTAINER ID|{self.args.container_id}\'"
            cmd = f"docker ps|grep -E {query_str} > " + self.get_outdir_filename(PackagingScriptHost.STANDARD_FILE_NAMES["docker_ps_cmd"])
            self.run_cmd(cmd)
        except Exception as exc:
            raise RuntimeError("Error in saveDockerContainerInfoFromHost") from exc

    # This generates a <args.outdir>.tar.gz in the parent directory of args.outdir
    def generateTarball(self):
        try:
            cmd = f"chmod -R 755 {str(self.outdir_path)}"
            self.run_cmd(cmd)
            parent_dir = os.path.dirname(self.outdir_path)
            tardir_name = os.path.basename(os.path.normpath(self.outdir_path))
            tarfile_name = f"{tardir_name}.tar.gz"
            os.chdir(parent_dir)
            tar = tarfile.open(tarfile_name, 'x:gz')
            tar.add(tardir_name)
            tar.close()
            cmd = f"chmod -R 755 {tarfile_name}"
            self.run_cmd(cmd)
        except Exception as exc:
            raise RuntimeError("Error in generateTarball") from exc

    def uploadTarball(self): pass

    def run(self):
        self.saveDockerContainerInfoFromHost()
        self.generateTarball()
        self.uploadTarball()
        print("Packaging script completed successfully (from host)")

def main():
    argparser = GatherInfoHostArgParser()
    if len(sys.argv) == 1:
        argparser.print_help()
        sys.exit(1)
    args = argparser.parse_args()

    outdir_path = create_output_dir(args.outdir, args.clear)

    with open(outdir_path.joinpath(EXECCMDLINEFILE), "w") as fdout:
        fdout.write("Command executed as: python3 {}\n".format(" ".join(sys.argv)))

    pkg = PackagingScriptHost(args, outdir_path)
    pkg.run()

if "__main__" == __name__:
    main()

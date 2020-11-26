"""
This code is based on: mepy_algo/appcode/DL/freespace/training/sm_setup.py
"""

from subprocess import call, Popen, PIPE, STDOUT
import os
import sys


def import_install(module_versions={}, req_file=None):
    """
    Import a python package.
    Installs using pip, if it isn't already installed.
    """
    from pkgutil import iter_modules
    call(["pip", "install", "--upgrade", "pip"])
    # Check if the module is installed
    for module in module_versions.keys():
        call(["pip", "install", "--no-cache-dir", "%s==%s" % (module, module_versions[module])])
    if req_file:
        if req_file.startswith("s3"):
            tmp_req_file = '/tmp/requirements.txt'
            command = ["aws", "s3", "cp", req_file, tmp_req_file]
            p = Popen(command, stdout=PIPE, stderr=STDOUT)
            _ = p.communicate()
            req_file = tmp_req_file
        pip = "pip" if sys.version_info.major < 3 else "pip3"
        call([pip, "install", "-r", req_file])


def sm_setup(req_file=None):
    import_install(module_versions={'boto3': '1.14.46', 'awscli': '1.18.123', 's3fs': '0.4.2', 'tfmpl': '1.0.2',
                                    'opencv-python': '4.4.0.42'}, req_file=req_file)
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
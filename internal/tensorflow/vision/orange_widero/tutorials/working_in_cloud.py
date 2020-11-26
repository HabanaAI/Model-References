# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# Wide paragraph in notebook:
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# In this notebook, we show examples for running jobs and interfacing with the cloud. Some tasks may have several APIs, which may be confusing. We also mention what knowledge is still missing in this context.

# Some API here would be command-line, which can be executed from jupyter notebook with '!'.
#
# Another option:

# terminal command
import subprocess as sp
# ecl = execute (in) command-line
def ecl(cmd_line):
    cmd = cmd_line.split(' ')
    output = sp.Popen( cmd, stdout=sp.PIPE ).communicate()[0]
    return output


print(ecl('aws s3 ls s3://mobileye-team-stereo/temp/test/'))

# # Running in cloud

# We currently use two methods for "batch" runs. For working in cloud, both wrap the cloud-run API of the nebula package (cloud team), see 
# http://wiki.mobileye.com/mw/index.php/cloud-run#Python_API, with the `run_jobs` function.
#
# Aaron Siegal suggested using `jobs_runner`, which by default uses the 'algo-run:9' docker image (probably maintained by him). 
#
# A second option is calling the same docker image with `me_pjs`.

# ## `jobs_runner`

# ### Simple save with jobs_runner

# A simple script for testing:

import os
cwd = os.getcwd()
SCRIPT_PATH = cwd + '/functions_for_examples/example_cloud_function.py'
SCRIPT_PATH

# Saving a simple array in cloud:

# + code_folding=[0]
# Saving a simple array in cloud:
from me_jobs_runner import Task, dispatch_tasks, CLOUD

n_files = 2 # 2 runs, just for testing

task_name = 'test_cloud'
output_path = '/mobileye/algo_STEREO3/stereo/temp/test'
cmd_lines = []
cmd_names = []
for j in range(n_files):
    cmd_line  = "python %s -i %d" % (SCRIPT_PATH, j)
    cmd_lines.append(cmd_line)
    cmd_name = "cloud_example" + str(j)
    cmd_names.append(cmd_name)
print("cmd_lines: ", cmd_lines)
print("cmd_names: ", cmd_names)


# me_jobs_runner:
task = Task(task_name=task_name, commands=cmd_lines, input_folders_or_clips='',
                output_folders=output_path, timeout_minutes=5, memory_gb=1,
                command_names=cmd_names, dependecies=[])  


dispatch_tasks([task], task_name=task_name, log_dir=os.path.join(output_path, 'cloud_logs'),
                        dispatch_type=CLOUD) # container_image='algo-run:9' (insert & check)

# -

# Check that files were saved in s3 (see about commands later):

# !aws s3 ls s3://mobileye-team-stereo/temp/test/  --human-readable --summarize

# ### MeClip in cloud

# Now also using `MeClip`, saving frames from clips:

# +
# Saving frames from clips in cloud:
from me_jobs_runner import Task, dispatch_tasks, CLOUD

n_files = 2 # 2 runs, just for testing

task_name = 'test_cloud'
output_path = '/mobileye/algo_STEREO3/stereo/temp/test'
clip_names = ['18-08-29_09-59-46_Alfred_Front_0085', '19-03-19_15-42-25_Alfred_Front_0058']
cmd_lines = []
cmd_names = []
for j,clip_name in enumerate(clip_names):
    cmd_line  = "python %s -c %s" % (SCRIPT_PATH, clip_name)
    cmd_lines.append(cmd_line)
    cmd_name = "cloud_example_clip_" + str(j)
    cmd_names.append(cmd_name)
print("cmd_lines: ", cmd_lines)
print("cmd_names: ", cmd_names)


# me_jobs_runner:
task = Task(task_name=task_name, commands=cmd_lines, input_folders_or_clips='',
                output_folders=output_path, timeout_minutes=5, memory_gb=1,
                command_names=cmd_names, dependecies=[])  


dispatch_tasks([task], task_name=task_name, log_dir=os.path.join(output_path, 'cloud_logs'),
                        dispatch_type=CLOUD, extra_dependecies=[]) 

# -

# If additional modules are required, add their full path in `extra_dependecies` (with the spelling mistake).
#
# For example:
#
# `extra_dependecies=['/mobileye/algo_STEREO3/ohrl/gitlab/stereo/stereo/data/clip_utils.py']`
#

# !aws s3 ls s3://mobileye-team-stereo/temp/test/  --human-readable --summarize

# ### Missing in `jobs_runner`
# * The 'interactive' mode is not implemented.

# ## me_pjs

# * If `poll=False`, the execution will wait until all jobs will finish, and the output can be used. If `poll=False`, will run in the background.

# ### Simple save with mepjs

# +
# Functions:
import numpy as np
import os
from devkit.cloud.s3_filesystem import upload_file
def save_to_our_s3(a, key):
    # Save numpy array `a` to `key` ('s3://mobileye-team-stereo/'+key).
    # Example for a `key`: 'temp/try/array.npz'
    file_name = os.path.split(key)[1]
    np.savez('/tmp/' + file_name, a=a)
    full_s3_path = 's3://mobileye-team-stereo/' + key
    upload_file(full_s3_path, '/tmp/' + file_name)
    print("Uploaded '%s'." % full_s3_path)
    return None

def simple_numpy_save(file_ind):
    # Save a simple numpy array.
    file_name = 'array' + str(file_ind) + '.npz'
    a = np.ones((2, 5))
    save_to_our_s3(a, 'temp/test/' + file_name)
    return None


# -

simple_numpy_save(7)

# +
n_files = 2 # 2 runs, just for testing

from me_pjs.python_job_server import PythonJobServer, SchedulerType
mepjs_dir = 's3://mobileye-team-stereo/temp/test/mepjs' # the output folder is required to be in s3
pjs = PythonJobServer(task_scheduler_type=SchedulerType.AWS, folder=mepjs_dir, poll=False, docker_name='algo-run:9',
                                                        job_name='cloud_mepjs_simple_example', full_copy_modules=[stereo],
                                                    max_rerun=0, timeout_minutes=5, memory_gb=8)
inds_vec = (4,5) # tuple of input to function
out = pjs.run(simple_numpy_save, inds_vec)
# -

# ### MeClip with mepjs

from devkit.clip import MeClip
def simple_save_from_clip(clip_name):
    # Save a frame from `clip_name`.
    clip = MeClip(clip_name)
    frame = clip.get_frame(frame=5, camera_name='main', tone_map='ltm')
    im = frame[0]['pyr'][-1].im
    save_to_our_s3(im, 'temp/test/' + clip_name + '.npz')
    return None


from me_pjs.python_job_server import PythonJobServer, SchedulerType
mepjs_dir = 's3://mobileye-team-stereo/temp/test/mepjs'
pjs = PythonJobServer(task_scheduler_type=SchedulerType.AWS, folder=mepjs_dir, poll=False, docker_name='algo-run:9',
                                                        job_name='cloud_mepjs_clip_example', full_copy_modules=[stereo],
                                                    max_rerun=0, timeout_minutes=5, memory_gb=8)
clip_names = ('18-08-29_09-59-46_Alfred_Front_0085', '19-03-19_15-42-25_Alfred_Front_0058') # tuple of input to function
out = pjs.run(simple_save_from_clip, clip_names)

# ### Missing in mepjs
# * Interactive mode also not implemented here.

# # Cloud-jobs utils

# For monitoring and controlling our cloud tasks. See wiki for some (currently partial) details:
#
# http://wiki.mobileye.com/mw/index.php/cloud-jobs-jobim
#
# The terms they use here are "run" for a batch of instances sent together, made out of "jobs" which are the instances inside the "run".

# See the runs of a user (`-u <user_name>`) in the last day (`-t 1d`) or hours (`-t 5h`):

# !cloud-jobs-jobim -u ohrl -t 20h

# You may also bump into the API:
#
# # !cloud-jobs -u ohrl -j
#
# This is an older version, much slower (not recommended)

# Find the failed jobs in a specifc run, given by `-r <RUN_ID>`:

# !cloud-debug -r test_cloud_ohrl_20200804_171748_226802  --failed --jobim

# For debugging, observe the log of a specific job in a specifi run (`-r <RUN_ID>:<#JOB>`):

# !cloud-debug -r test_cloud_ohrl_20200804_171748_226802:0 --jobim --all-stdout

# Kill a run:

# !cloud-kill -r test_cloud_ohrl_20200804_171748_226802

# # Interface with s3

# There are several ways to interface with s3, the main are using command-line commands, using the boto3 package, or using the devkit.cloud package.
#
# I think the devkit.cloud commands are convenient with python, and the s3 CLI in the terminal.

# ## s3 CLI

# The AWS command-line interface.
#
# See:
# https://docs.aws.amazon.com/cli/latest/reference/s3/

# The copy `cp` commmand serves both for downloads and uploads:

# !aws s3 cp s3://mobileye-team-stereo/temp/test/18-08-29_09-59-46_Alfred_Front_0085.npz /tmp/

# List files:
# !aws s3 ls s3://mobileye-team-stereo/temp/test/
# # !aws s3 ls s3://mobileye-team-stereo/temp/test --recursive

# ## boto3

# boto3 is the AWS package for python.
#
# See: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html

# Initiate an object:

import boto3
s3 = boto3.resource('s3')
bucket = s3.Bucket(name='mobileye-team-stereo')

# A file for testing API:
# !echo "Hello there!" > /tmp/hello.txt

# !cat /tmp/hello.txt

# `upload file( <local_path>, <s3_key>)` :
#
#

bucket.upload_file('/tmp/hello.txt', 'temp/test/hello.txt')

# List files (somewhat cumbersome):

for obj in bucket.objects.filter(Prefix='temp/test/'):
    print(obj.key)

# `download_file(<s3_key>, <local_path>)`:

bucket.download_file('temp/test/hello.txt', '/tmp/hello2.txt')

# Check:

# !cat /tmp/hello2.txt

# Delete a key:

# +
key = 'temp/test/hello.txt'
for obj_temp in bucket.objects.filter(Prefix=key):
    if obj_temp.key == key:
        our_obj = obj
    
our_obj.delete()
# -

# Sync one folder to another:

# !aws s3 sync s3://mobileye-team-stereo/temp/test /tmp/test  

# ## devkit.cloud

# devkit have several boto3 commands conveniently wrapped. Below are some examples. For more, see:
#
# http://wiki.mobileye.com/mw/index.php/Cloud_Devkit_Tools
#
# * `devkit.cloud.s3_utils`: mostly for s3 url
# * `devkit.cloud.s3_filesystem`: mostly for interfacing (os-like) with s3
# * `devkit.cloud.file_abstraction`: generalization of os-methods, applying both to s3 and local
#
# We also have some functions in `stereo.common.s3_utils`, there may be some redundancy.

# Splitting the bucket name and key:

from devkit.cloud.s3_utils import split_s3_url
bucket_name, key = split_s3_url('s3://mobileye-team-stereo/temp/test/array1.npz')
print(bucket_name, key)

# Abstraction of `glob`:

from devkit.cloud.file_abstraction import glob_s3, remove_path
glob_s3('s3://mobileye-team-stereo/temp/test/*')

glob_s3('/tmp/*.txt')

remove_path('/tmp/hello2.txt')
remove_path('s3://mobileye-team-stereo/temp/test/hello.txt')

# Upload and download (pull):

# +
from devkit.cloud.s3_filesystem import upload_file, delete_key, pull_file

upload_file('s3://mobileye-team-stereo/temp/test/hello.txt', '/tmp/hello.txt')
pull_file('s3://mobileye-team-stereo/temp/test/hello.txt', '/tmp/hello2.txt')
# -

# # Partition

# For reading/saving many files simultaneously with s3. The limit without partitioning the bucket is supposed to be ~3-5K per second.
#
# Our Bucket should currently have 16 partitions when using with the `Party` API below. If this will not be sufficient, then shay Margalit can demand more for us (up to the maximal 4096).

# ## `Party` module

# http://wiki.mobileye.com/mw/index.php/s3-partition

# The `Party` method works similar to the `file2path` we use to avoid a large flat folder. All files which utilize the partition are saved in the 'parts/' folder.
#
# It is different from `file2path` in that the intermediate subfolders are created in the beginning of key, after 'parts/'.
#
# The 'key_index' parameter determines according to which file/path-names the subfolders will be named.
#
# In the example, if we want to partition according to the filename 'pickletest.npz', then we count how many levels it is below 'parts/'

import numpy as np
a = np.random.rand(20,50)

from nebula.common.party import Party
import io
import pickle
bucket = 'mobileye-team-stereo'
s3_key = 'parts/temp/test/pickletest.npz'
client = Party(key_index=3) # 'pickletest.npz' is 3 levels below 'parts/'
my_array_data = io.BytesIO()
pickle.dump(a, my_array_data)
my_array_data.seek(0)
args = {'Bucket': bucket, 'Key': s3_key, 'Body': my_array_data}
client.put_object(**args)

# Download the file:

args = {'Bucket': bucket, 'Key': s3_key}
res = client.get_object(**args)
my_array_data2 = io.BytesIO(res['Body'].read())
my_array_data2.seek(0)
my_array2 = pickle.load(my_array_data2)

# Verify that it has been downloaded correctly:

print(my_array2.shape)
print(my_array2.mean())
print(a.mean())

# Find the intermediate folder for the file:

client.get_object_key(s3_key)

# In this case, the intermediate folder is '703', and therefore the full key is:
#
# ``parts/703/temp/test/pickletest.npz``

# ## cloud-s3-invertory

# http://wiki.mobileye.com/mw/index.php/cloud-s3-inventory

# * Search for files, with the partition-subfolder is taken into account:

# !cloud-s3-inventory --ls s3://mobileye-team-stereo/parts/temp/test/

# Note that the inventory is created once a day, so a recent file will not be available when using this tool.
#
# Probably it is therefore not very useful, and we're currently better off with using `client.get_object_key`.

# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# +
# For all steps, make sure you know which python trees (stereo, mepy_algo, devkit) you use, and what is your PYTHONPATH.
# for example to run scripts over NETBATCH you'll need the code to be accessible from NETBATCH machines, i.e. not in 
# /homes/..

# Currently I'm using code in /mobileye/algo_STEREO2/stereo/code.freeze
# and my PYTHONPATH is - 
# setenv PYTHONPATH /mobileye/algo_STEREO2/stereo/code.freeze/:/mobileye/shared/Tools/di_scripts:/mobileye/shared/Tools/qa_algo_tools/avm_tools:/mobileye/algo_STEREO2/stereo/code/site-packages/

# +
# step 1 - predump:
# predump is consists of 4 major components:
# 1) Mest itrks with VD3D and PEDS info
# 2) GTEM itrks for 6 cameras - main,rear + 4 corners
# 3) Yotam's cam2cam calibration
# 4) Lidar dump

# The first 3 steps are done by DATA_ENG team.
# The current output is at 

predump_dir = '/mobileye/algo_STEREO3/old_stereo/data/data_eng/'

# Then the predump should be initially indexed by

from stereo.data.predump_utils import PreDumpIndex
predump_index = PreDumpIndex(predump_dir, rebuild_index=True, save_index=True, new_format=True)

# which will take few minutes, and will save a pickle at os.path.join(predump_dir, 'predump_index.pickle')
# be carefull not to override this file, as other procesess might use it (TODO - make safer), 
# I've added typos here to be safe

# After the initial indexing, you should always load it like this -
predump_index = PreDumpIndex(predump_dir, rebuild_index=False, save_index=False, new_format=True)

# Then to run the 4th step, you should run lidar_dump_utils like this - 
# >> python /mobileye/algo_STEREO2/stereo/code.freeze/stereo/data/lidar_dump_utils.py 
#           -p /mobileye/algo_STEREO2/stereo/data/data_eng -nbt lidar_dump 
#           -t /mobileye/algo_STEREO2/stereo/code.freeze/stereo

# This will create local lidar dir for each clip

# +
# step 2 - dump1:

# example cmd
# >> python /mobileye/algo_STEREO2/stereo/code.freeze/stereo/data/make_view_dataset.py 
#           -j /mobileye/algo_STEREO2/stereo/code.freeze/stereo/data/conf/full_dump.json 
#           -p /mobileye/algo_STEREO2/stereo/data/data_eng -pf new -nbt view_dataset_v2 
#           --tree_base /mobileye/algo_STEREO2/stereo/code.freeze/ -o /mobileye/algo_RP_NVME/stereo/data/view_dataset_v2

# when this dump is done (longest step..), dataSetIndex should be indexed

from stereo.data.dataset_utils import ViewDatasetIndex
# dataset_dir = '/mobileye/algo_RP_NVME/stereo/data/view_dataset_v2'
dataset_dir = '/mobileye/algo_RP_8/jeff/MOVE/data/dump_phase1'
test_percentage = 0.1
dataSetIndex = ViewDatasetIndex(dataset_dir,index_path=None, rebuild_index=True, 
                                save_index=True, use_file2path=True, 
                                test_percentage=test_percentage, preserve_tt_sessions=False)

# This will index the data, and seperate it to train/test be sessions.
# If clips are added to the dataset, rebuild can be done, with preserving the train/test split of the previus build like this:

# dataSetIndex = ViewDatasetIndex(dataset_dir,index_path=None, rebuild_index=<True>, 
#                                 save_index=<True>, use_file2path=True, 
#                                 test_percentage=test_percentage, preserve_tt_sessions=True)
# 
# # after initial indexing it should be initiated like this:
# dataSetIndex = ViewDatasetIndex(dataset_dir)
# 
# # To be safer you can pass it explicit index_path, instead of the default os.path.join(dataset_dir, 'dataset_index.pickle'):
# index_path = '/mobileye/algo_RP_NVME/stereo/data/view_dataset_v2/my_index.pickle'
# # rebuild index
# dataSetIndex = ViewDatasetIndex(dataset_dir,index_path=index_path, rebuild_index=<True>, 
#                                 save_index=<True>, use_file2path=True, 
#                                 test_percentage=test_percentage, preserve_tt_sessions=True)
# dataSetIndex = ViewDatasetIndex(dataset_dir,index_path=index_path)

# +
# step 3 - dump2:

# example cmd
# >> /homes/davidn/env/sagemaker/bin/python2.7 /mobileye/algo_STEREO2/stereo/code.freeze/stereo/data/make_tf_dataset.py 
#       -j /mobileye/algo_STEREO2/stereo/code.freeze/stereo/data/conf_tf/main_with_pf_tf_dataset.json  
#       --exec_path /mobileye/algo_STEREO2/stereo/code.freeze --run_local false --shuffle false

# shuffle should be ran as true only once to shuffle the frames order in train/test lists.
# be carfull not to change dataSetIndex pickle during running jobs of dump2. It is safer to use explicit index_path

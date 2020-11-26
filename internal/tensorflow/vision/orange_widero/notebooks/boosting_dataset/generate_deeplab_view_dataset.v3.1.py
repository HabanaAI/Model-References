# ---
# jupyter:
#   jupytext:
#     formats: py:light
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

# This notebook is for generation and saving of deeplab segmentation masks, run for images from ViewDataSetIndex (v3.1).

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# ### Imports

# +
from random import shuffle, seed
import os
import time
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
# %matplotlib notebook

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import stereo
from stereo.common.general_utils import batch_list, flatten_mepjs_out
from stereo.common.probe_utils import batch_func
from stereo.data.dataset_utils import ViewDatasetIndex
from stereo.common.visualization.vis_utils import points3d_from_image
from stereo.data.label_utils import Labeler
from itertools import product
from stereo.data.dataset_utils import my_file2path

from me_pjs import PythonJobServer, SchedulerType
# %load_ext autoreload
# %autoreload 2

# + code_folding=[0]
# Make sounds notification
import os; import time
def plays():
    for i in range(4):
        os.system("printf '\a'")
        time.sleep(0.2)


# -

CENTER_VIEWS = ['main_to_main', 'rear_to_rear', 
                'frontCornerLeft_to_frontCornerLeft', 'frontCornerRight_to_frontCornerRight',
                'rearCornerLeft_to_rearCornerLeft', 'rearCornerRight_to_rearCornerRight'
               ]


# ### Deeplab function

@batch_func
def save_deeplab(frame):
    views=CENTER_VIEWS
    print("In 'save_deeplab'")
    print("frame:", frame)
    print(views)
    labeler = Labeler('deeplab')
    for view_name in views:
        print("view_name: ", view_name)
        
        view = vdsi.read_views([view_name], frame)[view_name]
        image = view['image']
        T = view['path']
        print("Read the view %s" % view_name)
#         deeplab_save_path = os.path.splitext(view_path)[0] + '_deeplab.npz'
        deeplab_save_path = my_file2path('/mobileye/algo_STEREO3/stereo/data/sky/view_dataset.v3.1/deeplab/' + os.path.split(view_path)[1]) 
        if os.path.exists(deeplab_save_path):
            print("File %s exists, skipping." % deeplab_save_path)
            continue
        
        t1 = time.time()
        seg_mask, _ = labeler.generate_segmentation('deeplab',image)
        t2 =time.time()
        print("Segmented image", (t2-t1))
        
        
        if not os.path.exists(os.path.split(deeplab_save_path)[0]):
            os.mkdir(os.path.split(deeplab_save_path)[0])
        
        np.savez(deeplab_save_path, seg_mask=seg_mask)
        print("Saved: %s" % deeplab_save_path)

    print("Finished views in current frame.")
    return 1

# ### Prepare vdsi for NB

save_blacklist = False
dataset_dir = '/mobileye/algo_STEREO3/stereo/data/view_dataset.v3.1'
mepjs_dir = '/mobileye/algo_STEREO3/stereo/data/sky/view_dataset.v3.1/mepjs'


vdsi = ViewDatasetIndex(dataset_dir)
labeler = Labeler('deeplab')

frames = vdsi.test_frames_list + vdsi.train_frames_list
frames = list(set(frames))
# frames = frames[:100] # for debug
# CENTER_VIEWS = CENTER_VIEWS[:2] # for debug

len(frames) / 1e6

# +
# args_in_batched = batch_list(args_in, 3)
frames_batched = batch_list(frames, 5)

# frames_batched = frames_batched[:2] # for debug
# -

len(frames_batched)

len(frames_batched[0])

# + [markdown] heading_collapsed=true
# ### Check local

# + hidden=true
frames[0]

# + hidden=true
vdsi.dataset_dir

# + hidden=true
view = vdsi.read_views([center_views[0]], frames[1])

# + hidden=true
view['main_to_main']['path']

# + hidden=true
from stereo.data.dataset_utils import my_file2path
my_file2path('/mobileye/algo_STEREO3/stereo/data/view_dataset.v2.3/18-08-20_14-29-22_Alfred_Front_0048_main_to_main_0093827.npz')

# + hidden=true
type(view['main_to_main']['image'])

# + hidden=true
save_deeplab(frames_batched[0])
# -

# ### Run batch (NB)

pjs = PythonJobServer(task_scheduler_type=SchedulerType.NET_BATCH,
                      folder=mepjs_dir, poll=False, job_name='deeplab_vdsi3.1',
                      full_copy_modules=[stereo], max_rerun=0, timeout_minutes=600,
                                    memory_gb=12)
out_batched = pjs.run(save_deeplab, frames_batched)
out = flatten_mepjs_out(out_batched, frames_batched)


plays() # notify end of run with sound

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: stereo_deps_py2_v0.1.0
#     language: python
#     name: stereo_deps_py2_v0.1.0
# ---

# +
import numpy as np
# %matplotlib inline
from matplotlib import pyplot as plt

import json
import os
from tqdm import tqdm

from stereo.data.dataset_utils import ViewDatasetIndex
from stereo.data.predump_utils import PreDumpIndex
from stereo.data.view_generator.view_generator import ViewGenerator

from stereo.prediction.vidar.stereo_predictor import StereoPredictor
from stereo.common.general_utils import tree_base

from file2path import file2path
# -


#const args
predump_dir = '/mobileye/algo_STEREO3/stereo/data/data_eng/'
predump = PreDumpIndex(predump_dir)
dataset_dir = '/mobileye/algo_STEREO3/stereo/data/canonical_examples'
dataSetIndex = ViewDatasetIndex(dataset_dir) 

# + tags=["parameters"]
# args to be feed by papermill
cam = 'main'
conf = 'diet_main_v2.3_reg_v0.0.2.0.json'
restore_iter = -1
clim_minZ = 4.
clim_maxZ = 40.
# -

# init objects
json_path = os.path.join(tree_base(), 'stereo', 'evaluation', 'canonical_examples.json')
with open(json_path) as f:
    canonical_examples_json = json.load(f)
canonical_examples = canonical_examples_json[cam]
conf = conf.split('/')[-1]
predictor = StereoPredictor(os.path.join(tree_base(), 'stereo', 'models', 'conf', conf), sector_name=cam, restore_iter=restore_iter)

#read views, and run predict
cntr_view = '%s_to_%s' %(cam,cam)
canonical_examples_data = {}
rebuild = False
for name, example in tqdm(canonical_examples.items()):
    try:
        views = dataSetIndex.read_views(frame='%s@gi@%07d' %(example['clip'], example['grab_index']), view_names=predictor.views_names)        
    except:
        rebuild = True
        print("couldn't find %s in dataset, extract from clip and update database" % name)
        vg = ViewGenerator(example['clip'], predictor.views_names, mode='pred', etc_dir=predump.etc_dir(example['clip']))
        views = vg.get_gi_views(example['grab_index'])
        for view_name, view in views.items():
            out_fn = "%s_%s_%07d.npz" % (example['clip'], view_name, example['grab_index'])
            try:
                out_path = file2path(out_fn, dataset_dir, create=True)
            except OSError as err:
                print (err)
            np.savez_compressed(out_path, **view)
    canonical_examples_data[name] =  {'image': views[cntr_view]['image'],
                                      'pred': np.squeeze(predictor.pred(views=views))}
if rebuild:
    ViewDatasetIndex(dataset_dir,index_path=None, rebuild_index=True, save_index=True) 

#display
clim = (1./clim_maxZ, 1./clim_minZ)
print("clim_minZ = %.1f, clim_maxZ = %.1f" % (clim_minZ, clim_maxZ))
for name in canonical_examples_data.keys():
    fig = plt.figure(figsize=(9,6))
    ax1 = fig.add_subplot(121, title=name)
    ax2 = fig.add_subplot(122, title='predict')
    plt.tight_layout()
    ax1.imshow(canonical_examples_data[name]['image'], origin='lower', cmap='gray')
#     ax1.imshow(1/canonical_examples_data[name]['lidar'], origin='lower', alpha=1.)
    ax2.imshow(canonical_examples_data[name]['pred'], origin='lower', cmap='jet', clim=clim)





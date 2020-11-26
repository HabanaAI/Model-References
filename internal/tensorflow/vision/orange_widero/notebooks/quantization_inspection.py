import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ### Retrieve and clean log

# +
# # !python /homes/davidn/gitlab/stereo/stereo/common/aws_logs.py jobs -n davidn -l 3

# +
inspection_dir = "/homes/davidn/gitlab/stereo/sandbox/dlo/vidar_main_v0.1.0_conf_dlo/q_inspection"
os.makedirs(inspection_dir, exist_ok=True)

quantization_job = "me-davidn-menta-Q-vidar-main-v0-1-0-c-dlo-0813-213710"
log_filename = os.path.join(inspection_dir, "cw_long.log")
# quantization_job = "me-davidn-menta-Q-vidar-main-v0-1-2-c-dlo-0806-212209"
# log_filename = "/homes/davidn/gitlab/stereo/sandbox/dlo/vidar_main_v0.1.0_conf_dlo/cw_short.log"
# -

# !python /homes/davidn/gitlab/stereo/stereo/common/aws_logs.py logs {quantization_job} \
#     -o {log_filename}

# +
def to_remove(line):
    patterns = [
        ".*Loaded .* map_func.*",
        ".*Using menta map_func.*",
        ".*Reading configuration JSON:.*",
        ".*num_gpus \(1 even if none\).*",
        ".*batch size: \d*, num_gpus: \d.*",
        ".*out keys:.*",
        ".*shape=.*dtype=.*",
        ".*TENSOR RENAMING:.*",
        ".*Using dataset weight factor array:.*",
        ".*Number of TFRecord files.*"
    ]
    for p in patterns:
#         print(p, re.match(p, line))
        if re.match(p, line) is not None:
            return True
    return False

def clean_line(line):
    # remove color codes:
    line = re.subn("(#033\[\d*m)|(#011)", "", line)[0]
    return line


# +
with open(log_filename, 'r') as f:
    q_log = f.readlines()

q_log = [clean_line(l) for l in q_log if not to_remove(l)]
print(len(q_log))

with open(log_filename, 'w') as f:
    f.writelines(q_log)
# -
# ### Parse log

FLOAT_PATTERN = "\d*\.\d+"
test_pattern = re.compile(f".*total loss = (?P<total_loss>{FLOAT_PATTERN}), "+
                          f"output_losses = {FLOAT_PATTERN:} "+
                          f"\['(?P<lidar_loss>{FLOAT_PATTERN})', '(?P<conf_loss>{FLOAT_PATTERN})'\]"+
                          f".*\(step: ((?P<layer>.*) (?P<pre_or_post>post|pre) train|(base))\).*$")

test_results = []
for line in q_log:
    res = test_pattern.match(line)
    if res:
        res_dict = res.groupdict()
        if res_dict['layer'] is not None:
            test_results.append(res_dict)
        else:
            base_result = res_dict


test_results = pd.DataFrame(test_results)
for c in test_results.columns:
    if "loss" in c:
        test_results[c] = test_results[c].astype(float)

# +
key = 'lidar_loss'

fig, ax = plt.subplots(figsize=(10,8), facecolor=(1, 1, 1))
for s in ['pre', 'post']:
# for s in ['post']:
    cond = test_results['pre_or_post'] == s
    ax.plot(range(1, np.count_nonzero(cond)+1),
            test_results[key][cond], '-o', label=s)
_ = ax.set_xticks(range(1, np.count_nonzero(cond)+1))
_ = ax.set_xticklabels(test_results['layer'][cond], rotation='vertical')

ax.plot([0], float(base_result[key]), '-o', label='base')
ax.set_title(f"Quantization {key} with pre/post train per layer")

ax.grid(which='both')
ax.legend()
ax.set_ylim([0, 0.5])
# -

# ### Predict

import os
import json
from stereo.data.dataset_utils import ViewDatasetIndex
from stereo.prediction.vidar.stereo_predictor import StereoPredictor
from stereo.common.s3_utils import my_listdir, my_open

# +
# quantization_path = "s3://mobileye-team-stereo/quantized_models/diet_main_v3.0.v0.0.12_conf_dlo/1/"+\
#                     "me-davidn-menta-Q-diet-main-v3-0-v0-0-12-c-test-dlo-0730-194431/raw"
quantization_path = "s3://mobileye-team-stereo/quantized_models/vidar_main_v0.1.0_conf_dlo/"+\
                    "me-davidn-menta-Q-vidar-main-v0-1-2-c-dlo-0807-055820/raw"

quantization_steps = [d for d in my_listdir(quantization_path) if d.startswith("vidar_quantization_")]
print(os.linesep.join(quantization_steps))
# -

[d for d in my_listdir(quantization_path+"/vidar_quantization_step_mono_l2_dcl2")]


# q_steps = ["vidar_quantization_init",
#            "vidar_quantization_preprocess",
#            "vidar_quantization_step_corr_l1_dcl1",
#            "vidar_quantization_step_corr_l2_upucl2"]
q_steps = ["vidar_quantization_step_corr_l2_up_ucl1"]

# +
with my_open(os.path.join(quantization_path, "params.json"), mode='r') as f:
    params = json.load(f)
model_name = os.path.basename(os.path.splitext(params['json_path'])[0])
model_conf_file = '../stereo/models/conf/stereo/' + model_name + '.json'

print("Loading model: {}".format(model_name))
predictors = []
for i, q_step in enumerate(q_steps):
    print(f"{i}/{len(q_steps)} | step: {q_step}")
    predictors.append(StereoPredictor(model_conf_file, 
                                      sector_name='main', 
                                      keras_json=os.path.join(quantization_path, q_step, "vidar.json")))
# -

from tensorflow.keras.utils import plot_model
for i, q_step in enumerate(q_steps):
    filename = os.path.join(inspection_dir,
                            f"{predictors[i].keras_model.name}_{q_step}.png")
    _ = plot_model(predictors[i].keras_model, show_shapes=True, show_layer_names=True,
                   to_file=filename, rankdir='TB')

# clip_name, gi = '18-10-22_11-20-37_Diego_Front_0053', 109802
# frame = '19-05-14_13-23-59_Alfred_Front_0015@gi@0032969'
# frame = '19-05-14_11-20-42_Alfred_Front_0052@gi@0102837'
frame = '19-05-14_10-58-28_Alfred_Front_0025@gi@0054449'
# frame = '19-05-14_11-03-06_Alfred_Front_0032@gi@0065745'
# frame = '18-12-26_11-57-36_Alfred_Front_0073@gi@0261453'
# frame = '19-05-14_15-40-30_Alfred_Front_0020@gi@0048435'
# frame = '19-05-14_15-40-30_Alfred_Front_0020@gi@0048435'
# frame = '19-05-14_13-38-48_Alfred_Front_0033@gi@0065133'

if 'frame' in locals():
    if frame.startswith('Town'):
        dataset_dir = '/mobileye/algo_STEREO3/stereo/data/simulator/v3/main'
    else:
        dataset_dir = '/mobileye/algo_STEREO3/stereo/data/view_dataset.v3.1'
    view_names = pred.views_names
    vdsi = ViewDatasetIndex(dataset_dir)
    views = vdsi.read_views(view_names, frame)
else:
    if 'vds_clip_name' not in locals() or vds_clip_name != clip_name:
        vds = ViewGenerator(clip_name, view_names=predictors[0].views_names, predump=None, mode='pred', 
                            no_labels=True, etc_dir=None)
        vds_clip_name = clip_name

    if 'views_gi' not in locals() or views_gi != gi:
        views = vds.get_gi_views(gi=gi)
        views_gi = gi

pred.keras_model.outputs

# +
# extra_tensor_names = ['arch/corr_scores_l2/Maximum_1'] if len(predictors[0].views_names) == 4 else ['arch/corr_scores_l2/Maximum']
extra_tensor_names = ['corr_scores_l2/Maximum']
extra_tensor_names.append('out_conf')

outs = []
confs = []
corrs = []
out, extra_tensors = pred.pred(views=views,
                               extra_tensor_names=extra_tensor_names)
if isinstance(out, dict):
    out = out['out']
corr_scores_l2_max = extra_tensors[0]
error_map = extra_tensors[-1]


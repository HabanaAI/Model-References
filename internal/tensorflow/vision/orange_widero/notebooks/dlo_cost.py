import os
import json
import tensorflow as tf
import sys
# from stereo.common.s3_utils import my_open
from stereo.models.menta.menta_utils import generate_menta_checkpoint

# +
if "DLO_IMPORT_PATH" not in os.environ:
    os.environ["DLO_IMPORT_PATH"] = "/mobileye/shared/DLO/last/dl/dl-optimizer"
sys.path.insert(0, os.environ["DLO_IMPORT_PATH"])
# os.environ['DL_COMPILER'] = "/homes/shirsh/for_vidar/dl/dl-compiler/"

from dl_sdk import KerasDeployment, DLConfigType
from dl_sdk.settings import QuantizationSettings
# -

# model_name = "vidar_main_v0.1.0_conf_dlo"
model_name = "vidar_main_v0.2.2_conf_dlo_add"
# model_name = "vidar_main_v0.2.2_conf_dlo_conv"

# Generate a menta model. It might take a while if there model wasn't already generated...
s3_menta_model = os.path.join("s3://mobileye-team-stereo", "menta_models", model_name)
# _ = generate_menta_checkpoint(model_name,
#                                         s3_dst=s3_menta_model,
#                                         use_existing_model=True)

# +
# OUTPUT_PATH = f"/mobileye/algo_STEREO3/davidn/dlo/{model_name}"
OUTPUT_PATH = f"/homes/davidn/gitlab/stereo/sandbox/dlo/{model_name}"
JSON_FNAME = os.path.join(s3_menta_model, "keras.json")

assert OUTPUT_PATH is not None and JSON_FNAME is not None, "Please set OUTPUT_PATH and JSON_FNAME"


# +
def my_open(name, *args, **kwargs):
    if name.startswith("s3"):
        import s3fs
        fs = s3fs.S3FileSystem()
        return fs.open(name, *args, **kwargs)
    else:
        return open(name, *args, **kwargs)

with my_open(JSON_FNAME, mode='r') as f:
    model_config = json.load(f)


# +
# remove clamp layer (assuming there's only one)
clamp_idx = None
clamp_name = None
clamp_inboud = None
for i, l in enumerate(model_config['config']['layers']):
    if l['class_name'] == "Clamp":
        clamp_name = l['name']
        clamp_idx = i
        clamp_inboud = l['inbound_nodes'][0][0][0]
    if clamp_idx and l['inbound_nodes'] and l['inbound_nodes'][0][0][0] == clamp_name:
        clamp_outbound = l['name']
        l['inbound_nodes'][0][0][0] = clamp_inboud
        break

if clamp_idx:
    model_config['config']['layers'].pop(clamp_idx)
    print("Removed clamp:", clamp_inboud, "->", clamp_outbound)
# -

custom_objects = KerasDeployment.get_custom_objects()
keras_model = tf.keras.models.model_from_config(config=model_config,
                                                custom_objects=custom_objects)

# +
quantization_settings = {
    QuantizationSettings.SETTING_NAME: {
        QuantizationSettings.Keys.DRY_RUN: True,
        "TranslateFromCalibration": {
            "is_float": True
        }
    }
}

keras_deployment = KerasDeployment()
q_model = keras_deployment.quantize(model=keras_model,
                          settings=quantization_settings,
                          output_path=OUTPUT_PATH)

# +
# q_model_dir = os.path.join(OUTPUT_PATH, keras_model.name+"_quantization_final_dry_run")
# with open(os.path.join(q_model_dir, keras_model.name+'.json'), 'r') as f:
#     q_model = tf.keras.models.model_from_json(f.read(),
#                                               custom_objects=custom_objects)
# q_model.load_weights(os.path.join(q_model_dir, keras_model.name+'.h5'))
# -

print(q_model.output_names)
print(len(q_model.layers)-len(keras_model.layers))

# +
from dlo_internal.keras_impl.keras_split_model import KerasSplitModel
splitter = KerasSplitModel(q_model)

corr_features = splitter.split(['I_srnd_0'], ['corr_l2_up_ucl2'],
                             output_path=OUTPUT_PATH, sub_model_name='corr_features')

tfc_layers = ["translate_from_calibration",
              "translate_from_calibration_1",
              "translate_from_calibration_2"]
tfc_out_layers = [n.outbound_layer.name for tfc in tfc_layers
                  for n in q_model.get_layer(tfc)._outbound_nodes]
tfc_out_layers = list(set(tfc_out_layers))
mono_unet = splitter.split(['I_cntr', 'corr_l2_up_ucl2'] + tfc_out_layers,
                           q_model.output_names,
                           output_path=OUTPUT_PATH, sub_model_name='mono_unet')
# mono_unet = splitter.split(['I_cntr', 'T_cntr_srnd', 'focal', 'origin_l2', 'corr_l2_upucl2'],
#                             q_model.output_names,
#                            output_path=OUTPUT_PATH, sub_model_name='mono_unet')
submodels = {'corr_features': corr_features,
             'mono_unet': mono_unet}
# -


for name in submodels.keys():
    with open(os.path.join(OUTPUT_PATH, "submodel", name+'.json'), 'r') as f:
        submodels[name] = tf.keras.models.model_from_json(f.read(),
                                                          custom_objects=custom_objects)
    submodels[name].load_weights(os.path.join(OUTPUT_PATH, "submodel", name+'.h5'))
submodels

# +
model = keras_model
filename = os.path.join(OUTPUT_PATH, model.name+'.png')
_ = tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file=filename, rankdir='TB')

# model = q_model
# filename = os.path.join(OUTPUT_PATH, model.name+'_q.png')
# _ = tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file=filename, rankdir='TB')

# model = submodels["corr_features"]
# filename = os.path.join(OUTPUT_PATH, "submodel", model.name+'.png')
# _ = tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file=filename, rankdir='TB')

# model = submodels["mono_unet"]
# filename = os.path.join(OUTPUT_PATH, "submodel", model.name+'.png')
# _ = tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file=filename, rankdir='TB')


# +
settings = {
    'deploy_settings': {
        'input_data_type': 'I8'
    },
    'gc_settings': {
        'batch_size': 1,
        'batch_scattered': True
    }
}

keras_deployment = KerasDeployment()
for name in submodels.keys():
    if name == "corr_features":
        settings['gc_settings']['batch_size'] = 1
    else:
        settings['gc_settings']['batch_size'] = 1
    print("Cost for sub model: {}".format(name))
    keras_deployment.cost(
        model=submodels[name],
        custom_objects=custom_objects,
        ir_path=None,
        platform_configuration=DLConfigType.EYEQ5_MID,
        output_path=os.path.join(OUTPUT_PATH, "submodel", name),
        do_dry_run_on_model=False,
        settings=settings,
        log_level='DEBUG')


# # # Calculating cost for sub-model #1 (a.k.a. 'shared' part)
# keras_deployment.cost(
#     model=sub_model1,
#     custom_objects=custom_objects,
#     ir_path=None,
#     platform_configuration=DLConfigType.EYEQ5_MID,
#     output_path=os.path.join(OUTPUT_PATH, 'submodel1'),
#     do_dry_run_on_model=False,
#     settings=deploy_settings,
#     log_level='DEBUG')


# # Calculating cost for sub-model #2
# keras_deployment.cost(
#     model=sub_model2,
#     custom_objects=custom_objects,
#     ir_path=None,
#     platform_configuration=DLConfigType.EYEQ5_MID,
#     output_path=os.path.join(OUTPUT_PATH, 'submodel2'),
#     do_dry_run_on_model=False,
#     settings=deploy_settings,
#     log_level='DEBUG')

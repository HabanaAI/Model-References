import numpy as np
import tensorflow as tf
import os
import json
from importlib.machinery import SourceFileLoader

# stereo imports
from stereo.common.general_utils import tree_base
from stereo.models.model_utils import keras_warm_start
from stereo.interfaces.implements import load_format, get_arch_format, get_loss_format
from stereo.models.sm.create_dataset import create_dataset, ds_spec_from_ds

# menta imports
from menta.src import models, optimizers
from menta.src.data import RecordTemplate


# def read_ds_spec_from_json(filename):
#     with open(filename, 'r') as f:
#         ds_dict = json.load(f)
#     TensorSpec = tf.TensorSpec if hasattr(tf, "TensorSpec") else tf.contrib.framework.TensorSpec
#     ds_spec = {k: TensorSpec(shape=ds_dict[k].shape,
#                              dtype=ds_dict[k].dtype,
#                              name=k) for k in ds_dict.keys()}
#     return ds_spec
#
#
# def write_ds_spec_to_json(ds_spec, filename):
#     json_output = []
#     json_output.append("{")
#     keys_max_length = max((len(k) for k in ds_spec.keys()))
#     for k in ds_spec.keys():
#         if len(json_output) > 1:
#             json_output[-1] += ","
#         key_prop = {"dtype": ds_spec[k].dtype.name,
#                     "shape": ds_spec[k].shape.as_list()}
#         json_output.append(
#             "    {key:{width}}: {json_content}".format(key='"{}"'.format(k), width=keys_max_length + 2,
#                                                        json_content=json.dumps(key_prop))
#         )
#     json_output.append("}")
#
#     with open(filename, 'w') as f:
#         f.write(os.linesep.join(json_output))
#         print("Dataset spec was written to: {}".format(filename))
#     print(os.linesep.join(json_output))


class VIDAR(models.MentaModel):

    def __init__(self, *args, **kwargs):
        super(VIDAR, self).__init__(*args, **kwargs)
        # self.model_name = self.model_name + "_" + os.path.splitext(os.path.basename(
        #     self.user_params["json_path"]))[0]

        if not self.user_params["local"]:
            from stereo.models.sm.sm_setup import sm_setup
            sm_setup()

            num_gpus = int(os.getenv('SM_NUM_GPUS', 1))
            if num_gpus > 1:
                batch_size = self.user_params["batch_size"]
                self.user_params["batch_size"] = batch_size * num_gpus
                print("Updating batch size to {}={}*{}".format(self.user_params["batch_size"], batch_size, num_gpus))
            else:
                print("batch size: {}, num_gpus: {}".format(self.user_params["batch_size"], num_gpus))


        json_path = os.path.join(tree_base(), self.user_params["json_path"])
        with open(json_path, 'rb') as f:
            print("Reading configuration JSON: {}".format(json_path))
            self.conf = json.load(f)

        self.model_params = self.conf['model_params']
        self.data_params = self.conf['data_params']
        arch_module = SourceFileLoader('arch', os.path.join(tree_base(), 'stereo', 'models', 'arch',
                                                            self.model_params['arch']['name'] + '.py')).load_module()
        loss_module = SourceFileLoader('loss', os.path.join(tree_base(), 'stereo', 'models', 'loss',
                                                            self.model_params['loss']['name'] + '.py')).load_module()
        self.arch_func_ = arch_module.arch
        self.loss_func_ = loss_module.loss

        # TODO: source files are loaded too many times in this function...
        self.arch_arg_names = load_format('arch', get_arch_format(self.model_params), model_params=self.model_params)
        self.loss_arg_names = load_format('loss', get_loss_format(self.model_params), model_params=self.model_params)

        self.train_ds = None
        self.test_ds = None
        self.ds_spec = None
        self.with_name_scopes = True
        self.output_names = None

    def init_record_templates(self):
        if self.ds_spec is None:
            self.init_datasets()
            # self.ds_spec = read_ds_spec_from_json(os.path.join(self._output_base_path, "ds_spec.json"))

        record_templates = []
        for name, spec in self.ds_spec.items():
            # fill shape with None's to get a general NDHWC format, without the batch dimension.
            # RecordTemplate ignores the None dimensions.
            shape = [None] * (4 - spec.shape.ndims) + spec.shape.as_list()
            depth, height, width, channels = shape
            record_templates.append(
                RecordTemplate(name=name,
                               depth=depth, height=height, width=width, channels=channels,
                               dtype=spec.dtype.as_numpy_dtype)
            )
        return record_templates

    def _load_weights(self):
        super(VIDAR, self)._load_weights()
        if len(os.listdir(self._output_checkpoints)) == 0:
            # warm start only if there are no checkpoints for this model
            warm_start_conf = self.model_params['init'].get('warm_start', None)
            if self.user_params['menta_from_ckpt']:
                # warm start from the Estimator training of this model
                warm_start_conf = {
                    "model_name": os.path.splitext(os.path.basename(self.user_params["json_path"]))[0]
                }
            if warm_start_conf:
                warm_start_conf['ckpt_dir'] = os.path.join(*self.conf['model_base_path'].split('/')[3:],
                                                           warm_start_conf['model_name'])
                warm_start_conf['model_prefix'] = "vidar/model/"
                restore_iter = keras_warm_start(self._keras_model, warm_start_conf)
                self._keras_model.save_weights(os.path.join(self._output_base_path, "warm_start_weights.h5"))
                self._keras_model.save_weights(os.path.join(self._output_base_path, self._output_checkpoints,
                                                            f"{self._keras_model.name}_weights.{restore_iter}.h5"))

    def _create_dataset(self, channel, pipe_mode, batch_sz, shuffle, epochs):
        ds_list = []
        ds_weights = []
        weighted_sampling_after_batching = self.data_params.get('weighted_sampling_after_batching', False)
        prefetch_buffer = tf.data.experimental.AUTOTUNE \
            if int(tf.version.VERSION.split('.')[1]) > 12 else 4

        for section in self.data_params['datasets'].keys():
            channel_k = '%s_%s' % (channel, section) if pipe_mode else channel
            augm = (channel == 'train') and ('augmentation' in self.data_params.keys())
            ds = create_dataset(self.conf, section, channel=channel_k, pipe_mode=pipe_mode, augm=augm,
                                run_in_menta=True)
            if shuffle:
                ds = ds.shuffle(buffer_size=100 * 10, reshuffle_each_iteration=True,
                                seed=self.data_params.get('random_seed', None))
            ds = ds.prefetch(buffer_size=prefetch_buffer)
            ds = ds.repeat(epochs)
            if weighted_sampling_after_batching:
                ds = ds.batch(batch_sz, drop_remainder=True)
            ds_list.append(ds)
            if self.data_params.get('weights', False):
                if channel in self.data_params['weights'].keys():
                    w = self.data_params['weights'][channel][section]
                else:
                    w = self.data_params['weights'][section]
                ds_weights.append(w)

        if ds_weights:
            ds_weights = np.array(ds_weights)
            ds_weights = (ds_weights / np.sum(ds_weights)).tolist()
            print("Using dataset weight factor array: ", ds_weights)
        else:  # when no weights in the conf, uniform sampling of ds_list performed
            ds_weights = None

        ds = tf.data.experimental.sample_from_datasets(ds_list, ds_weights)
        if not weighted_sampling_after_batching:
            ds = ds.batch(batch_sz, drop_remainder=True)

        ds = ds.prefetch(buffer_size=prefetch_buffer)
        return ds

    def init_datasets(self):
        self.train_ds = self._create_dataset(channel='train', pipe_mode=self.user_params["pipe_mode"],
                                                 batch_sz=self.user_params["batch_size"],
                                                 shuffle=not self.user_params["local"], epochs=self.user_params["num_epochs"])
        self.test_ds = self._create_dataset(channel='test', pipe_mode=self.user_params["pipe_mode"],
                                            batch_sz=self.user_params["batch_size"],
                                            shuffle=not self.user_params["local"], epochs=1)
        if self.ds_spec is None:
            self.ds_spec = ds_spec_from_ds(self.train_ds)

        return self.train_ds, self.test_ds

    def init_model(self):
        arch_inputs = {name: self.inputs[name] for name in self.arch_arg_names}
        # with tf.name_scope("arch"):
        out = self.arch_func_(**arch_inputs,
                              **self.conf['model_params']['arch']['kwargs'])
        if isinstance(out, dict):
            self.output_names = list(out.keys())
            out_list = [out[k] for k in self.output_names]
        else:
            out_list = [out]
        return out_list

    def init_losses(self):
        loss_inputs = {name: self.inputs[name] for name in self.loss_arg_names}
        # with tf.name_scope("loss"):
        model_output = {n: t for n, t in
                        zip(self.output_names, self._keras_model.outputs)}
        self.losses["loss"] = tf.identity(
            self.loss_func_(model_output,
                            **loss_inputs,
                            **self.conf['model_params']['loss']['kwargs'],
                            keras_train=True),
            name="loss")

    # def init_learning_rate(self):
    #     # TODO: this is an opportunity to make a better learning rate schedule
    #     return learning_rates.ExponentialDecay(base_lr=0.025, gamma=0.99999651, restart=1000002)

    def init_optimizer(self):
        learning_rate = tf.Variable(self.model_params['learning_rate'], trainable=False, name='learning_rate')
        if self.model_params['train_op'] == "AdamOptimizer":
            opt = optimizers.Adam(lr=learning_rate)
        elif self.model_params['train_op'] == "GradientDescentOptimizer":
            opt = optimizers.SGD(lr=learning_rate)
        else:
            raise ValueError("Unsupported optimizer")
        return opt

    def _rename_tensor(self, name):
        """
        Rename the tensor name to correspond to menta name scopes
        """
        prev_name = name
        if self.with_name_scopes:
            name = name.replace("loss/", "vidar/losses/")\
                       .replace("arch/", "vidar/model/")\
                       .replace("features/", "vidar/inputs/")
        else:
            name = name.replace("loss/", "")\
                       .replace("arch/", "")\
                       .replace("features/", "")
        print("TENSOR RENAMING: {} -> {}".format(prev_name, name))
        return name

    def init_metrics(self):
        for s in self.conf['model_params'].get('scalars_to_log', []):
            if "loss_reg" in s:
                self.metrics[s+"_m"] = sum([loss for loss in self._keras_model.losses
                                       if "regularizer" in loss.name.lower()])
            else:
                s_renamed = self._rename_tensor(s)
                self.metrics[s+"_m"] = tf.compat.v1.get_default_graph().get_tensor_by_name(s_renamed + ':0')

    def init_summaries(self):
        model_params = self.conf['model_params']

        # add tensorboard summaries and merge them
        summary_scalars = []
        summary_images = []
        summary_strings = []
        summary_histograms = []

        for s in model_params.get('scalars_to_summarize', []):
            s_renamed = self._rename_tensor(s)
            summary_scalars.append(tf.compat.v1.summary.scalar(s,
                                                     tf.compat.v1.get_default_graph().get_tensor_by_name(s_renamed + ':0')))

        for s in model_params.get('images_to_summarize', []):
            s_renamed = self._rename_tensor(s)
            im = tf.compat.v1.get_default_graph().get_tensor_by_name(s_renamed + ':0')
            im = im if len(im.shape.as_list()) == 4 else tf.expand_dims(im, -1)
            im = tf.reverse(im, axis=[1]) if model_params.get('reverse_image_summaries', False) else im
            summary_images.append(tf.compat.v1.summary.image(s, im, max_outputs=1))

        for s in model_params.get('strings_to_summarize', []):
            s_renamed = self._rename_tensor(s)
            str_ = tf.compat.v1.get_default_graph().get_tensor_by_name(s_renamed + ':0')[0, ...]
            if str_.dtype == tf.float32:
                str_ = tf.compat.v1.py_func(lambda val: val.tobytes().decode().strip('0'), [str_], tf.string)
            summary_strings.append(tf.compat.v1.summary.text(s, str_))

        for s in model_params.get('histograms_to_summarize', []):
            s_renamed = self._rename_tensor(s)
            hist = tf.compat.v1.get_default_graph().get_tensor_by_name(s_renamed + ':0')
            summary_histograms.append(tf.compat.v1.summary.histogram(s, hist))

    def init_callbacks(self, *args, **kwargs):
        pass

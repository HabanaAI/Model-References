import json
import os
import tensorflow as tf
import imp

from stereo.models.sm.sm_utils import get_checkpoint_path
from stereo.common.general_utils import tree_base
from stereo.models.model_utils import get_cached_model, to_batch, get_model_graph_path, pred_keras
from stereo.models.model_utils import load_model_from_meta, load_model_from_mvs_model, load_keras_model


class Predictor(object):
    def __init__(self, model_json_path, checkpoint_path=None, restore_iter=-1,
                 single_core=False, from_meta=True, load_loss=False, **kwargs):
        """
        :param model_json_path: path to JSON file defining the trained model
        :param checkpoint_path: take model from here
        :param restore_iter: set this to a particular restore iter (otherwise restore best iter)
        """
        with open(model_json_path, 'rb') as fp:
            self.model_conf = json.load(fp)
        model_name = os.path.splitext(os.path.split(model_json_path)[1])[0]

        # load this model's prep function and view_names
        self.dataset_attributes = self.get_dataset_attributes(**kwargs)
        print(self.dataset_attributes)
        self.prep_ = imp.load_source('prep', os.path.join(tree_base(), 'stereo', 'data', 'prep',
                                                          self.dataset_attributes['prep'] + '.py')).prep_frame

        # load map_funcs
        self._arch_map_func, self._loss_map_func = self.get_map_funcs(load_loss, **kwargs)

        if 'keras_json' in kwargs:
            print("Loading keras model...")
            keras_json = kwargs['keras_json']
            self.keras_model = load_keras_model(keras_json, load_weights=True)
        else:
            # figure out the self.checkpoint_dir on s3
            model_s3 = '/'.join(self.model_conf['model_base_path'].split('/')[3:])
            self.checkpoint_dir = '/'.join([model_s3, model_name])

            # initialize session
            tf.compat.v1.reset_default_graph()
            if single_core:
                session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
                self.sess = tf.compat.v1.Session(config=session_conf)
            else:
                self.sess = tf.compat.v1.Session()

            # get a cached checkpoint path (may be slower when cache downloads)
            if checkpoint_path is None:
                checkpoint_path = get_checkpoint_path(self.checkpoint_dir, restore_iter)
                self.checkpoint_path = get_cached_model(checkpoint_path)
            else:
                self.checkpoint_path = checkpoint_path

            # load model graph
            if from_meta:
                model_graph_path = get_model_graph_path(model_json_path, use_cache=True)
                self.out, self.placeholders = load_model_from_meta(model_graph_path=model_graph_path)
            else:
                self.out, self.placeholders = load_model_from_mvs_model(model_json_path, load_loss=load_loss)

            # load weights from checkpoint
            saver = tf.compat.v1.train.Saver()
            print('Restoring weights from %s' % self.checkpoint_path)
            saver.restore(self.sess, self.checkpoint_path)

    def get_dataset_attributes(self, **kwargs):
        raise NotImplementedError

    def get_map_funcs(self, load_loss, **kwargs):
        raise NotImplementedError

    def run_prep_func(self, **kwargs):
        raise NotImplementedError

    def pred(self, extra_tensor_names=None, **kwargs):
        """ Return the prediction of the initialized model on the provided views.


        views -- A dict of views that provide the input for the model.
        extra_tensor_names -- Either None, in which case only the model's "out" is predicted and returned as an np
            array, or a list of tf session tensor names, in which case a list of np arrays is returned with "out" at
            index 0 and the evaluation of the extra_tensor_names follows in the list in order provided.
        """
        features_list = self.run_prep_func(**kwargs)
        if self._arch_map_func is not None:
            features_list = [self._arch_map_func(features) for features in features_list]
        if self._loss_map_func is not None:
            features_list = [self._loss_map_func(features) for features in features_list]
        placeholders = to_batch(features_list)

        if hasattr(self, 'keras_model'):
            out = pred_keras(keras_model=self.keras_model,
                             inputs=placeholders,
                             extra_tensor_names=extra_tensor_names)
        else:
            feed_dict = {}
            for k, v in self.placeholders.items():
                if k in placeholders.keys():
                    feed_dict[v] = placeholders[k]

            all_tensors = [self.out]
            if extra_tensor_names is not None:
                for tensor_name in extra_tensor_names:
                    all_tensors.append(self.sess.graph.get_tensor_by_name(tensor_name + ':0'))
            out = self.sess.run(all_tensors, feed_dict=feed_dict)

        if extra_tensor_names is None:
            return out[0]
        else:
            return out[0], out[1:]

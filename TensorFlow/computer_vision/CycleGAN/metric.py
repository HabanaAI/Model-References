###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import _pickle as pkl
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.inception_v3 import preprocess_input

from data import TrasformInputs, normalize_img, denormalize_img


class GeneratorToInceptionFeats(keras.Model):
    def __init__(self, gen, gen_post_process=None, extractor=None, extractor_preprocess=None):
        self.extractor = extractor or keras.applications.InceptionV3(
            False, pooling='avg')
        self.gen = gen
        self.gen_post_process = gen_post_process or denormalize_img
        self.extractor_preprocess = extractor_preprocess or TrasformInputs().preprocess_test_image

    @tf.function
    def call(self, x):
        return self.extractor(self.extractor_preprocess(self.gen_post_process(self.gen(x))))


def fid_from_stats(ref_mean, ref_cov, fake_mean, fake_cov):
    return keras.losses.mean_squared_error(ref_mean, fake_mean) + tf.linalg.trace(ref_cov) + \
        tf.linalg.trace(
            fake_cov) - 2*tf.linalg.trace(tf.linalg.sqrtm(tf.matmul(ref_cov, fake_cov)))


class FID(object):
    def __init__(self, real_ds_dict, extractor=None, preprocess=None, ref_stats_dict=None):
        self.extractor = extractor or keras.applications.InceptionV3(
            False, pooling='avg')
        self.preprocess = preprocess or TrasformInputs().preprocess_train_image
        if ref_stats_dict:
            self.ref_stats_dict = ref_stats_dict
        else:
            self.ref_stats_dict = {}
            self._extract_ref_stats(real_ds_dict)

    def _extract_stats(self, ds, batch_size=128, num_batch=64, extractor=None, preprocess=None):
        extractor = extractor or self.extractor
        preprocess = preprocess or self.preprocess
        ds = ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat(
        ).batch(batch_size)
        acts = extractor.predict(ds, batch_size, 1, num_batch)
        mean = tf.reduce_mean(acts, 0, True)
        acts_c = acts - mean
        cov = tf.matmul(acts_c, acts_c, True) / \
            tf.cast(tf.shape(acts_c)[0], tf.float32)
        return mean, cov

    def _extract_ref_stats(self, real_ds, update=False, save_path='reference_stats_dict.pkl'):
        if type(real_ds) == dict:
            for k, ds in real_ds.items():
                if k not in self.ref_stats_dict or update:
                    self.ref_stats_dict[k] = self._extract_stats(ds)
        if save_path:
            with open(save_path, 'wb') as f:
                pkl.dump(self.ref_stats_dict, f)

    def fid_from_generator(self, generator, feed_ds, target_ds_names, generator_post_process, generator_pre_process):
        metric = {}
        for k in target_ds_names:
            ref_mean, ref_cov = self.ref_stats_dict[k]
            fake_mean, fake_cov = self._extract_stats(feed_ds, extractor=generator,
                                                      generator_post_process=generator_post_process,
                                                      generator_pre_process=generator_pre_process)

            metric[k] = self.fid_from_stats(
                ref_mean, ref_cov, fake_mean, fake_cov)

        return metric

    def fid_from_ds(self, fake_ds_dict, real_ds_dict=None):
        if real_ds_dict:
            self._extract_ref_stats(real_ds_dict, True)

        metric = {}
        for k, fake_ds in fake_ds_dict.items():
            ref_mean, ref_cov = self.ref_stats_dict[k]
            fake_mean, fake_cov = self._extract_stats(fake_ds)
            metric[k] = self.fid_from_stats(
                ref_mean, ref_cov, fake_mean, fake_cov)

        return metric


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras
    import tensorflow_datasets as tfds

    """
    ## Create `Dataset` objects
    """
    tfds.disable_progress_bar()

    # Load the horse-zebra dataset using tensorflow-datasets.
    dataset, _ = tfds.load("cycle_gan/horse2zebra",
                           with_info=True, as_supervised=True, download=True)
    train_horses, train_zebras = dataset["trainA"], dataset["trainB"]
    test_horses, test_zebras = dataset["testA"], dataset["testB"]
    fid = FID({'F': train_horses, 'G': train_horses})

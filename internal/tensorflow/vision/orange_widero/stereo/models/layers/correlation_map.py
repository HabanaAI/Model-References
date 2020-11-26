import tensorflow as tf
import numpy as np
# from menta.src.utils import export_api, ExportTypes

# @export_api(collection=ExportTypes.LAYER)
class Correlation(tf.keras.layers.Layer):
    def __init__(self, steps, interpolation='BILINEAR', name="corr_map", **kwargs):
        self.steps = steps
        self.interpolation = interpolation

        super(Correlation, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]
        origin = inputs[2]
        focal = inputs[3]
        T12 = inputs[4]

        corr_tensors = []
        if type(self.steps) is list or type(self.steps) is np.ndarray:
            n_steps = len(self.steps)
        else:
            n_steps = self.steps.shape.as_list()[1]
        for i in np.arange(n_steps):
            if type(self.steps) is list or type(self.steps) is np.ndarray:
                d = self.steps[i]
                if d == 0:
                    D = 0
                else:
                    D = ((d ** -1) + T12[:, 2]) ** -1
            else:
                D = tf.compat.v1.where(self.steps[:, i] > 0,
                             tf.divide(tf.ones_like(T12[:, 2]),
                                       tf.add(tf.multiply(self.steps[:, i] ** -1, tf.ones_like(T12[:, 2])), T12[:, 2])),
                             tf.zeros_like(T12[:, 2])
                             )
            alpha = 1 - (D * T12[:, 2])
            beta = D * T12[:, 2] * origin[:, 0] + D * focal[:, 0] * T12[:, 0]
            gamma = D * T12[:, 2] * origin[:, 1] + D * focal[:, 1] * T12[:, 1]

            transforms = tf.transpose(a=tf.stack([alpha, tf.zeros(tf.shape(input=alpha)), beta,
                                                tf.zeros(tf.shape(input=alpha)), alpha, gamma,
                                                tf.zeros(tf.shape(input=alpha)), tf.zeros(tf.shape(input=alpha))]),
                                      perm=[1, 0])

            shifted = tf.contrib.image.transform(images=x,
                                                 transforms=transforms,
                                                 interpolation=self.interpolation,
                                                 name="corr_transform")

            corr = tf.reduce_mean(input_tensor=tf.multiply(shifted, y), axis=3)
            corr_tensors.append(corr)

        return tf.transpose(a=tf.stack(corr_tensors), perm=[1, 2, 3, 0])

    def get_config(self):
        config = super(Correlation, self).get_config()
        config.update({
            'interpolation': self.interpolation,
            'steps': self.steps
        })
        return config


def correlation_map(x, y, origin, focal, T12, steps, interpolation='BILINEAR', name="corr_map"):
    return Correlation(steps, interpolation, name)([x, y, origin, focal, T12])

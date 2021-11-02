###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

from data import TrasformInputs
from matplotlib import pyplot as plt
import io
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


"""
## Create a callback that periodically saves generated images
"""


class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, file_writer, test_ds_F, test_ds_G, num_img=4, postprocessor=None, freq=1):
        self.num_img = num_img
        self.freq = freq
        self.file_writer = file_writer
        self.transform = postprocessor or TrasformInputs()
        self.test_ds_F = (
            test_ds_F.map(self.transform.preprocess_test_image,
                          num_parallel_calls=None)
        )
        self.test_ds_G = (
            test_ds_G.map(self.transform.preprocess_test_image,
                          num_parallel_calls=None)
        )
        self._fid = None

    def _visualize(self, real_imgs, generator):
        fakes = self.transform.denormalizer(
            generator.predict(real_imgs)).numpy()
        imgs = self.transform.denormalizer(real_imgs).numpy()

        fig, ax = plt.subplots(self.num_img, 2, figsize=(12, 15))
        for i, (img, prediction) in enumerate(zip(imgs, fakes)):
            ax[i, 0].imshow(img)
            ax[i, 1].imshow(prediction)
            ax[i, 0].set_title(f"Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")
        fig.tight_layout()
        return fig

    def _calc_fid(self):
        return 0

    def on_epoch_end(self, epoch, logs=None):
        def _generator_log(name, ds, gen):
            if self._fid:
                fid = self._calc_fid(name, ds, gen)
            fig_t = self._visualize(
                next(iter(ds.take(self.num_img).batch(self.num_img))), gen)
            with self.file_writer.as_default():
                tf.summary.image(name, plot_to_image(fig_t), step=epoch)
                if self._fid:
                    tf.summary.scalar(f'FID {name}', fid, epoch)

        if epoch % self.freq == 0:
            _generator_log('Test Y->X', self.test_ds_G, self.model.gen_X)
            _generator_log('Test Y->Y', self.test_ds_G, self.model.gen_Y)
            _generator_log('Test X->Y', self.test_ds_F, self.model.gen_Y)
            _generator_log('Test X->X', self.test_ds_F, self.model.gen_X)

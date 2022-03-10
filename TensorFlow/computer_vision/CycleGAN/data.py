###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import tensorflow as tf


def normalize_img(img):
    img = tf.clip_by_value(img, 0, 255)
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0


def denormalize_img(img):
    img = tf.cast(img * 127.5 + 127.5, tf.uint8)
    return img


class TrasformInputs():
    def __init__(self, orig_img_size=(286, 286), input_img_size=(256, 256)):
        self.orig_img_size, self.input_img_size = orig_img_size, input_img_size
        self.normalizer = normalize_img
        self.denormalizer = denormalize_img

    def preprocess_train_image(self, img, label):
        # Random flip
        img = tf.image.random_flip_left_right(img)
        # Resize to the original size first
        img = tf.image.resize(img, self.orig_img_size)
        # Random crop to 256X256
        img = tf.image.random_crop(img, size=self.input_img_size+(3,))
        # Normalize the pixel values in the range [-1, 1]
        img = self.normalizer(img)
        return img

    def preprocess_test_image(self, img, label):
        # Only resizing and normalization for the test images.
        img = tf.image.resize(img, self.input_img_size)
        img = self.normalizer(img)
        return img

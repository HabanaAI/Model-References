"""utils.py

Commond utils functions for TensorFlow & Keras.
"""


import tensorflow as tf


def config_keras_backend_for_gpu():
    """Config tensorflow backend to use less GPU memory."""
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)


def clear_keras_session():
    """Clear keras session.

    This is for avoiding the problem of: 'Exception ignored in: <bound method BaseSession.__del__ of <tensorflow.python.client.session.Session object ...'
    """
    tf.keras.backend.clear_session()

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np
from stereo.models.layers.correlation_layer import Correlation

tf.compat.v1.disable_eager_execution()

if __name__ == '__main__':
    NumExamples = 100
    B = 2
    H = 3
    W = 4
    C = 20
    M = 10


    cntr = keras.Input(name="cntr", shape=(H, W, C))
    srnd = keras.Input(name="srnd", shape=(H, W, C))
    # scales = tf.Variable(initial_value=tf.zeros((1, M)),
    #                      trainable=True, name="scales")
    # trans_x = tf.Variable(initial_value=tf.zeros((1, M)),
    #                       trainable=True, name="trans_x")
    # trans_y = tf.Variable(initial_value=tf.zeros((1, M)),
    #                       trainable=True, name="trans_y")
    inputs = [cntr, srnd]#, scales, trans_x, trans_y]
    outputs = Correlation(M)(inputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.SGD(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.BinaryCrossentropy(),
        # List of metrics to monitor
        metrics=['accuracy'],
    )

    cntr_train = np.random.uniform(0, 100, (NumExamples * B, H, W, C))
    srnd_train = np.random.uniform(0, 100, (NumExamples * B, H, W, C))
    output_train = np.random.uniform(0, 100, (NumExamples * B, H, W, M))

    cntr_val = cntr_train[-10:]
    srnd_val = srnd_train[-10:]
    output_val = output_train[-10:]

    cntr_train = cntr_train[:-10]
    srnd_train = srnd_train[:-10]
    output_train = output_train[:-10]



    history = model.fit(
        [cntr_train, srnd_train],
        output_train,
        batch_size=2,
        epochs=1,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=([cntr_val, srnd_val], output_val),
    )

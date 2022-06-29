import tensorflow as tf

import horovod.tensorflow.keras as hvd
hvd.init()

# Ensure only 1 process downloads the data on each node
if hvd.local_rank() == 0:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    hvd.broadcast(0, 0)
else:
    hvd.broadcast(0, 0)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Data partition for different workers
num_pics_per_rank = x_train.shape[0] // hvd.size()
pic_begin = num_pics_per_rank * hvd.rank()
pic_end = pic_begin + num_pics_per_rank
x_train = x_train[pic_begin:pic_end, ]
y_train = y_train[pic_begin:pic_end, ]

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10),
])

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Using hvd.size()(number of workers) to scale learning rate and wrapping
# optimizer with Distributed optimizer class provided by horovod.
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01*hvd.size())
if hvd.size() > 1:
    optimizer = hvd.DistributedOptimizer(optimizer)


callbacks = [
    # Horovod: broadcast initial variable states from rank0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
] if hvd.size() > 1 else []

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, batch_size=128, callbacks=callbacks)

model.evaluate(x_test, y_test)

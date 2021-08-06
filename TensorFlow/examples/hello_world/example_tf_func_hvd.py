import tensorflow as tf

from TensorFlow.common.library_loader import load_habana_module
from TensorFlow.common.training import grad_utils
from TensorFlow.common.horovod_helpers import hvd, horovod_enabled

load_habana_module()
hvd.init()

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import time

if hvd.local_rank() == 0:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    hvd.broadcast(0, 0)
else:
    hvd.broadcast(0, 0)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

num_pics_per_rank = x_train.shape[0] // hvd.size()
pic_begin = num_pics_per_rank * hvd.rank()
pic_end = pic_begin + num_pics_per_rank
x_train = x_train[pic_begin:pic_end,]
y_train = y_train[pic_begin:pic_end,]

x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)



class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)
  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)


model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
optimizer =hvd.DistributedOptimizer(optimizer)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels, step):
  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)

  if horovod_enabled():
    tape = hvd.DistributedGradientTape(tape)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables), experimental_aggregate_gradients=True)
    tf.cond(step==0,
          lambda: hvd.broadcast_variables(model.variables + optimizer.variables(),
                                          root_rank=0),
          lambda: tf.constant(True))
  else:
    grad_utils.minimize_using_explicit_allreduce(
          tape, optimizer, loss, model.trainable_variables)

  train_loss(loss)
  train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)
  test_loss(t_loss)
  test_accuracy(labels, predictions)


EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  t1 = time.time()
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()
  for step, (images, labels) in enumerate(train_ds):
    train_step(images, labels, tf.constant(step))

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)
  print(
    f'Epoch {epoch + 1}, '
    f'Time: {time.time()-t1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )
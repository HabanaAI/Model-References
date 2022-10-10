import tensorflow as tf
from tensorflow.keras.metrics import categorical_accuracy as accuracy
from tensorflow.keras.layers import Dense
import input_data # A local file to help download and process the mnist dataset

from habana_frameworks.tensorflow import load_habana_module
load_habana_module()

tf.compat.v1.disable_eager_execution()

sess = tf.compat.v1.Session()

img = tf.compat.v1.placeholder(tf.float32, shape=(None, 784))

# Keras layers can be called on TensorFlow tensors:
x = Dense(128, activation='relu')(img)  # fully-connected layer with 128 units and ReLU activation
x = Dense(128, activation='relu')(x)
preds = Dense(10,activation='softmax')(x)  # output layer with 10 units and a softmax activation

labels = tf.compat.v1.placeholder(tf.float32, shape=(None,10))

loss = tf.compat.v1.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, preds))

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

acc_value = accuracy(labels, preds)

train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(loss)
sess.run(tf.compat.v1.global_variables_initializer())
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1]})
        result = acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels})
        if i % 10 == 0:
            print("Step:", '{:4d}'.format(i), " Evaluation acc: ", result.mean())

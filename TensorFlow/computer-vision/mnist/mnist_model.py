import logging

import tensorflow as tf


class MnistModel(object):
    tf_log = logging.getLogger('tf_log')

    def __init__(self, image_shape, num_classes, data_type, optimizer):
        super().__init__()
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.data_type = data_type
        self.optimizer = optimizer
        self.dropout = 0.8
        self.use_bn = False
        self.X = None
        self.labels = None
        self.learning_rate = 0.001
        self.loss_op, self.train_op, self.accuracy, self.init, self.init_l = \
            self.build_model()

    def get_topology_name(self):
        return "mnist"

    def build_model(self, optimizer="Adam"):
        if self.optimizer == 'Momentum':
            tf.compat.v1.enable_resource_variables()
        self.tf_log.info("Model: Simple convolutions topology written by DEV Team")

        self.X = tf.compat.v1.placeholder(self.data_type, self.image_shape)
        self.labels = tf.compat.v1.placeholder(self.data_type, shape=[None])

        weights, biases = self._get_weights_biases()

        # Construct model
        logits = self._conv_net(self.X, weights, biases, self.dropout, self.use_bn)

        sparse_softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(self.labels, dtype=tf.int32))
        loss_op = tf.reduce_mean(sparse_softmax)
        momentum = 0.9
        if self.optimizer == 'Adam':
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'Momentum':
            optimizer = tf.compat.v1.train.MomentumOptimizer(
                    self.learning_rate,
                    momentum,
                    use_locking=False,
                    name='Momentum',
                    use_nesterov=False )
        else:
          raise ValueError(f"unsupported optimizer {self.optimizer}")
        train_op = optimizer.minimize(loss_op, name='momentum')
        if self.use_bn:
            update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group([train_op, update_ops])

        # Evaluate model (with test logits, for dropout to be disabled)
        pred_classes = tf.argmax(logits, axis=1)
        accuracy = tf.compat.v1.metrics.accuracy(labels=self.labels, predictions=pred_classes)

        # Initialize the variables (i.e. assign their default value)
        init = tf.compat.v1.global_variables_initializer()
        init_l = tf.compat.v1.local_variables_initializer()

        return loss_op, train_op, accuracy, init, init_l

    def _get_weights_biases(self):
        weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.compat.v1.random_normal([5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.compat.v1.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.compat.v1.random_normal([7 * 7 * 64, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.compat.v1.random_normal([1024, self.num_classes]))
        }
        biases = {
            'bc1': tf.Variable(tf.compat.v1.random_normal([32])),
            'bc2': tf.Variable(tf.compat.v1.random_normal([64])),
            'bd1': tf.Variable(tf.compat.v1.random_normal([1024])),
            'out': tf.Variable(tf.compat.v1.random_normal([self.num_classes]))
        }

        return weights, biases

    def _conv2d(self, x, W, b, strides=1, use_bn=False):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        if use_bn:
            x = tf.compat.v1.layers.batch_normalization(x, training=True, axis=3, scale=True, fused=True, center=True, epsilon=1e-5, momentum=0.997)
        else:
            x = tf.nn.bias_add(x, b)

        return tf.nn.relu(x)

    def _maxpool2d(self, x, k=3, s=2, name=None):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME', name=name)

    def _conv_net(self, x, weights, biases, dropout, use_bn=False):
        # Convolution Layer
        conv1 = self._conv2d(x, weights['wc1'], biases['bc1'], use_bn=use_bn)
        # Max Pooling (down-sampling)
        conv1 = self._maxpool2d(conv1, k=3, s=2, name='pool1')

        # Convolution Layer
        conv2 = self._conv2d(conv1, weights['wc2'], biases['bc2'], use_bn=use_bn)
        # Max Pooling (down-sampling)
        conv2 = self._maxpool2d(conv2, k=3, s=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    def get_scalar_acc(self, acc):
        return acc[0][0]

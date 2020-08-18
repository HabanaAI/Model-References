"""models.py

Implemented model: Densenet121 ('densenet121')
"""


import tensorflow as tf

from config import config
from .adamw import AdamW
from .optimizer import convert_to_accum_optimizer
from .optimizer import convert_to_lookahead_optimizer
from .densenet import DenseNet


def DenseNet121(batch_size,
                include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
  """Instantiates the Densenet121 architecture."""
  return DenseNet(batch_size, [6, 12, 24, 16], include_top, weights, input_tensor,
                  input_shape, pooling, classes)

IN_SHAPE = (224, 224, 3)  # shape of input image tensor
NUM_CLASSES = 1000        # number of output classes (1000 for ImageNet)


def _set_l2(model, weight_decay):
    """Add L2 regularization into layers with weights
    Reference: https://jricheimer.github.io/keras/2019/02/06/keras-hack-1/
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            layer.add_loss(tf.keras.regularizers.l2(weight_decay)(layer.kernel))
            print('added wd to layer %s' % layer.name)
        elif isinstance(layer, tf.keras.layers.Conv2D):
            layer.add_loss(tf.keras.regularizers.l2(weight_decay)(layer.kernel))
            print('added wd to layer %s' % layer.name)
        elif isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(tf.keras.regularizers.l2(weight_decay)(layer.kernel))
            print('added wd to layer %s' % layer.name)

def get_batch_size(model_name, value):
    """get_batch_size

    These default batch_size values were chosen based on available
    GPU RAM (11GB) on GeForce GTX-2080Ti.
    """
    if value > 0:
        return value
    elif 'densenet121' in model_name:
        return 16
    else:
        raise ValueError


def get_iter_size(model_name, value):
    """get_iter_size

    These default iter_size values were chosen to make 'effective'
    batch_size to be 256.
    """
    if value > 0:
        return value
    elif 'densenet121' in model_name:
        return 16
    else:
        raise ValueError


def get_initial_lr(model_name, value):
    return value if value > 0. else 3e-4


def get_final_lr(model_name, value):
    return value if value > 0. else 3e-4


def get_lr_func(total_epochs, lr_sched='linear',
                initial_lr=1e-2, final_lr=1e-5):
    """Returns a learning decay function for training.

    2 types of lr_sched are supported: 'linear' or 'exp' (exponential).
    """
    def linear_decay(epoch):
        """Decay LR linearly for each epoch."""
        if total_epochs == 1:
            return initial_lr
        else:
            ratio = max((total_epochs - epoch - 1.) / (total_epochs - 1.), 0.)
            lr = final_lr + (initial_lr - final_lr) * ratio
            print('Epoch %d, lr = %f' % (epoch+1, lr))
            return lr

    def exp_decay(epoch):
        """Decay LR exponentially for each epoch."""
        if total_epochs == 1:
            return initial_lr
        else:
            lr_decay = (final_lr / initial_lr) ** (1. / (total_epochs - 1))
            lr = initial_lr * (lr_decay ** epoch)
            print('Epoch %d, lr = %f' % (epoch+1, lr))
            return lr

    def steps_decay(epoch):
        if total_epochs == 1:
            return initial_lr
        else:
            lr_decay = pow(0.1, epoch // 30)
            lr = initial_lr * lr_decay
            print('Epoch %d, lr = %f' % (epoch+1, lr))
            return lr



    if total_epochs < 1:
        raise ValueError('bad total_epochs (%d)' % total_epochs)
    if lr_sched == 'linear':
        return tf.keras.callbacks.LearningRateScheduler(linear_decay)
    elif lr_sched == 'exp':
        return tf.keras.callbacks.LearningRateScheduler(exp_decay)
    elif lr_sched == 'steps':
        return tf.keras.callbacks.LearningRateScheduler(steps_decay)
    else:
        raise ValueError('bad lr_sched')


def get_weight_decay(model_name, value):
    return value if value >= 0. else 1e-5


def get_optimizer(model_name, optim_name, initial_lr, epsilon=1e-2):
    """get_optimizer

    Note:
    1. Learning rate decay is implemented as a callback in model.fit(),
       so I do not specify 'decay' in the optimizers here.
    2. Refer to the following for information about 'epsilon' in Adam:
       https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/python/keras/optimizer_v2/adam.py#L93
    """
    if optim_name == 'sgd':
        return tf.keras.optimizers.SGD(lr=initial_lr, momentum=0.9)
    elif optim_name == 'adam':
        return tf.keras.optimizers.Adam(lr=initial_lr, epsilon=epsilon)
    elif optim_name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(lr=initial_lr, epsilon=epsilon,
                                           rho=0.9)
    else:
        # implementation of 'AdamW' is removed temporarily
        raise ValueError


def get_training_model(model_name, dropout_rate, optimizer, label_smoothing,
                       use_lookahead, iter_size, weight_decay, batch_size):
    """Build the model to be trained."""
    # initialize the model from scratch
    backbone = DenseNet121(
        input_shape=IN_SHAPE, include_top=False, weights=None, batch_size=batch_size)

    # Add a Dropout layer before the final Dense output
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    if dropout_rate and dropout_rate > 0.0 and dropout_rate < 1.0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    kernel_initializer = tf.compat.v1.random_normal_initializer(mean=0.0, stddev=0.02)
    bias_initializer = tf.compat.v1.constant_initializer(value=0.0)
    x = tf.keras.layers.Dense(
        NUM_CLASSES, activation='softmax', name='Logits',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer)(x)
    model = tf.keras.models.Model(inputs=backbone.input, outputs=x)

    if weight_decay > 0.:
        _set_l2(model, weight_decay)
    if iter_size > 1:
        optimizer = convert_to_accum_optimizer(optimizer, iter_size)
    if use_lookahead:
        optimizer = convert_to_lookahead_optimizer(optimizer)

    # make sure all layers are set to be trainable
    for layer in model.layers:
        layer.trainable = True

    if tf.__version__ >= '1.14':
        smooth = 0.1 if label_smoothing else 0.
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=smooth)
    else:
        tf.compat.v1.logging.warning('"--label_smoothing" not working for '
                           'tensorflow-%s' % tf.__version__)
        loss = 'categorical_crossentropy'
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy'])
    print(model.summary())

    return model

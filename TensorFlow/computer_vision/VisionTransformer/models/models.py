###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import math

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow_addons.utils import types
from typeguard import typechecked


class GradientAccumulator(tf.keras.optimizers.Optimizer):
    """Optimizer wrapper for gradient accumulation."""

    @typechecked
    def __init__(
        self,
        optimizer: types.Optimizer,
        accum_steps: types.TensorLike = 4,
        name: str = "GradientAccumulator",
        **kwargs,
    ):
        r"""Construct a new GradientAccumulator optimizer.
        Args:
            optimizer: str or `tf.keras.optimizers.Optimizer` that will be
                used to compute and apply gradients.
            accum_steps: int > 0. Update gradient in every accumulation steps.
            name: Optional name for the operations created when applying
                gradients. Defaults to "GradientAccumulator".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
                norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(name, **kwargs)
        self._optimizer = tf.keras.optimizers.get(optimizer)
        self._gradients = []
        self._accum_steps = accum_steps

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, "ga")

        self._gradients = [self.get_slot(var, "ga") for var in var_list]

    @property
    def gradients(self):
        """The accumulated gradients on the current replica."""
        if not self._gradients:
            raise ValueError(
                "The accumulator should be called first to initialize the gradients"
            )
        return list(
            gradient.read_value() if gradient is not None else gradient
            for gradient in self._gradients
        )

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        self._optimizer._iterations = self.iterations
        return super().apply_gradients(grads_and_vars, name, **kwargs)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        accum_gradient = self.get_slot(var, "ga")
        if accum_gradient is not None and grad is not None:
            accum_gradient.assign_add(
                grad / self._accum_steps, use_locking=self._use_locking, read_value=False
            )

        def _apply():
            if "apply_state" in self._optimizer._dense_apply_args:
                train_op = self._optimizer._resource_apply_dense(
                    accum_gradient.read_value(), var, apply_state=apply_state
                )
            else:
                train_op = self._optimizer._resource_apply_dense(
                    accum_gradient.read_value(), var
                )
            reset_op = accum_gradient.assign(
                tf.zeros_like(accum_gradient),
                use_locking=self._use_locking,
                read_value=False,
            )
            return tf.group(train_op, reset_op)

        apply_op = tf.cond(
            (self.iterations + 1) % self._accum_steps == 0, _apply, lambda: tf.no_op()
        )
        return apply_op

    def _resource_apply_sparse(self, grad: types.TensorLike, var, indices, apply_state):
        accum_gradient = self.get_slot(var, "ga")
        if accum_gradient is not None and grad is not None:
            self._resource_scatter_add(accum_gradient, indices, grad)

        def _apply():
            if "apply_state" in self._optimizer._sparse_apply_args:
                train_op = self._optimizer._resource_apply_sparse(
                    accum_gradient.sparse_read(indices),
                    var,
                    indices,
                    apply_state=apply_state,
                )
            else:
                train_op = self._optimizer._resource_apply_sparse(
                    accum_gradient.sparse_read(indices), var, indices
                )
            reset_op = accum_gradient.assign(
                tf.zeros_like(accum_gradient),
                use_locking=self._use_locking,
                read_value=False,
            )
            return tf.group(train_op, reset_op)

        apply_op = tf.cond(
            (self.iterations + 1) % self._accum_steps == 0, _apply, lambda: tf.no_op()
        )
        return apply_op

    def reset(self):
        """Resets the accumulated gradients on the current replica."""
        assign_ops = []
        if not self._gradients:
            return assign_ops

        for gradient in self._gradients:
            if gradient is not None:
                assign_ops.append(
                    gradient.assign(
                        tf.zeros_like(gradient),
                        use_locking=self._use_locking,
                        read_value=False,
                    )
                )

        return tf.group(assign_ops)

    @property
    def lr(self):
        return self._optimizer._get_hyper("learning_rate")

    @lr.setter
    def lr(self, lr):
        self._optimizer._set_hyper("learning_rate", lr)  #

    @property
    def learning_rate(self):
        return self._optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer._set_hyper("learning_rate", learning_rate)

    def get_config(self):
        config = {
            "accum_steps": self._accum_steps,
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects
        )
        return cls(optimizer, **config)


IN_SHAPE = (224, 224, 3)  # shape of input image tensor
NUM_CLASSES = 1000        # number of output classes (1000 for ImageNet)


def _set_l2(model, weight_decay):
    """Add L2 regularization into layers with weights
    Reference: https://jricheimer.github.io/keras/2019/02/06/keras-hack-1/
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            layer.add_loss(lambda: tf.keras.regularizers.l2(
                weight_decay)(layer.kernel))
            print('added wd to layer %s' % layer.name)
        elif isinstance(layer, tf.keras.layers.Conv2D):
            #layer.add_loss(lambda: keras.regularizers.l2(weight_decay)(layer.kernel))
            print('added wd to layer %s' % layer.name)
        elif isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l2(
                weight_decay)(layer.kernel))
            print('added wd to layer %s' % layer.name)
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.add_loss(lambda: tf.keras.regularizers.l2(
                weight_decay)(layer.kernel))
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


class CosineLearningRateScheduleWithWarmup(tf.keras.callbacks.Callback):
    def __init__(self, schedule, initial_lr, warmup_steps, resume_step, total_steps, cycles=0.5, verbose=0):
        super(CosineLearningRateScheduleWithWarmup, self).__init__()

        self.schedule = schedule
        self.verbose = verbose
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.resume_step = resume_step
        self.total_steps = total_steps
        self.cycles = cycles

    def on_train_begin(self, logs=None):
        self.iter = 0 + self.resume_step

    def on_train_batch_end(self, batch, logs=None):
        self.iter += 1
        if self.iter < self.warmup_steps:
            warmup_multiplier = float(self.iter) / float(self.warmup_steps)
            lr = self.initial_lr * warmup_multiplier
            backend.set_value(self.model.optimizer.lr, lr)
        else:
            progress = float(self.iter - self.warmup_steps) / \
                float(max(1, self.total_steps - self.warmup_steps))
            lr = max(0.0, 0.5 * (1. + tf.math.cos(math.pi *
                     float(self.cycles) * 2.0 * progress)))*self.initial_lr
            backend.set_value(self.model.optimizer.lr, lr)

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'learning_rate'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            lr = float(backend.get_value(self.model.optimizer.learning_rate))
        except TypeError:  # Support for old API for backward compatibility
            print("An exception occurred")

        backend.set_value(self.model.optimizer.learning_rate,
                          backend.get_value(lr))
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = backend.get_value(self.model.optimizer.learning_rate)


def get_lr_func(total_epochs, lr_sched='linear',
                initial_lr=6e-2, final_lr=1e-5, warmup_steps=0, resume_step=0, total_steps=5004):
    """Returns a learning decay function for training.

    5 types of lr_sched are supported: 'linear' or 'exp', 'steps' , 'constant' , 'WarmupCosine' (exponential).
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
            # learning rate is reduced by x10 at the end of epochs: 30,60,80,110,140,...
            return initial_lr
        else:
            if (epoch < 80):
                lr_decay = pow(0.1, epoch // 30)
            else:
                lr_decay = pow(0.1, (epoch+10) // 30)
                print(epoch)
                print(lr_decay)

            lr = initial_lr * lr_decay
            print('Epoch %d, lr = %f' % (epoch+1, lr))
            return lr

    def constant(epoch):
        """Decay LR exponentially for each epoch."""
        if total_epochs == 1:
            return initial_lr
        else:
            lr = initial_lr
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
    elif lr_sched == 'constant':
        return tf.keras.callbacks.LearningRateScheduler(constant)
    elif lr_sched == 'WarmupCosine':
        return CosineLearningRateScheduleWithWarmup(constant, initial_lr, warmup_steps, resume_step, total_steps)

    else:
        raise ValueError('bad lr_sched')


def get_weight_decay(model_name, value):
    return value if value >= 0. else 1e-5


def get_optimizer(optim_name, initial_lr, accumulation_steps=1, epsilon=1e-2):
    """get_optimizer

    Note:
    1. Learning rate decay is implemented as a callback in model.fit(),
       so I do not specify 'decay' in the optimizers here.
    2. Refer to the following for information about 'epsilon' in Adam:
       https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/python/keras/optimizer_v2/adam.py#L93
    """
    from functools import partial
    if optim_name == 'sgd':
        optimizer = partial(tf.keras.optimizers.SGD,
                            momentum=0.9, nesterov=False, global_clipnorm=1.0)
    elif optim_name == 'adam':
        optimizer = partial(tf.keras.optimizers.Adam, epsilon=epsilon)
    elif optim_name == 'rmsprop':
        optimizer = partial(tf.keras.optimizers.RMSprop,
                            rho=0.9, epsilon=epsilon)
    else:
        # implementation of 'AdamW' is removed temporarily
        raise ValueError

    optimizer = optimizer(learning_rate=initial_lr)
    if accumulation_steps > 1:
        optimizer = GradientAccumulator(
            optimizer, accum_steps=accumulation_steps)

    return optimizer

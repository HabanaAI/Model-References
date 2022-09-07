"""adamw.py

Taken from https://github.com/shaoanlu/AdamW-and-SGDW
"""


from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Optimizer


class AdamW(Optimizer):
    """AdamW optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: float >= 0. Decoupled weight decay over each update.
        weight_decay_normalizer: float >= 0. Calculated as: 1. / sqrt(batches_per_epoch * num_epochs)
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [Optimization for Deep Learning Highlights in 2017](http://ruder.io/deep-learning-optimization-2017/index.html)
        - [Fixing Weight Decay Regularization in Adam](https://arxiv.org/abs/1711.05101)
    """

    def __init__(self,
                 lr=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=None,
                 decay=0.,
                 weight_decay=0.0025,         # decoupled weight decay (1/6)
                 weight_decay_normalizer=1.,  #
                 **kwargs):
        super(AdamW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.init_lr = lr  # decoupled weight decay (2/6)
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.wd = K.variable(weight_decay, name='weight_decay')  # decoupled weight decay (3/6)
            self.wd_normalizer = weight_decay_normalizer             #
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        wd = self.wd * self.wd_normalizer  # decoupled weight decay (4/6)

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. /
                       (1. + self.decay *
                             math_ops.cast(self.iterations,
                                          K.dtype(self.decay))))
        eta_t = lr / self.init_lr    # decoupled weight decay (5/6)

        with ops.control_dependencies([state_ops.assign_add(self.iterations, 1)]):
            t = math_ops.cast(self.iterations, K.floatx())
        """Bias corrections according to the Adam paper."""
        lr_t = lr * (K.sqrt(1. - math_ops.pow(self.beta_2, t)) /
                     (1. - math_ops.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(g)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
            p_t -= eta_t * wd * p  # decoupled weight decay (6/6)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'weight_decay': float(K.get_value(self.wd)),
                  'epsilon': self.epsilon}
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

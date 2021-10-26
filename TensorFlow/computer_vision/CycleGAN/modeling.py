###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import tensorflow_addons as tfa


def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization


def get_resnet_generator(input_shape=(256, 256, 3),
                         output_channels=3,
                         dim=64,
                         n_downsamplings=2,
                         n_blocks=9,
                         norm='instance_norm',
                         name='generator'):
    Norm = _get_norm_layer(norm)

    def _residual_block(x):
        dim = x.shape[-1]
        h = x
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)
        return keras.layers.add([x, h])

    h = inputs = keras.Input(shape=input_shape, name=name + '_input')

    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)

    for _ in range(n_downsamplings):
        dim *= 2
        h = keras.layers.Conv2D(
            dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    for _ in range(n_blocks):
        h = _residual_block(h)

    for _ in range(n_downsamplings):
        dim //= 2
        h = keras.layers.Conv2DTranspose(
            dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(output_channels, 7, padding='valid')(h)
    h = tf.tanh(h)
    h = tf.cast(h, tf.float32)

    return keras.Model(inputs=inputs, outputs=h, name=name)


def get_discriminator(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm',
                      name='discriminator'):
    dim_ = dim
    Norm = _get_norm_layer(norm)

    h = inputs = keras.Input(shape=input_shape, name=name + '_input')

    h = keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = keras.layers.Conv2D(
            dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)

    dim = min(dim * 2, dim_ * 8)
    h = keras.layers.Conv2D(
        dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    h = keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)
    h = tf.cast(h, tf.float32)

    return keras.Model(inputs=inputs, outputs=h, name=name)


class TFPool:

    def __init__(self, initial_pool, sample_rate=0.5, batch_size=None):
        self.pool = tf.stop_gradient(initial_pool)
        self.items = []
        self.batch_size = batch_size
        self.sampler = tfp.distributions.Bernoulli(sample_rate,dtype=tf.bool)
        if self.batch_size is not None:
            self.batch_ids = tf.range(self.batch_size)

        self.pool_size = initial_pool.shape[0]
        self.pool_ids = tf.range(self.pool_size)

    @tf.function
    def __call__(self, in_batch):
        # `in_batch` should be a batch tensor

        if self.pool_size == 0:
            return in_batch

        if self.batch_size is None:
            tf.assert_greater(in_batch.shape[0], 0)
            self.batch_size = in_batch.shape[0]
            self.batch_ids = tf.range(in_batch.shape[0])

        replace_batch_bool = self.sampler.sample(self.batch_size)
        replace_batch_ids = self.batch_ids[replace_batch_bool]
        sample_size = tf.shape(replace_batch_ids)[0]
        if sample_size > 0: # scatterupdate will fail on empty sample, replace with static impl for batch size > 1
            sample_pool_ids = tf.random.shuffle(self.pool_ids)[:sample_size]
            batch_buffer = tf.stop_gradient(in_batch[replace_batch_bool])
            pooled_buffer = tf.gather(self.pool, sample_pool_ids)
            new_batch = tf.tensor_scatter_nd_update(in_batch, tf.expand_dims(replace_batch_ids,1), pooled_buffer)
            self.pool = tf.tensor_scatter_nd_update(self.pool, tf.expand_dims(sample_pool_ids,1), batch_buffer)
        else:
            new_batch = in_batch
        return new_batch


class CycleGan(keras.Model):
    def __init__(
        self,
        generator_Y,
        generator_X,
        discriminator_X,
        discriminator_Y,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super(CycleGan, self).__init__()
        self.gen_Y = generator_Y
        self.gen_X = generator_X
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    @tf.function()
    def _generate_cycle(self, input, transform, reverse, training=None):
        fake = transform(input, training=training)
        cycled = reverse(fake, training=training)
        return fake, cycled

    @tf.function()
    def call(self, inputs, training=None):
        x = inputs[0]
        y = inputs[1]
        fake_y = self.gen_Y(x, training=training)
        fake_x = self.gen_X(y, training=training)
        cycled_x, cycled_y = self.gen_X(
            fake_y, training=training), self.gen_Y(fake_x, training=training)
        return fake_x, cycled_y, fake_y, cycled_x

    def compile(
        self,
        gen_optimizer,
        disc_optimizer,
        gen_loss_fn, cycle_loss, id_loss,
        disc_loss_fn, hvd=None, pool_f=None, pool_g=None, **kwargs
    ):
        super(CycleGan, self).compile(**kwargs)
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss
        self.identity_loss_fn = id_loss
        self.hvd = hvd
        self.first_step = 1
        self.pool_f = pool_f
        self.pool_g = pool_g
        self.optimizer = dict(
            gen_optimizer=self.gen_optimizer,
            disc_optimizer=self.disc_optimizer,
        )
        self.disc_variables = self.disc_X.trainable_variables + \
            self.disc_Y.trainable_variables
        self.gen_variables = self.gen_X.trainable_variables + self.gen_Y.trainable_variables
        self.im_quality_loss = None
        self.im_quality_scale = 1.0

    @tf.function()
    def train_step(self, batch_data):
        # x is Horse and y is zebra
        real_x, real_y = batch_data

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adverserial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.disc_variables+self.gen_variables)
            # Horse to fake zebra f->fG
            fake_y = self.gen_Y(real_x, training=True)
            # Zebra to fake horse -> y2x g->gF
            fake_x = self.gen_X(real_y, training=True)

            # Cycle (Horse to fake zebra to fake horse): fG -> fGF
            cycled_x = self.gen_X(fake_y, training=True)
            # Cycle (Zebra to fake horse to fake zebra) gF -> gFG
            cycled_y = self.gen_Y(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_X(real_x, training=True)  # f -> F
            same_y = self.gen_Y(real_y, training=True)  # g -> G

            # Discriminator output
            disc_fake_x = self.disc_X(fake_x, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_y_loss = self.generator_loss_fn(disc_fake_y)
            gen_x_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_Y = self.cycle_loss_fn(real_y, cycled_y)
            cycle_loss_X = self.cycle_loss_fn(real_x, cycled_x)

            # Generator identity loss
            id_loss_Y = self.identity_loss_fn(real_y, same_y)
            id_loss_X = self.identity_loss_fn(real_x, same_x)

            # Total generator loss
            total_generator_loss = gen_y_loss + gen_x_loss + \
                (cycle_loss_Y + cycle_loss_X)*self.lambda_cycle + \
                (id_loss_Y + id_loss_X)*self.lambda_identity

            # Image quality loss
            vis_loss_x, vis_loss_y = 0, 0
            if self.im_quality_loss:
                vis_loss_y = tf.reduce_mean(self.im_quality_loss(
                    same_y) + self.im_quality_loss(cycled_y) + self.im_quality_loss(fake_y))*self.im_quality_scale
                vis_loss_x = tf.reduce_mean(self.im_quality_loss(
                    same_x) + self.im_quality_loss(cycled_x) + self.im_quality_loss(fake_x))*self.im_quality_scale
                total_generator_loss = total_generator_loss + vis_loss_x + vis_loss_y

            # compute discriminator loss on potentially new samples!
            if self.pool_g and self.pool_f:
                # pool swap
                fake_x = self.pool_f(fake_x)
                fake_y = self.pool_g(fake_y)
                disc_fake_x = self.disc_X(fake_x, training=True)
                disc_fake_y = self.disc_Y(fake_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_real_y = self.disc_Y(real_y, training=True)

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)
            total_discriminator_loss = disc_X_loss + disc_Y_loss

        if self.hvd:
            tape = self.hvd.DistributedGradientTape(tape)
        # Get the gradients for the generators
        grads_generator = tape.gradient(
            total_generator_loss, self.gen_variables)
        grads_discriminator = tape.gradient(
            total_discriminator_loss, self.disc_variables)

        self.gen_optimizer.apply_gradients(
            zip(grads_generator, self.gen_variables)
        )
        # Get the gradients for the discriminators
        self.disc_optimizer.apply_gradients(
            zip(grads_discriminator, self.disc_variables)
        )

        if self.first_step and self.hvd:
            self.hvd.broadcast_variables(self.variables, root_rank=0)
            self.hvd.broadcast_variables(
                self.gen_optimizer.variables(), root_rank=0)
            self.hvd.broadcast_variables(
                self.disc_optimizer.variables(), root_rank=0)

        self.first_step = False
        losses = {
            "loss": total_generator_loss + total_discriminator_loss,
            "total_generator_loss": total_generator_loss,
            "total_discriminator_loss": total_discriminator_loss,
            "gen_Y_loss": gen_y_loss,
            "gen_X_loss": gen_x_loss,
            "cycle_loss_Y": cycle_loss_Y,
            "cycle_loss_X": cycle_loss_X,
            "id_loss_Y": id_loss_Y,
            "id_loss_X": id_loss_X,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
            "disc_lr": self.disc_optimizer.lr,
            "gen_lr": self.gen_optimizer.lr,
        }
        if self.im_quality_loss:
            losses.update({"vis_y_loss": vis_loss_y,
                          "vis_x_loss": vis_loss_x, })
        return losses

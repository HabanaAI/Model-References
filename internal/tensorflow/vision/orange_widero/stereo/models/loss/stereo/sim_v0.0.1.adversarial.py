import tensorflow as tf
from stereo.models.loss_utils import get_reg_losses, loss_lidar
from stereo.interfaces.implements import implements_format
from tensorflow.python.training.training_util import get_or_create_global_step


def calc_scale(global_step, num_steps):
    return (2 / (1 + tf.exp(-10 * tf.minimum(tf.cast(global_step, tf.float32)/num_steps, 1.0)))) - 1


loss_format = ("loss", "sim_domain_short")
@implements_format(*loss_format)
def loss(out, sim_depth_inv, im_lidar_short_inv, domain, alpha_sim=1.0, alpha_adv=1.0, kernel_reg_constant=1.0,
         activation_reg_constant=1.0, label_smoothing=0.0):

    invZ = out['out']
    loss_sim_ = loss_lidar(invZ, None, tf.zeros_like(sim_depth_inv), inv_lidar=sim_depth_inv)
    loss_sim_ = tf.identity(loss_sim_ * 40.0, "batch_sim")

    kernel_reg_losses = get_reg_losses('kernel')
    activation_reg_losses = get_reg_losses('activation')
    loss_kernel_reg = kernel_reg_constant * tf.reduce_sum(input_tensor=kernel_reg_losses)
    loss_activation_reg = activation_reg_constant * tf.reduce_sum(input_tensor=activation_reg_losses)

    batch_loss = tf.identity(loss_sim_ * alpha_sim, name="batch_loss")

    loss_adv = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=label_smoothing)(y_true=domain,
                                                                                                     y_pred=out['adv'])
    loss_gen_adv = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                      label_smoothing=label_smoothing)(y_true=1 - domain,
                                                                                       y_pred=out['adv'])
    # scale = calc_scale(get_or_create_global_step(), adv_num_steps)
    loss_ = tf.reduce_mean(input_tensor=batch_loss) + loss_kernel_reg + loss_activation_reg

    loss_short = loss_lidar(invZ, None, tf.zeros_like(im_lidar_short_inv), inv_lidar=im_lidar_short_inv)
    real_examples_mask = tf.cast(tf.equal(domain, 0.0), tf.float32)
    loss_short = tf.reduce_sum(input_tensor=loss_short * real_examples_mask) / tf.maximum(tf.reduce_sum(input_tensor=real_examples_mask), 1.0)
    # prepare and name additional tensors to be summarized
    _ = tf.identity(tf.reduce_mean(input_tensor=loss_sim_), name="loss_sim")
    _ = tf.identity(loss_short, name="loss_short_on_real")
    _ = tf.identity(loss_adv, name="loss_adv")
    _ = tf.identity(loss_kernel_reg, name="loss_kernel_reg")
    _ = tf.identity(loss_activation_reg, name="loss_activation_reg")

    return {'loss_gen': loss_, 'loss_adv': alpha_adv * loss_adv, 'loss_gen_adv': alpha_adv * loss_gen_adv,
            'loss': loss_ + alpha_adv * loss_adv}

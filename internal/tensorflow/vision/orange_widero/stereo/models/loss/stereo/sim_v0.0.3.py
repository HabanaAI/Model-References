import tensorflow as tf
from stereo.models.loss_utils import get_reg_losses, loss_lidar
from stereo.interfaces.implements import implements_format


loss_format = ("loss", "sim_short")
@implements_format(*loss_format)
def loss(out, sim_depth_inv, im_lidar_short_inv, im_mask, alpha_sim=1.0, alpha_lidar_short=1.0, kernel_reg_constant=1.0,
         activation_reg_constant=1.0, lidar_short_loss_norm=None, short_max_Z=None, short_only_in_mask=False):

    loss_sim_ = loss_lidar(out, None, tf.zeros_like(sim_depth_inv), inv_lidar=sim_depth_inv)
    loss_sim_ = tf.identity(loss_sim_ * 40.0, "batch_sim")

    if short_max_Z is not None:
        im_lidar_short_inv = im_lidar_short_inv * tf.cast(im_lidar_short_inv > 1./short_max_Z, tf.float32)
    if short_only_in_mask:
        im_lidar_short_inv = im_lidar_short_inv*im_mask

    loss_lidar_short_ = loss_lidar(out, None, tf.zeros_like(im_lidar_short_inv), inv_lidar=im_lidar_short_inv,
                                   norm=lidar_short_loss_norm)
    loss_lidar_short_ = tf.identity(loss_lidar_short_ * 40.0, "batch_lidar_short")

    kernel_reg_losses = get_reg_losses('kernel')
    activation_reg_losses = get_reg_losses('activation')
    loss_kernel_reg = kernel_reg_constant * tf.reduce_sum(input_tensor=kernel_reg_losses)
    loss_activation_reg = activation_reg_constant * tf.reduce_sum(input_tensor=activation_reg_losses)

    batch_loss = tf.identity(loss_sim_ * alpha_sim + loss_lidar_short_ * alpha_lidar_short, name="batch_loss")
    loss_ = tf.reduce_mean(input_tensor=batch_loss) + loss_kernel_reg + loss_activation_reg

    # prepare and name additional tensors to be summarized
    _ = tf.identity(tf.reduce_mean(input_tensor=loss_sim_), name="loss_sim")
    _ = tf.identity(tf.reduce_mean(input_tensor=loss_lidar_short_), name="loss_lidar_short")
    _ = tf.identity(loss_kernel_reg, name="loss_kernel_reg")
    _ = tf.identity(loss_activation_reg, name="loss_activation_reg")

    return loss_

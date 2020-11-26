import tensorflow as tf
from stereo.models.loss_utils import get_reg_losses, loss_lidar, min_photo_loss_3_images
from stereo.interfaces.implements import implements_format


loss_format = ("loss", "sim_real_domain")
@implements_format(*loss_format)
def loss(out, sim_depth_inv, I_cntr, I_srnd_0, I_srnd_1, I_srnd_2, T_cntr_srnd, focal, origin,
         photo_loss_im_mask, x, y, blur_kernels, domain,
         measure='ssim', radius=1, alpha_sim=1.0, alpha_phot=1.0, kernel_reg_constant=1.0, activation_reg_constant=1.0):

    loss_phot_ = min_photo_loss_3_images(out, I_cntr, I_srnd_0, I_srnd_1, I_srnd_2, T_cntr_srnd, focal, origin,
                                         blur_kernels, photo_loss_im_mask, x, y, measure, radius)
    loss_phot_ = loss_phot_ * tf.cast(tf.equal(domain, 0), tf.float32)
    loss_phot_ = tf.identity(loss_phot_, name="batch_photo")

    loss_sim_ = loss_lidar(out, None, tf.zeros_like(sim_depth_inv), inv_lidar=sim_depth_inv)
    loss_sim_ = tf.identity(loss_sim_ * 40.0, "batch_sim")

    loss_phot = alpha_phot * tf.reduce_sum(input_tensor=loss_phot_)/tf.maximum(tf.reduce_sum(input_tensor=tf.cast(tf.equal(domain, 0), tf.float32)), 1.)
    kernel_reg_losses = get_reg_losses('kernel')
    activation_reg_losses = get_reg_losses('activation')
    loss_kernel_reg = kernel_reg_constant * tf.reduce_sum(input_tensor=kernel_reg_losses)
    loss_activation_reg = activation_reg_constant * tf.reduce_sum(input_tensor=activation_reg_losses)

    batch_loss = tf.identity(loss_sim_ * alpha_sim, name="batch_loss")

    loss_ = (tf.reduce_mean(input_tensor=batch_loss) + loss_phot + loss_kernel_reg + loss_activation_reg)

    # prepare and name additional tensors to be summarized
    _ = tf.identity(tf.reduce_mean(input_tensor=loss_sim_), name="loss_sim")
    _ = tf.identity(loss_phot, name="loss_phot")
    _ = tf.identity(loss_kernel_reg, name="loss_kernel_reg")
    _ = tf.identity(loss_activation_reg, name="loss_activation_reg")

    return loss_

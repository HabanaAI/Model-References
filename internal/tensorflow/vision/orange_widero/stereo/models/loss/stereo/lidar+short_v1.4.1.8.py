import tensorflow as tf
from stereo.models.loss_utils import loss_lidar
from stereo.interfaces.implements import implements_format


loss_format = ("loss", "lidar_only")
@implements_format(*loss_format)
def loss(out, im_lidar_inv, im_lidar_short_inv, im_mask, 
         alpha_lidar=1.0, alpha_lidar_short=1.0, reg_constant=1.0):

    loss_lidar_ = loss_lidar(out, None, im_mask, inv_lidar=im_lidar_inv)
    loss_lidar_ = tf.identity(loss_lidar_ * 40.0, "batch_lidar")

    loss_lidar_short_ = loss_lidar(out, None, tf.zeros_like(im_mask), inv_lidar=im_lidar_short_inv)
    loss_lidar_short_ = tf.identity(loss_lidar_short_ * 40.0, "batch_lidar_short")

    reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
    loss_reg = reg_constant * tf.reduce_sum(input_tensor=reg_losses)

    batch_loss = tf.identity(loss_lidar_ * alpha_lidar + 
                             loss_lidar_short_ * alpha_lidar_short, 
                             name="batch_loss")
    loss_ = tf.reduce_mean(input_tensor=batch_loss) + loss_reg

    # prepare and name additional tensors to be summarized
    _ = tf.identity(tf.reduce_mean(input_tensor=loss_lidar_), name="loss_lidar")
    _ = tf.identity(tf.reduce_mean(input_tensor=loss_lidar_short_), name="loss_lidar_short")
    _ = tf.identity(loss_reg, name="loss_reg")

    return loss_

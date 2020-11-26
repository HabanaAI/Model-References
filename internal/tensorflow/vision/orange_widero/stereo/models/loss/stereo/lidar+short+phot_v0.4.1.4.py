import tensorflow as tf
from stereo.models.loss_utils import loss_image, loss_lidar, min_photo_loss, batch_conv2d
from stereo.interfaces.implements import implements_format


loss_format = ("loss", "four_views_short")
@implements_format(*loss_format)
def loss(out, I_cntr, I_srnd_0, I_srnd_1, I_srnd_2, T_cntr_srnd, focal, origin, im_lidar, im_lidar_short, im_mask, x, y, blur_kernels,
         measure='ssim', radius=1, alpha_ooi=0.0, alpha_lidar=1.0, alpha_lidar_short=1.0, reg_constant=1.0):

    Z_inv_cntr = out

    I_cntr_conv_0 = batch_conv2d(I_cntr, tf.expand_dims(blur_kernels[:,:,:,0],-1))
    I_cntr_conv_1 = batch_conv2d(I_cntr, tf.expand_dims(blur_kernels[:,:,:,1],-1))
    I_cntr_conv_2 = batch_conv2d(I_cntr, tf.expand_dims(blur_kernels[:,:,:,2],-1))

    lidar = tf.expand_dims(im_lidar, axis=-1)
    lidar_short = tf.expand_dims(im_lidar_short, axis=-1)
    mask = tf.expand_dims(tf.cast(im_mask, dtype=tf.float32), axis=-1)

    I_warp_cntr_0, loss_im_0, mask_ioi_0, mean_ioi_0, mean_ooi_0 = loss_image(I1=I_cntr_conv_0, I2=I_srnd_0,
                                                                              Z1_inv=Z_inv_cntr, focal=focal,
                                                                              origin=origin,
                                                                              RT12=T_cntr_srnd[:, 0, :], x1=x, y1=y,
                                                                              measure=measure, radius=radius,
                                                                              cut_off=None, mask1=None,
                                                                              ooi_on_mask=True)

    I_warp_cntr_1, loss_im_1, mask_ioi_1, mean_ioi_1, mean_ooi_1 = loss_image(I1=I_cntr_conv_1, I2=I_srnd_1,
                                                                              Z1_inv=Z_inv_cntr, focal=focal,
                                                                              origin=origin,
                                                                              RT12=T_cntr_srnd[:, 1, :], x1=x, y1=y,
                                                                              measure=measure, radius=radius,
                                                                              cut_off=None, mask1=None,
                                                                              ooi_on_mask=True)

    I_warp_cntr_2, loss_im_2, mask_ioi_2, mean_ioi_2, mean_ooi_2 = loss_image(I1=I_cntr_conv_2, I2=I_srnd_2,
                                                                              Z1_inv=Z_inv_cntr, focal=focal,
                                                                              origin=origin,
                                                                              RT12=T_cntr_srnd[:, 2, :], x1=x, y1=y,
                                                                              measure=measure, radius=radius,
                                                                              cut_off=None, mask1=None,
                                                                              ooi_on_mask=True)

    loss_phot_ = 2 * min_photo_loss([loss_im_0, loss_im_1, loss_im_2], [mask_ioi_0, mask_ioi_1, mask_ioi_2],
                                    radius=radius)
    loss_phot_ = tf.identity(loss_phot_, name="batch_photo")

    loss_lidar_ = loss_lidar(Z_inv_cntr, lidar, mask)
    loss_lidar_ = tf.identity(loss_lidar_ * 40.0, "batch_lidar")

    loss_lidar_short_ = loss_lidar(Z_inv_cntr, lidar_short, tf.zeros_like(mask))
    loss_lidar_short_ = tf.identity(loss_lidar_short_ * 40.0, "batch_lidar_short")

    reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
    reg_loss = reg_constant * sum(reg_losses)

    batch_loss = tf.identity(loss_phot_ + loss_lidar_ * alpha_lidar + loss_lidar_short_* alpha_lidar_short,
                             name="batch_loss")
    loss_ = tf.reduce_mean(input_tensor=batch_loss)

    # prepare and name tensors to be summarized
    _ = tf.reverse(Z_inv_cntr, axis=[1], name='Z_inv_cntr')
    _ = tf.reverse(I_cntr, axis=[1], name='I_cntr')
    _ = tf.reverse(I_cntr_conv_0, axis=[1], name='I_cntr_conv_0')
    _ = tf.reverse(I_cntr_conv_1, axis=[1], name='I_cntr_conv_1')
    _ = tf.reverse(I_warp_cntr_0, axis=[1], name='I_warp_cntr_0')
    _ = tf.reverse(I_warp_cntr_1, axis=[1], name='I_warp_cntr_1')
    _ = tf.reverse(I_warp_cntr_2, axis=[1], name='I_warp_cntr_2')
    _ = tf.reverse(I_srnd_0, axis=[1], name='I_srnd_0')
    _ = tf.reverse(I_srnd_1, axis=[1], name='I_srnd_1')
    _ = tf.reverse(I_srnd_2, axis=[1], name='I_srnd_2')
    _ = tf.reverse((1.0 - tf.cast(tf.equal(lidar, 0.0), dtype=tf.float32)) * (lidar + 1e-9)**-1, axis=[1], name='lidar_inv')
    _ = tf.reverse((1.0 - tf.cast(tf.equal(lidar_short, 0.0), dtype=tf.float32)) * (lidar_short + 1e-9)**-1, axis=[1], name='lidar_short_inv')
    _ = tf.reverse(mask, axis=[1], name='mask')
    _ = tf.identity(tf.reduce_mean(input_tensor=loss_phot_), name="loss_phot")
    _ = tf.identity(tf.reduce_mean(input_tensor=loss_lidar_), name="loss_lidar")
    _ = tf.identity(tf.reduce_mean(input_tensor=loss_lidar_short_), name="loss_lidar_short")
    _ = tf.identity(reg_loss, name="reg_loss")
    _ = tf.identity(loss_, name="loss")

    return loss_

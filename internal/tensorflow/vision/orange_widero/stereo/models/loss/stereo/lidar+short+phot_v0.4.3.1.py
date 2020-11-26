import tensorflow as tf
from stereo.models.loss_utils import min_photo_loss, loss_image, loss_lidar
from stereo.interfaces.implements import implements_format


loss_format = ("loss", "inp_views_short")
@implements_format(*loss_format)
def loss(out, inp , T_cntr_srnd, focal, origin, im_lidar, im_lidar_short, im_mask, x, y,
         measure='ssim', radius=1, alpha_ooi=0.0, alpha_lidar=1.0, alpha_lidar_short=1.0, reg_constant=1.0,  blur_kernel_names=None):

    Z_inv_cntr = out

    I_cntr = tf.expand_dims(inp[:, 0, :, :], 3)
    if blur_kernel_names is not None:
        with tf.compat.v1.variable_scope('consts', reuse=True):
            blur_kernel_0 = tf.compat.v1.get_variable(blur_kernel_names[0])
            blur_kernel_1 = tf.compat.v1.get_variable(blur_kernel_names[1])
            blur_kernel_2 = tf.compat.v1.get_variable(blur_kernel_names[2])
        I_cntr_conv_0 = tf.nn.conv2d(input=I_cntr, filters=blur_kernel_0,
                                      strides=[1,1,1,1], padding='SAME')
        I_cntr_conv_1 = tf.nn.conv2d(input=I_cntr, filters=blur_kernel_1,
                                      strides=[1,1,1,1], padding='SAME')
        I_cntr_conv_2 = tf.nn.conv2d(input=I_cntr_conv_0, filters=blur_kernel_2,
                                     strides=[1, 1, 1, 1], padding='SAME')
    else:
        I_cntr_conv_0 = tf.identity(I_cntr)
        I_cntr_conv_1 = tf.identity(I_cntr)
        I_cntr_conv_2 = tf.identity(I_cntr)
    I_srnd_0 = tf.expand_dims(inp[:, 1, :, :], 3)
    I_srnd_1 = tf.expand_dims(inp[:, 2, :, :], 3)
    I_srnd_2 = tf.expand_dims(inp[:, 3, :, :], 3)



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

    loss_phot_ = 2 * min_photo_loss([loss_im_0, loss_im_1, loss_im_2], [mask_ioi_0, mask_ioi_1, mask_ioi_2], radius=radius)
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

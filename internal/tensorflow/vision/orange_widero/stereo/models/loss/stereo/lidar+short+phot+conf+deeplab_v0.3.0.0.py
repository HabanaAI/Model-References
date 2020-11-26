import tensorflow as tf
from stereo.models.layers.clamp_layer import Clamp
from stereo.models.loss_utils import loss_image, loss_lidar, min_photo_loss, batch_conv2d, loss_conf
from stereo.models.loss_utils import Z_inv_of_tapered_Z_inv
from stereo.interfaces.implements import implements_format


loss_format = ("loss", "four_views_deeplab_photo_mask_cropped")
@implements_format(*loss_format)
def loss(out, I_cntr, I_srnd_0, I_srnd_1, I_srnd_2, deeplab, T_cntr_srnd, focal, origin,
         im_lidar_inv, im_lidar_short_inv, im_mask, photo_loss_im_mask, x, y, blur_kernels,
         taper_Z=20.0, min_Z=1.0, max_Z=1000.0,
         phot_measure='ssim', phot_radius=1,
         alpha_phot=1.0, alpha_lidar=1.0, alpha_lidar_short=1.0, alpha_lidar_sky=1.0,
         alpha_conf=1.0, alpha_conf_short=1.0, error_stop_grad=False,
         keras_train=False):

    print("out keys: ", [k for k in out.keys()])
    pred_tapered_inv_depth = Clamp(min_val=(max_Z + taper_Z) ** -1, max_val=(min_Z + taper_Z) ** -1)(out['out'])
    pred_inv_depth = Z_inv_of_tapered_Z_inv(pred_tapered_inv_depth, taper_Z)
    pred_error = out['out_conf']

    I_cntr_conv_0 = batch_conv2d(I_cntr, tf.expand_dims(blur_kernels[:, :, :, 0], -1))
    I_cntr_conv_1 = batch_conv2d(I_cntr, tf.expand_dims(blur_kernels[:, :, :, 1], -1))
    I_cntr_conv_2 = batch_conv2d(I_cntr, tf.expand_dims(blur_kernels[:, :, :, 2], -1))

    I_warp_cntr_0, loss_im_0, mask_ioi_0, mean_ioi_0, mean_ooi_0 = loss_image(I1=I_cntr_conv_0, I2=I_srnd_0,
                                                                              Z1_inv=pred_inv_depth, focal=focal,
                                                                              origin=origin,
                                                                              RT12=T_cntr_srnd[:, 0, :], x1=x, y1=y,
                                                                              measure=phot_measure, radius=phot_radius,
                                                                              cut_off=None, mask1=photo_loss_im_mask,
                                                                              ooi_on_mask=True)

    I_warp_cntr_1, loss_im_1, mask_ioi_1, mean_ioi_1, mean_ooi_1 = loss_image(I1=I_cntr_conv_1, I2=I_srnd_1,
                                                                              Z1_inv=pred_inv_depth, focal=focal,
                                                                              origin=origin,
                                                                              RT12=T_cntr_srnd[:, 1, :], x1=x, y1=y,
                                                                              measure=phot_measure, radius=phot_radius,
                                                                              cut_off=None, mask1=photo_loss_im_mask,
                                                                              ooi_on_mask=True)

    I_warp_cntr_2, loss_im_2, mask_ioi_2, mean_ioi_2, mean_ooi_2 = loss_image(I1=I_cntr_conv_2, I2=I_srnd_2,
                                                                              Z1_inv=pred_inv_depth, focal=focal,
                                                                              origin=origin,
                                                                              RT12=T_cntr_srnd[:, 2, :], x1=x, y1=y,
                                                                              measure=phot_measure, radius=phot_radius,
                                                                              cut_off=None, mask1=photo_loss_im_mask,
                                                                              ooi_on_mask=True)

    loss_phot_ = 2 * min_photo_loss([loss_im_0, loss_im_1, loss_im_2], [mask_ioi_0, mask_ioi_1, mask_ioi_2],
                                    radius=phot_radius)
    loss_phot_ = tf.identity(loss_phot_, name="batch_photo")

    loss_lidar_ = loss_lidar(pred_inv_depth, None, im_mask, inv_lidar=im_lidar_inv)
    loss_lidar_ = tf.identity(loss_lidar_ * 40.0, "batch_lidar")

    loss_lidar_short_ = loss_lidar(pred_inv_depth, None, tf.zeros_like(im_mask), inv_lidar=im_lidar_short_inv)
    loss_lidar_short_ = tf.identity(loss_lidar_short_ * 40.0, "batch_lidar_short")

    sky_inv_depth = tf.cast(tf.equal(deeplab, 10), dtype=tf.float32) * 1e-4
    loss_lidar_sky_ = loss_lidar(pred_inv_depth, None, None, inv_lidar=sky_inv_depth)
    loss_lidar_sky_ = tf.identity(loss_lidar_sky_ * 40.0, "batch_lidar_sky")

    loss_conf_, real_error_ = \
        loss_conf(pred_inv_depth, im_lidar_inv, pred_error, mask=im_mask,
                  stop_grad=error_stop_grad, mean_ioi_per_sample=True)
    loss_conf_short_, real_error_short_ = \
        loss_conf(pred_inv_depth, im_lidar_short_inv, pred_error, mask=tf.zeros_like(im_mask),
                  stop_grad=error_stop_grad, mean_ioi_per_sample=True)

    if not keras_train:
        reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        loss_reg = tf.reduce_sum(input_tensor=reg_losses)
    else:
        loss_reg = 0

    batch_loss = tf.identity(loss_phot_ * alpha_phot +
                             loss_lidar_ * alpha_lidar + 
                             loss_lidar_short_ * alpha_lidar_short +
                             loss_lidar_sky_ * alpha_lidar_sky +
                             loss_conf_ * alpha_conf +
                             loss_conf_short_ * alpha_conf_short,
                             name="batch_loss")

    loss_ = tf.reduce_mean(input_tensor=batch_loss) + loss_reg

    # prepare and name additional tensors to be summarized
    _ = tf.identity(I_warp_cntr_0, name='I_warp_cntr_0')
    _ = tf.identity(I_warp_cntr_1, name='I_warp_cntr_1')
    _ = tf.identity(I_warp_cntr_2, name='I_warp_cntr_2')
    _ = tf.identity(real_error_, name='real_error')
    _ = tf.identity(real_error_short_, name='real_error_short')
    _ = tf.reduce_mean(input_tensor=loss_phot_, name="loss_phot")
    _ = tf.reduce_mean(input_tensor=loss_lidar_, name="loss_lidar")
    _ = tf.reduce_mean(input_tensor=loss_lidar_short_, name="loss_lidar_short")
    _ = tf.reduce_mean(input_tensor=loss_lidar_sky_, name="loss_lidar_sky")
    _ = tf.reduce_mean(input_tensor=loss_conf_, name="loss_conf")
    _ = tf.reduce_mean(input_tensor=loss_conf_short_, name="loss_conf_short")
    _ = tf.identity(loss_reg, name="loss_reg")

    return loss_

import tensorflow as tf
import numpy as np
from stereo.models.loss_utils import loss_image, sum_photo_loss


def loss(out, invZ, I_cntr, T_cntr_srnd, focal, origin, photo_loss_im_mask,
         measure='ssim', radius=1, reg_constant=1.0):

    invZ = tf.expand_dims(invZ, -1)
    im_sz = I_cntr.shape.as_list()[1:3]
    origin_x, origin_y = np.float32(im_sz[1] / 2), np.float32(im_sz[0] / 2)
    left, right = -origin_x, np.float32(im_sz[1]) - origin_x
    bottom, top = -origin_y, np.float32(im_sz[0]) - origin_y

    x, y = tf.meshgrid(tf.range(left, right), tf.range(bottom, top))

    x = tf.tile(tf.expand_dims(tf.expand_dims(x, 0), 3), [tf.shape(input=I_cntr)[0], 1, 1, 1])
    y = tf.tile(tf.expand_dims(tf.expand_dims(y, 0), 3), [tf.shape(input=I_cntr)[0], 1, 1, 1])

    I_srnd_0 = out['I_srnd_0']
    I_srnd_1 = out['I_srnd_1']
    I_srnd_2 = out['I_srnd_2']

    I_warp_cntr_0, loss_im_0, mask_ioi_0, mean_ioi_0, mean_ooi_0 = loss_image(I1=I_cntr, I2=I_srnd_0,
                                                                              Z1_inv=invZ, focal=focal,
                                                                              origin=origin,
                                                                              RT12=T_cntr_srnd[:, 0, :], x1=x, y1=y,
                                                                              measure=measure, radius=radius,
                                                                              cut_off=None, mask1=photo_loss_im_mask,
                                                                              ooi_on_mask=True)

    I_warp_cntr_1, loss_im_1, mask_ioi_1, mean_ioi_1, mean_ooi_1 = loss_image(I1=I_cntr, I2=I_srnd_1,
                                                                              Z1_inv=invZ, focal=focal,
                                                                              origin=origin,
                                                                              RT12=T_cntr_srnd[:, 1, :], x1=x, y1=y,
                                                                              measure=measure, radius=radius,
                                                                              cut_off=None, mask1=photo_loss_im_mask,
                                                                              ooi_on_mask=True)

    I_warp_cntr_2, loss_im_2, mask_ioi_2, mean_ioi_2, mean_ooi_2 = loss_image(I1=I_cntr, I2=I_srnd_2,
                                                                              Z1_inv=invZ, focal=focal,
                                                                              origin=origin,
                                                                              RT12=T_cntr_srnd[:, 2, :], x1=x, y1=y,
                                                                              measure=measure, radius=radius,
                                                                              cut_off=None, mask1=photo_loss_im_mask,
                                                                              ooi_on_mask=True)

    _ = tf.identity(loss_im_0 * mask_ioi_0, name="ssim_im_0")
    _ = tf.identity(loss_im_1 * mask_ioi_1, name="ssim_im_1")
    _ = tf.identity(loss_im_2 * mask_ioi_2, name="ssim_im_2")

    loss_phot_ = sum_photo_loss([loss_im_0, loss_im_1, loss_im_2], [mask_ioi_0, mask_ioi_1, mask_ioi_2])
    loss_phot_ = tf.identity(loss_phot_, name="batch_photo")
    reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
    loss_reg = reg_constant * tf.reduce_sum(input_tensor=reg_losses)
    loss_ = tf.reduce_mean(input_tensor=loss_phot_) + loss_reg
    _ = tf.identity(tf.reduce_mean(input_tensor=loss_phot_), name="loss_phot")
    _ = tf.identity(loss_reg, name="loss_reg")

    return loss_

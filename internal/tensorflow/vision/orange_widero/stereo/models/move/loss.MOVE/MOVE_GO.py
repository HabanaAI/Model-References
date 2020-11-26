import numpy as np
import tensorflow as tf
from stereo.models.MOVE_utils import loss_geom_motion, loss_phot_motion, advanced_object_consistency_loss


def loss(out, curr_inp, prev_inp, next_inp, prev_curr2frame, next_curr2frame, curr_T_cntr_srnd,
         prev_T_cntr_srnd, next_T_cntr_srnd, curr_focal,
         curr_origin, curr_im_mask, prev_im_mask, next_im_mask, curr_instance_mask,
         prev_instance_mask, next_instance_mask, curr_instance_ids, prev_instance_ids,
         next_instance_ids, curr_seg_im, prev_seg_im, next_seg_im, curr_vcl, prev_vcl, next_vcl,
         x, y, reg_constant=1.0):

    with tf.compat.v1.name_scope("Loss_"):

        curr_motion = out[0]
        Z_inv2_cntr = out[1]
        Z_inv1_cntr = out[2]
        motion_input1 = out[3]
        motion_input2 = out[4]
        motion_input3 = out[5]
        motion_input4 = out[6]

        curr_focal = tf.reshape(curr_focal[:, 0], [-1, 1, 1, 1])

        I1_cntr = tf.expand_dims(prev_inp[:, 0, :, :], 3)
        I2_cntr = tf.expand_dims(curr_inp[:, 0, :, :], 3)

        curr_im_mask = tf.expand_dims(curr_im_mask, 3)
        curr_instance_mask = tf.expand_dims(curr_instance_mask, 3)
        curr_seg_im = tf.expand_dims(curr_seg_im, 3)

        seg_mask = tf.compat.v1.where(tf.logical_and(tf.greater(curr_seg_im, 3.), tf.not_equal(curr_seg_im, 6.)),
                                tf.ones_like(curr_seg_im), tf.zeros_like(curr_seg_im))

        motion_masked_no_y = tf.concat(
            [tf.expand_dims(curr_motion[:, :, :, 0], 3), tf.zeros_like(tf.expand_dims(curr_motion[:, :, :, 1], 3)),
             tf.expand_dims(curr_motion[:, :, :, 2], 3)], axis=3)


        control_loss, I_warp_control, control_loss_im = loss_phot_motion(I1_cntr, I2_cntr, Z_inv2_cntr,
                                                                         tf.zeros_like(curr_motion), curr_focal,
                                                                         curr_origin, prev_curr2frame, x, y,
                                                                         mask1=seg_mask)

        loss_photo_motion, I_warp_motion, motion_loss_im = loss_phot_motion(I1_cntr, I2_cntr, Z_inv2_cntr, motion_masked_no_y,
                                                                            curr_focal, curr_origin, prev_curr2frame,
                                                                            x, y, mask1=seg_mask)

        loss_geo_X, loss_geo_Y, loss_geo_Z, Z_inv2_warp, Z_inv2, Z_diff_im, X_diff_im, Y_diff_im = loss_geom_motion(
            Z_inv1_cntr, Z_inv2_cntr,
            motion_masked_no_y, curr_focal, curr_origin,
            tf.linalg.inv(prev_curr2frame),
            x, y, mask2=seg_mask, return_diff_ims=True)

        loss_object, object_loss_Z, direct_loss_X, direct_loss_Z, sum_direct = advanced_object_consistency_loss(motion_masked_no_y,
                                                                                          curr_instance_mask,
                                                                                          curr_instance_ids,
                                                                                          curr_seg_im,
                                                                                          curr_vcl)

        loss_geo_motion = loss_geo_Z

        # static_motion_loss = 0.
        seg_not_mask = tf.compat.v1.where(tf.greater(curr_seg_im, 6.), tf.zeros_like(curr_seg_im), tf.ones_like(curr_seg_im))
        static_motion_loss = tf.reduce_mean(input_tensor=curr_motion * seg_not_mask)
        control_loss *= 10.

        loss_motion_ = loss_geo_motion + loss_object

        reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = reg_constant * sum(reg_losses)

        loss_ = loss_motion_ * 100.
        # loss_ = direct_loss_X * 100. + direct_loss_Z
        # loss_ *= 10.

        motion_gradient = tf.gradients(ys=loss_, xs=curr_motion)[0]

        # prepare and name tensors to be summarized
        _ = tf.reverse(Z_inv1_cntr, axis=[1], name='Z_inv1_cntr')
        _ = tf.reverse(Z_inv2_cntr, axis=[1], name='Z_inv2_cntr')
        _ = tf.reverse(motion_input1, axis=[1], name='motion_input1')
        _ = tf.reverse(motion_input2, axis=[1], name='motion_input2')
        _ = tf.reverse(motion_input3, axis=[1], name='motion_input3')
        _ = tf.reverse(motion_input4, axis=[1], name='motion_input4')
        _ = tf.reverse(I_warp_motion, axis=[1], name='I_warp_motion')
        _ = tf.reverse(I_warp_control, axis=[1], name='I_warp_control')
        _ = tf.reverse(tf.expand_dims(motion_gradient[:, :, :, 0], 3), axis=[1], name='gradient_x')
        _ = tf.reverse(tf.expand_dims(motion_gradient[:, :, :, 1], 3), axis=[1], name='gradient_y')
        _ = tf.reverse(tf.expand_dims(motion_gradient[:, :, :, 2], 3), axis=[1], name='gradient_z')
        _ = tf.reverse(tf.abs(motion_input1 - motion_input2), axis=[1], name='motion_input_diff')
        _ = tf.reverse(tf.expand_dims(curr_motion[:, :, :, 0] * 5., 3), axis=[1], name='X_motion')
        _ = tf.reverse(tf.expand_dims(curr_motion[:, :, :, 1] * 5., 3), axis=[1], name='Y_motion')
        _ = tf.reverse(tf.expand_dims(curr_motion[:, :, :, 2] * 5., 3), axis=[1], name='Z_motion')
        _ = tf.reverse(curr_instance_mask, axis=[1], name='curr_instance_mask')
        _ = tf.reverse(motion_loss_im, axis=[1], name='motion_loss_im')
        _ = tf.reverse(control_loss_im, axis=[1], name='control_loss_im')
        _ = tf.reverse(I1_cntr, axis=[1], name='I1_cntr')
        _ = tf.reverse(I2_cntr, axis=[1], name='I2_cntr')
        _ = tf.reverse(curr_im_mask, axis=[1], name='mask2')
        _ = tf.reverse(Z_diff_im, axis=[1], name='Z_diff_im')
        _ = tf.reverse(X_diff_im, axis=[1], name='X_diff_im')
        _ = tf.reverse(Y_diff_im, axis=[1], name='Y_diff_im')
        _ = tf.reverse(object_loss_Z, axis=[1], name='object_loss_Z')
        _ = tf.reverse(tf.expand_dims(sum_direct[:,:,:,0], 3), axis=[1], name='sum_direct_x')
        _ = tf.reverse(tf.expand_dims(sum_direct[:,:,:,2], 3), axis=[1], name='sum_direct_z')
        _ = tf.identity(loss_motion_, name="loss_motion_full")
        _ = tf.identity(loss_geo_motion, name='loss_motion_geo')
        _ = tf.identity(loss_geo_X, name='loss_geo_X')
        _ = tf.identity(loss_geo_Y, name='loss_geo_Y')
        _ = tf.identity(loss_geo_Z, name='loss_geo_Z')
        _ = tf.identity(loss_photo_motion, name='loss_motion_photo')
        _ = tf.identity(static_motion_loss, name='loss_motion_static')
        _ = tf.identity(control_loss, name="loss_control")
        _ = tf.identity(reg_loss, name="reg_loss")
        _ = tf.identity(direct_loss_X, name='direct_loss_X')
        _ = tf.identity(direct_loss_Z, name='direct_loss_Z')
        _ = tf.identity(loss_, name="loss")
        _ = tf.identity(tf.reduce_max(input_tensor=motion_gradient), name="max_gradient")
        _ = tf.identity(tf.reduce_min(input_tensor=motion_gradient), name="min_gradient")
        _ = tf.identity(loss_object, name="loss_object")

    return loss_
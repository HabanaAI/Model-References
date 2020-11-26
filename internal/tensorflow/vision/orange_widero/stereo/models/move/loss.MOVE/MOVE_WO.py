import numpy as np
import tensorflow as tf
from stereo.models.MOVE_utils import warp_loss, object_consistency_loss


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
        prev_instance_mask = tf.expand_dims(prev_instance_mask, 3)
        curr_seg_im = tf.expand_dims(curr_seg_im, 3)

        motion_masked_no_y = tf.concat(
            [tf.expand_dims(curr_motion[:, :, :, 0], 3), tf.zeros_like(tf.expand_dims(curr_motion[:, :, :, 0], 3)),
             tf.expand_dims(curr_motion[:, :, :, 2], 3)], axis=3)


        object_loss, Z_loss_sum = object_consistency_loss(motion_masked_no_y, curr_instance_mask, curr_instance_ids, curr_seg_im)

        warp_obj_loss = warp_loss(prev_instance_mask, curr_instance_mask, Z_inv2_cntr, motion_masked_no_y, curr_focal,
                              curr_origin, tf.linalg.inv(prev_curr2frame), x, y)

        seg_not_mask = tf.compat.v1.where(tf.logical_and(tf.greater(curr_seg_im, 3.), tf.not_equal(curr_seg_im, 6.)),
                                tf.zeros_like(curr_seg_im), tf.ones_like(curr_seg_im))
        static_motion_loss = tf.reduce_mean(input_tensor=tf.abs(curr_motion) * seg_not_mask)


        reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = reg_constant * sum(reg_losses)

        loss_ = warp_obj_loss * 200. + object_loss + static_motion_loss
        loss_ *= 10.

        motion_gradient = tf.gradients(ys=loss_, xs=curr_motion)[0]

        # prepare and name tensors to be summarized
        _ = tf.reverse(Z_inv1_cntr, axis=[1], name='Z_inv1_cntr')
        _ = tf.reverse(Z_inv2_cntr, axis=[1], name='Z_inv2_cntr')
        _ = tf.reverse(motion_input1, axis=[1], name='motion_input1')
        _ = tf.reverse(motion_input2, axis=[1], name='motion_input2')
        _ = tf.reverse(motion_input3, axis=[1], name='motion_input3')
        _ = tf.reverse(motion_input4, axis=[1], name='motion_input4')
        _ = tf.reverse(tf.expand_dims(motion_gradient[:, :, :, 0], 3), axis=[1], name='gradient_x')
        _ = tf.reverse(tf.expand_dims(motion_gradient[:, :, :, 1], 3), axis=[1], name='gradient_y')
        _ = tf.reverse(tf.expand_dims(motion_gradient[:, :, :, 2], 3), axis=[1], name='gradient_z')
        _ = tf.reverse(tf.abs(motion_input1 - motion_input2), axis=[1], name='motion_input_diff')
        _ = tf.reverse(tf.expand_dims(curr_motion[:, :, :, 0] * 5., 3), axis=[1], name='X_motion')
        _ = tf.reverse(tf.expand_dims(curr_motion[:, :, :, 1] * 5., 3), axis=[1], name='Y_motion')
        _ = tf.reverse(tf.expand_dims(curr_motion[:, :, :, 2] * 5., 3), axis=[1], name='Z_motion')
        _ = tf.reverse(curr_instance_mask, axis=[1], name='curr_instance_mask')
        _ = tf.reverse(I1_cntr, axis=[1], name='I1_cntr')
        _ = tf.reverse(I2_cntr, axis=[1], name='I2_cntr')
        _ = tf.reverse(curr_im_mask, axis=[1], name='mask2')
        _ = tf.reverse(seg_not_mask, axis=[1], name='seg_not_mask')
        _ = tf.reverse(Z_loss_sum, axis=[1], name='object_diff')
        _ = tf.identity(warp_obj_loss, name='warp_obj_loss')
        _ = tf.identity(static_motion_loss, name='loss_motion_static')
        _ = tf.identity(object_loss, name='object_loss')
        _ = tf.identity(reg_loss, name="reg_loss")
        _ = tf.identity(loss_, name="loss")
        _ = tf.identity(tf.reduce_max(input_tensor=motion_gradient), name="max_gradient")
        _ = tf.identity(tf.reduce_min(input_tensor=motion_gradient), name="min_gradient")

    return loss_
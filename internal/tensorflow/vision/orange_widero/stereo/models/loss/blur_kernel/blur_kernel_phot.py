import numpy as np
import tensorflow as tf
from stereo.models.loss_utils import ssim, ncc
from stereo.interfaces.implements import implements_format


loss_format = ("loss", "blur_kernel")
@implements_format(*loss_format)
def loss(out,
         frontCornerRight_to_main,
         frontCornerLeft_to_main,
         parking_front_to_main,
         parking_front_to_frontCornerLeft,
         parking_left_to_frontCornerLeft,
         parking_right_to_frontCornerRight,
         parking_front_to_frontCornerRight,
         parking_left_to_rearCornerLeft,
         parking_rear_to_rearCornerLeft,
         parking_rear_to_rearCornerRight,
         parking_right_to_rearCornerRight,
         rearCornerRight_to_rear,
         rearCornerLeft_to_rear,
         parking_rear_to_rear,
         main_to_main_mask,
         frontCornerLeft_to_frontCornerLeft_mask,
         frontCornerRight_to_frontCornerRight_mask,
         rearCornerLeft_to_rearCornerLeft_mask,
         rearCornerRight_to_rearCornerRight_mask,
         rear_to_rear_mask,
         measure='ssim', radius=1):


    (main_to_frontCornerLeft, main_to_frontCornerRight, main_to_parking_front,
     frontCornerLeft_to_parking_front, frontCornerLeft_to_parking_left,
     frontCornerRight_to_parking_front, frontCornerRight_to_parking_right,
     rearCornerLeft_to_parking_rear, rearCornerLeft_to_parking_left,
     rearCornerRight_to_parking_rear, rearCornerRight_to_parking_right,
     rear_to_rearCornerLeft, rear_to_rearCornerRight, rear_to_parking_rear) = tf.split(out, 14, axis=1)

    tens = {'main_to_frontCornerLeft': main_to_frontCornerLeft,
            'main_to_frontCornerRight': main_to_frontCornerRight,
            'main_to_parking_front': main_to_parking_front,
            'frontCornerLeft_to_parking_front': frontCornerLeft_to_parking_front,
            'frontCornerLeft_to_parking_left': frontCornerLeft_to_parking_left,
            'frontCornerRight_to_parking_front': frontCornerRight_to_parking_front,
            'frontCornerRight_to_parking_right': frontCornerRight_to_parking_right,
            'rearCornerLeft_to_parking_rear': rearCornerLeft_to_parking_rear,
            'rearCornerLeft_to_parking_left': rearCornerLeft_to_parking_left,
            'rearCornerRight_to_parking_rear': rearCornerRight_to_parking_rear,
            'rearCornerRight_to_parking_right': rearCornerRight_to_parking_right,
            'rear_to_rearCornerLeft': rear_to_rearCornerLeft,
            'rear_to_rearCornerRight': rear_to_rearCornerRight,
            'rear_to_parking_rear': rear_to_parking_rear}

    pairs = {'main_to_frontCornerLeft': frontCornerLeft_to_main,
             'main_to_frontCornerRight': frontCornerRight_to_main,
             'main_to_parking_front': parking_front_to_main,
             'frontCornerLeft_to_parking_front': parking_front_to_frontCornerLeft,
             'frontCornerLeft_to_parking_left': parking_left_to_frontCornerLeft,
             'frontCornerRight_to_parking_right': parking_right_to_frontCornerRight,
             'frontCornerRight_to_parking_front': parking_front_to_frontCornerRight,
             'rearCornerLeft_to_parking_rear': parking_rear_to_rearCornerLeft,
             'rearCornerLeft_to_parking_left': parking_left_to_rearCornerLeft,
             'rearCornerRight_to_parking_rear': parking_rear_to_rearCornerRight,
             'rearCornerRight_to_parking_right': parking_right_to_rearCornerRight,
             'rear_to_rearCornerLeft': rearCornerRight_to_rear,
             'rear_to_rearCornerRight': rearCornerLeft_to_rear,
             'rear_to_parking_rear': parking_rear_to_rear}

    masks = {'main_to_frontCornerLeft': 1-main_to_main_mask,
             'main_to_frontCornerRight': 1-main_to_main_mask,
             'main_to_parking_front': 1-main_to_main_mask,
             'frontCornerLeft_to_parking_front': 1 - frontCornerLeft_to_frontCornerLeft_mask,
             'frontCornerLeft_to_parking_left': 1 - frontCornerLeft_to_frontCornerLeft_mask,
             'frontCornerRight_to_parking_right': 1 - frontCornerRight_to_frontCornerRight_mask,
             'frontCornerRight_to_parking_front': 1 - frontCornerRight_to_frontCornerRight_mask,
             'rearCornerLeft_to_parking_rear': 1 - rearCornerLeft_to_rearCornerLeft_mask,
             'rearCornerLeft_to_parking_left': 1 - rearCornerLeft_to_rearCornerLeft_mask,
             'rearCornerRight_to_parking_rear': 1 - rearCornerRight_to_rearCornerRight_mask,
             'rearCornerRight_to_parking_right': 1 - rearCornerRight_to_rearCornerRight_mask,
             'rear_to_rearCornerLeft': 1 - rear_to_rear_mask,
             'rear_to_rearCornerRight': 1 - rear_to_rear_mask,
             'rear_to_parking_rear': 1 - rear_to_rear_mask}


    losses = []
    for blurred in tens.keys():
        mask_ioi = masks[blurred]*tf.cast(pairs[blurred] > 0., tf.float32)
        mean_ioi = tf.maximum(tf.reduce_mean(input_tensor=mask_ioi), 1e-5)
        im1 = tf.transpose(a=tens[blurred], perm=[0,2,3,1])
        im2 = tf.transpose(a=pairs[blurred], perm=[0,2,3,1])
        if measure == 'ssim':
            loss_im = ssim(im1, im2, radius=radius)
        else:
            assert measure == 'ncc'
            loss_im = 1-ncc(im1, im2, radius=radius)
        loss_im = tf.transpose(a=loss_im, perm=[0,3,1,2])
        loss_i = tf.reduce_mean(input_tensor=loss_im * mask_ioi) / mean_ioi
        _ = tf.identity(loss_i, name="loss_"+blurred)
        losses.append(loss_i)
        _ = tf.reverse(tf.transpose(a=tens[blurred], perm=[0, 2, 3, 1]), axis=[1], name=blurred+'_im')
        _ = tf.transpose(a=get_variable(blurred+'/kernel:0'), perm=[2,0,1,3], name=blurred+'_kernel')

    loss_ = tf.reduce_sum(input_tensor=losses)

    _ = tf.identity(loss_, name="loss")


    return loss_

def get_variable(name):
    return [v for v in tf.compat.v1.global_variables() if v.name == name][0]
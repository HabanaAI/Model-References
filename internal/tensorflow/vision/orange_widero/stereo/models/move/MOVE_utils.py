import tensorflow as tf
import numpy as np
from stereo.models.loss_utils import transform3_tf

def motion_warp(Ibase, Zcurr_inv, M, focal, origin, RTcurr2base, xcurr, ycurr):

    motion_factor = 5.
    M *= motion_factor

    Zcurr = Zcurr_inv**-1
    Xcurr = (xcurr * Zcurr) / focal
    Ycurr = (ycurr * Zcurr) / focal

    if len(RTcurr2base.shape) == 2:
        Xbase = Xcurr + tf.expand_dims(tf.expand_dims(tf.expand_dims(RTcurr2base[:, 0], 1), 2), 3)
        Ybase = Ycurr + tf.expand_dims(tf.expand_dims(tf.expand_dims(RTcurr2base[:, 1], 1), 2), 3)
        Zbase = Zcurr + tf.expand_dims(tf.expand_dims(tf.expand_dims(RTcurr2base[:, 2], 1), 2), 3)
    elif len(RTcurr2base.shape) == 3:
        Xbase, Ybase, Zbase = transform3_tf(Xcurr, Ycurr, Zcurr, RTcurr2base)
    else:
        raise Exception('Unsupported RTcurr2base shape')

    Xbase -= tf.expand_dims(M[:,:,:,0], 3)
    Ybase -= tf.expand_dims(M[:,:,:,1], 3)
    Zbase -= tf.expand_dims(M[:,:,:,2], 3)

    Zbase_inv = Zbase**-1
    xbase = (Xbase * focal) * Zbase_inv + origin[0, 0]
    ybase = (Ybase * focal) * Zbase_inv + origin[0, 1]

    xybase = tf.concat([xbase, ybase], 3)
    # Ibase_warped_to_Icurr = tf.contrib.resampler.resampler(Ibase, xybase)

    Ibase_warped_to_Icurr, ooi = bilinear_sampler(Ibase, xybase)
    return Ibase_warped_to_Icurr, ooi

def prep_motion2(Z_inv1, Z_inv2, focal, origin, RT12, x, y):

    Z1 = Z_inv1**-1
    X1 = (x * Z1) / focal
    Y1 = (y * Z1) / focal
    X1_, Y1_, Z1_ = transform3_tf(X1, Y1, Z1, RT12)
    Z_inv1_ = Z1_**-1
    # is this potentially a problem?

    x1 = (X1_ * focal * Z_inv1_) + origin[0,0]
    y1 = (Y1_ * focal * Z_inv1_) + origin[0,1]
    xy1 = tf.concat([x1,y1], 3)
    # Z_inv2_w = tf.contrib.resampler.resampler(Z_inv2, xy1)
    Z_inv2_w, ooi = bilinear_sampler(Z_inv2, xy1)

    Z2_w_1 = Z_inv2_w**-1
    X2 = (x * Z2_w_1) / focal
    Y2 = (y * Z2_w_1) / focal
    X2_,Y2_,Z2_ = transform3_tf(X2, Y2, Z2_w_1, tf.linalg.inv(RT12))

    Z2_inv_to1 = Z2_**-1

    Z2_inv_to1 = tf.compat.v1.where(tf.equal(ooi, 1.), tf.zeros_like(Z2_inv_to1), Z2_inv_to1)

    return Z2_inv_to1, Z_inv1

def prep_motion_ims(Z_inv2, inp1, inp2, focal, origin, RT21, x, y):

    Z2 = Z_inv2 ** -1
    X2 = (x * Z2) / focal
    Y2 = (y * Z2) / focal
    X2_, Y2_, Z2_ = transform3_tf(X2, Y2, Z2, RT21)
    Z_inv2_ = Z2_ ** -1

    x2 = (X2_ * focal) * Z_inv2_ + origin[0, 0]  # notice that x2, y2 isn't relative to origin
    y2 = (Y2_ * focal) * Z_inv2_ + origin[0, 1]  # while x1, y1 are
    xy2 = tf.concat([x2, y2], 3)
    # im2_w = tf.contrib.resampler.resampler(tf.expand_dims(inp1[:,0,:,:], 3), xy2)

    # im2_w, ooi = bilinear_sampler(tf.expand_dims(inp1[:,0,:,:], 3), xy2)
    im2_w, ooi = bilinear_sampler(tf.expand_dims(inp1[:,:,:,0], 3), xy2)

    im2_w = tf.compat.v1.where(tf.equal(ooi, 1.), tf.zeros_like(im2_w), im2_w)

    # return tf.expand_dims(inp2[:,0,:,:], 3), im2_w
    return tf.expand_dims(inp2[:,:,:,0], 3), im2_w

def loss_geom_motion(Z_inv1, Z_inv2, motion, focal, origin, RT12, x, y, measure='abs', alpha_ooi=0.1, mask2=None,
                     return_diff_ims=False, clip_G=False):

    motion_factor = 5.

    assert(measure == 'abs')
    RT21 = tf.linalg.inv(RT12)

    Z2_orig = Z_inv2 ** -1
    X2_orig = (x * Z2_orig) / focal
    Y2_orig = (y * Z2_orig) / focal
    X2 = X2_orig - tf.expand_dims(motion[:,:,:,0], 3) * motion_factor
    Y2 = Y2_orig - tf.expand_dims(motion[:,:,:,1], 3) * motion_factor
    Z2 = Z2_orig - tf.expand_dims(motion[:,:,:,2], 3) * motion_factor

    X2_, Y2_, Z2_ = transform3_tf(X2, Y2, Z2, RT21)
    Z_inv2_ = Z2_ ** -1

    x2 = (X2_ * focal) * Z_inv2_ + origin[0, 0]  # notice that x2, y2 isn't relative to origin
    y2 = (Y2_ * focal) * Z_inv2_ + origin[0, 1]  # while x1, y1 are
    xy2 = tf.concat([x2, y2], 3)
    # Z_inv1_w = tf.contrib.resampler.resampler(Z_inv1, xy2)
    Z_inv1_w, ooi = bilinear_sampler(Z_inv1, xy2)

    Z_inv1_w = tf.maximum(tf.minimum(0.5, Z_inv1_w), .0001)

    Z1_w = Z_inv1_w ** -1
    X1_w = (x * Z1_w) / focal
    Y1_w = (y * Z1_w) / focal

    X2_warp, Y2_warp, Z2_warp = transform3_tf(X1_w, Y1_w, Z1_w, RT12)

    X2_warp += tf.expand_dims(motion[:, :, :, 0], 3) * motion_factor
    Y2_warp += tf.expand_dims(motion[:, :, :, 1], 3) * motion_factor
    Z2_warp += tf.expand_dims(motion[:, :, :, 2], 3) * motion_factor

    X2_warp = tf.compat.v1.where(tf.equal(X2_warp, 0), tf.ones_like(X2_warp)*.001, X2_warp)
    Y2_warp = tf.compat.v1.where(tf.equal(Y2_warp, 0), tf.ones_like(Y2_warp)*.001, Y2_warp)
    X2_warp = tf.maximum(tf.minimum(15., X2_warp), -15.)
    Y2_warp = tf.maximum(tf.minimum(20.,Y2_warp), -5.)
    #
    X2_orig = tf.compat.v1.where(tf.equal(X2_orig, 0), tf.ones_like(X2_orig) * .001, X2_orig)
    Y2_orig = tf.compat.v1.where(tf.equal(Y2_orig, 0), tf.ones_like(Y2_orig) * .001, Y2_orig)
    X2_orig = tf.maximum(tf.minimum(15., X2_orig), -15.)
    Y2_orig = tf.maximum(tf.minimum(20., Y2_orig), -5.)

    Z2_warp = tf.maximum(tf.minimum(1000., Z2_warp), 2.)

    Z_inv_warp = Z2_warp ** -1

    if mask2 is None:
        mask2 = tf.ones_like(Z_inv1)

    mask2 *= (1-ooi)

    sum_mask2 = tf.maximum(tf.reduce_sum(input_tensor=mask2), 1)

    if clip_G:
        loss_Z = tf.abs(Z_inv2 - Z_inv_warp) * mask2
        loss_Z = tf.clip_by_value(loss_Z, 0., .01)
        loss_Z = tf.reduce_sum(input_tensor=loss_Z) / sum_mask2
    else:
        loss_Z = tf.reduce_sum(input_tensor=tf.abs(Z_inv2 - Z_inv_warp) * mask2) / sum_mask2
    loss_X = tf.reduce_sum(input_tensor=tf.abs(X2_warp - X2_orig) * mask2) / sum_mask2
    loss_Y = tf.reduce_sum(input_tensor=tf.abs(Y2_warp - Y2_orig) * mask2) / sum_mask2



    # return tf.reduce_mean(tf.abs(Y2_warp - Y2_orig) * mask2 * Z_inv2), Z_inv_warp, Z_inv2

    # loss = (tf.reduce_sum(tf.abs(Z_inv2 - Z_inv_warp) * 10. * mask2) +
    #         tf.reduce_sum(tf.abs(X2_warp - X2_orig) * mask2 * Z_inv2) +
    #         tf.reduce_sum(tf.abs(Y2_warp - Y2_orig) * mask2 * Z_inv2)) / sum_mask2

    # loss = (tf.reduce_sum(tf.abs(Z2_warp - Z2_orig) * mask2) +
    #         tf.reduce_sum(tf.abs(X2_warp - X2_orig) * mask2) +
    #         tf.reduce_sum(tf.abs(Y2_warp - Y2_orig) * mask2)) / sum_mask2
    if return_diff_ims:
        return loss_X, loss_Y, loss_Z, Z_inv_warp, Z_inv2, tf.abs(Z_inv2 - Z_inv_warp)*mask2, tf.abs(X2_warp-X2_orig)*mask2, tf.abs(Y2_warp-Y2_orig)*mask2
    return loss_X, loss_Y, loss_Z, Z_inv_warp, Z_inv2

def open_bit_mask(bit_mask):
    # equivalent of numpy unpack bits function
    bit_ref = tf.constant(np.array([128, 64, 32, 16, 8, 4, 2, 1]).reshape(1, 1, 1, 8), dtype=tf.uint8)
    bit_mask = tf.cast(bit_mask, tf.uint8)

    _, height, width, _ = bit_mask.get_shape().as_list()
    open_mask = tf.cast(
        tf.reshape(tf.reshape(tf.bitwise.bitwise_and(bit_mask, bit_ref), [-1]), [-1, height, width, 8]),
        tf.float32)
    open_mask = tf.compat.v1.where(open_mask > 0.5, tf.ones_like(open_mask, tf.float32), tf.zeros_like(open_mask, tf.float32))

    return open_mask

def object_consistency_loss(motion, curr_instance_mask, curr_instance_ids, seg_im):
    motion *= 5.

    open_mask = open_bit_mask(curr_instance_mask)

    mean_sum = tf.zeros_like(motion[:, 0, 0, 0])
    # add_mask = tf.zeros_like(tf.expand_dims(motion[:,:,:,0], 3))
    Z_loss_sum = tf.expand_dims(tf.zeros_like(motion[:, :, :, 0]), 3)

    seg_cars = tf.compat.v1.where(tf.equal(seg_im, 8), tf.ones_like(seg_im), tf.zeros_like(seg_im))

    for i in range(8):
        # check distance of all motion outputs inside mask against mean inside mask
        mask = tf.expand_dims(open_mask[:, :, :, i], 3) * seg_cars
        other_cars_overlap_mask = tf.compat.v1.where(
            tf.logical_and(tf.greater(tf.expand_dims(tf.reduce_sum(input_tensor=open_mask, axis=3), 3) * seg_cars - mask,
                                      tf.zeros_like(mask)), tf.equal(mask, tf.ones_like(mask))), tf.ones_like(mask),
            tf.zeros_like(mask))

        mask *= (1. - other_cars_overlap_mask)
        # mask = tf.where(tf.equal(add_mask, tf.ones_like(add_mask)), tf.zeros_like(mask), mask)
        # add_mask = tf.where(tf.greater_equal(mask, add_mask), mask, add_mask)

        temp_sum, temp_Z_loss = calc_one_mask_loss(mask, motion)
        mean_sum += temp_sum
        Z_loss_sum += temp_Z_loss

    num_means = tf.reduce_sum(input_tensor=tf.minimum(curr_instance_ids, tf.ones_like(curr_instance_ids)), axis=[1])
    total_mean = mean_sum / tf.maximum(num_means, tf.ones_like(num_means))

    mean_loss = tf.reduce_sum(input_tensor=total_mean) / tf.maximum(tf.reduce_sum(input_tensor=num_means), 1.)

    return mean_loss, Z_loss_sum


def advanced_object_consistency_loss(motion, curr_instance_mask, curr_instance_ids, seg_im, motion_gt):

    motion *= 5.

    open_mask = open_bit_mask(curr_instance_mask)

    mean_sum = tf.zeros_like(motion[:,0,0,0])
    # add_mask = tf.zeros_like(tf.expand_dims(motion[:,:,:,0], 3))
    Z_loss_sum = tf.expand_dims(tf.zeros_like(motion[:,:,:,0]), 3)

    seg_cars = tf.compat.v1.where(tf.equal(seg_im, 8), tf.ones_like(seg_im), tf.zeros_like(seg_im))

    sum_direct = tf.zeros_like(seg_cars)

    direct_mask_sum = 0.

    for i in range(8):
        # check distance of all motion outputs inside mask against mean inside mask
        mask = tf.expand_dims(open_mask[:,:,:,i], 3) * seg_cars
        other_cars_overlap_mask = tf.compat.v1.where(tf.logical_and(tf.greater(tf.expand_dims(tf.reduce_sum(input_tensor=open_mask, axis=3), 3)*seg_cars-mask,
                                                             tf.zeros_like(mask)), tf.equal(mask, tf.ones_like(mask))), tf.ones_like(mask), tf.zeros_like(mask))

        mask *= (1.-other_cars_overlap_mask)
        # mask = tf.where(tf.equal(add_mask, tf.ones_like(add_mask)), tf.zeros_like(mask), mask)
        # add_mask = tf.where(tf.greater_equal(mask, add_mask), mask, add_mask)

        temp_sum, temp_Z_loss = calc_one_mask_loss(mask, motion)
        mean_sum += temp_sum
        Z_loss_sum += temp_Z_loss
        direct_diff, expanded_mask = calc_direct_loss(motion, mask, motion_gt[:, i, :])
        direct_mask_sum += tf.reduce_sum(input_tensor=expanded_mask)
        sum_direct += direct_diff

    direct_loss_X = tf.reduce_sum(input_tensor=sum_direct[:,:,:,0]) / tf.maximum(direct_mask_sum / 2., 1.)
    direct_loss_Z = tf.reduce_sum(input_tensor=sum_direct[:,:,:,2]) / tf.maximum(direct_mask_sum / 2., 1.)

    num_means = tf.reduce_sum(input_tensor=tf.minimum(curr_instance_ids, tf.ones_like(curr_instance_ids)), axis=[1])
    total_mean = mean_sum / tf.maximum(num_means, tf.ones_like(num_means))

    mean_loss = tf.reduce_sum(input_tensor=total_mean)/ tf.maximum(tf.reduce_sum(input_tensor=num_means), 1.)

    return mean_loss, Z_loss_sum, direct_loss_X, direct_loss_Z, sum_direct

def object_consistency_descriptor_loss(curr_motion, curr_instance_mask, curr_instance_ids, seg_im, motion_gt,
                                       min_des_loss=False, with_static=False, euclid=True):

    motion = curr_motion[:,:,:,:3] * 5.

    des = curr_motion[:,:,:,3:]

    open_mask = open_bit_mask(curr_instance_mask)

    mean_sum = tf.zeros_like(motion[:,0,0,0])
    # add_mask = tf.zeros_like(tf.expand_dims(motion[:,:,:,0], 3))
    Z_loss_sum = tf.expand_dims(tf.zeros_like(motion[:,:,:,0]), 3)

    seg_cars = tf.compat.v1.where(tf.equal(seg_im, 8), tf.ones_like(seg_im), tf.zeros_like(seg_im))

    sum_direct = tf.zeros_like(seg_cars)

    direct_mask_sum = 0.

    des_pull_loss = 0.

    for i in range(8):
        # check distance of all motion outputs inside mask against mean inside mask
        mask = tf.expand_dims(open_mask[:,:,:,i], 3) * seg_cars
        other_cars_overlap_mask = tf.compat.v1.where(tf.logical_and(tf.greater(tf.expand_dims(tf.reduce_sum(input_tensor=open_mask, axis=3), 3)*seg_cars-mask,
                                                             tf.zeros_like(mask)), tf.equal(mask, tf.ones_like(mask))), tf.ones_like(mask), tf.zeros_like(mask))

        mask *= (1.-other_cars_overlap_mask)
        # mask = tf.where(tf.equal(add_mask, tf.ones_like(add_mask)), tf.zeros_like(mask), mask)
        # add_mask = tf.where(tf.greater_equal(mask, add_mask), mask, add_mask)

        tiled_mask = tf.tile(mask, [1, 1, 1, 3])
        des_mean = tf.reduce_sum(input_tensor=des*mask, axis=[1,2]) / tf.maximum(tf.reduce_sum(input_tensor=mask, axis=[1,2]), tf.ones_like(tf.reduce_sum(input_tensor=mask, axis=[1,2])))
        # L1 now, change to euclidean?
        tiled_mean = tf.tile(tf.expand_dims(tf.expand_dims(des_mean, 1), 2), [1, tf.shape(input=mask)[1], tf.shape(input=mask)[2], 1])
        if euclid:
            des_pull_loss += tf.reduce_sum(input_tensor=euclid_dist(des, tiled_mean, tiled_mask))
        else:
            des_pull_loss += tf.reduce_sum(input_tensor=abs_dist(des, tiled_mean, tiled_mask))
        # des_pull_loss += tf.reduce_sum(tf.abs(des*tiled_mask - tiled_mean*tiled_mask)) / tf.maximum(tf.reduce_sum(tiled_mask), 1.)
        des_mean = tf.expand_dims(des_mean, 1)
        des_mask = np.ones_like(des_mean) * tf.tile(
            tf.expand_dims(tf.expand_dims(tf.reduce_max(input_tensor=mask, axis=[1, 2, 3]), 1), 2), [1, 1, 3])
        if i == 0:
            all_des = des_mean
            des_masks = des_mask
        else:
            all_des = tf.concat([all_des, des_mean], axis=1)
            des_masks = tf.concat([des_masks, des_mask], axis=1)

        temp_sum, temp_Z_loss = calc_one_mask_loss(mask, motion)
        mean_sum += temp_sum
        Z_loss_sum += temp_Z_loss
        direct_diff, expanded_mask = calc_direct_loss(motion, mask, motion_gt[:, i, :])
        direct_mask_sum += tf.reduce_sum(input_tensor=expanded_mask)
        sum_direct += direct_diff

    des_pull_loss /= 8.

    if with_static:
        # Use seg-non cars or people, not just seg-non cars
        static_mean = tf.reduce_sum(input_tensor=des * (1.-seg_cars), axis=[1,2]) / tf.maximum(tf.reduce_sum(input_tensor=1.-seg_cars, axis=[1,2]), tf.ones_like(tf.reduce_sum(input_tensor=1.-seg_cars, axis=[1,2])))
        all_des = tf.concat([all_des, tf.expand_dims(static_mean, axis=1)], axis=1)
        des_masks = tf.concat([des_masks, tf.ones_like(tf.expand_dims(static_mean, axis=1))], axis=1)

    ##### Insert loss here #######
    ## It should make sure that each item in object_des is as far as possible from the others
    ## L1 for now, change to euclidean?

    if min_des_loss:
        des_push_loss = min_push_loss(all_des, des_masks, with_static, euclid)
    else:
        des_push_loss = push_loss(all_des, des_masks, with_static, euclid)

    direct_loss_X = tf.reduce_sum(input_tensor=sum_direct[:,:,:,0]) / tf.maximum(direct_mask_sum / 2., 1.)
    direct_loss_Z = tf.reduce_sum(input_tensor=sum_direct[:,:,:,2]) / tf.maximum(direct_mask_sum / 2., 1.)

    num_means = tf.reduce_sum(input_tensor=tf.minimum(curr_instance_ids, tf.ones_like(curr_instance_ids)), axis=[1])
    total_mean = mean_sum / tf.maximum(num_means, tf.ones_like(num_means))

    mean_loss = tf.reduce_sum(input_tensor=total_mean)/ tf.maximum(tf.reduce_sum(input_tensor=num_means), 1.)

    return mean_loss, Z_loss_sum, direct_loss_X, direct_loss_Z, sum_direct, des_pull_loss, des_push_loss

def push_loss(all_des, des_masks, with_static=False, euclid=True):

    num_des = 8
    if with_static:
        num_des += 1

    des_push_loss = 0.
    rolled_des = tf.identity(all_des)
    rolled_des_masks = tf.identity(des_masks)
    for i in range(1, num_des):
        rolled_des = roll_3dtensor(rolled_des, 1, 1)
        rolled_des_masks = roll_3dtensor(rolled_des_masks, 1, 1)
        temp_mask = des_masks * rolled_des_masks
        if euclid:
            des_push_loss += tf.reduce_sum(input_tensor=euclid_dist(all_des, rolled_des, temp_mask))
        else:
            des_push_loss += tf.reduce_sum(input_tensor=abs_dist(all_des, rolled_des, temp_mask))
        # des_push_loss += tf.reduce_sum(tf.abs(all_des - rolled_des) * des_masks * rolled_des_masks) / tf.maximum(
        #     tf.reduce_sum(des_masks * rolled_des_masks) * 2., 1.)
    des_push_loss /= float(num_des-1)

    des_push_loss = 1. - des_push_loss

    return des_push_loss

def abs_dist(tens1, tens2, mask, factor=2.):

    return tf.abs(tens1 - tens2) * mask / tf.maximum(
            tf.reduce_sum(input_tensor=mask) * factor, 1.)

def euclid_dist(tens1, tens2, mask, factor=4.):

    # Assumes all input is same shape and last dimension is the vector
    sum = tf.reduce_sum(input_tensor=tf.square(tens1-tens2) * mask, axis=-1)
    # dist = tf.where(tf.equal(sum, 0.), tf.zeros_like(sum), tf.sqrt(sum)) / tf.maximum(
    #         tf.reduce_sum(mask) * factor, 1.)
    dist = tf.compat.v1.where(tf.equal(sum, 0.), tf.zeros_like(sum), sum) / tf.maximum(
            tf.reduce_sum(input_tensor=mask) * factor, 1.)
    # Output of this meant to be summed
    return dist

def min_push_loss(all_des, des_masks, with_static=False, euclid=True):

    num_des = 8
    if with_static:
        num_des += 1

    rolled_des = tf.identity(all_des)
    rolled_des_masks = tf.identity(des_masks)
    for i in range(1, 8):
        rolled_des = roll_3dtensor(rolled_des, 1, 1)
        rolled_des_masks = roll_3dtensor(rolled_des_masks, 1, 1)
        temp_mask = des_masks * rolled_des_masks
        if euclid:
            temp_push_loss = euclid_dist(all_des, rolled_des, temp_mask)
        else:
            temp_push_loss = abs_dist(all_des, rolled_des, temp_mask)
        # temp_push_loss = tf.abs(all_des - rolled_des) * des_masks * rolled_des_masks / tf.maximum(
        #     tf.reduce_sum(des_masks * rolled_des_masks) * 2., 1.)
        if i == 1:
            des_push_loss = temp_push_loss
        else:
            des_push_loss = tf.compat.v1.where(tf.logical_and(tf.not_equal(temp_push_loss, 0.), tf.equal(des_push_loss, 0.)),
                                     temp_push_loss, des_push_loss)
            des_push_loss = tf.compat.v1.where(tf.less(temp_push_loss, des_push_loss), temp_push_loss, des_push_loss)
    des_push_loss = tf.reduce_sum(input_tensor=des_push_loss) / float(num_des-1)

    des_push_loss = 1. - des_push_loss

    return des_push_loss

def warp_loss(prev_instance_mask, curr_instance_mask, Z_inv2, motion, focal, origin, RT12, x, y):

    motion_factor = 5.

    RT21 = tf.linalg.inv(RT12)

    Z2_orig = Z_inv2 ** -1
    X2_orig = (x * Z2_orig) / focal
    Y2_orig = (y * Z2_orig) / focal
    X2 = X2_orig - tf.expand_dims(motion[:,:,:,0], 3) * motion_factor
    Y2 = Y2_orig - tf.expand_dims(motion[:,:,:,1], 3) * motion_factor
    Z2 = Z2_orig - tf.expand_dims(motion[:,:,:,2], 3) * motion_factor

    X2_, Y2_, Z2_ = transform3_tf(X2, Y2, Z2, RT21)
    Z_inv2_ = Z2_ ** -1

    x2 = (X2_ * focal) * Z_inv2_ + origin[0, 0]  # notice that x2, y2 isn't relative to origin
    y2 = (Y2_ * focal) * Z_inv2_ + origin[0, 1]  # while x1, y1 are
    xy2 = tf.concat([x2, y2], 3)

    open_prev_mask = open_bit_mask(prev_instance_mask)
    open_curr_mask = open_bit_mask(curr_instance_mask)

    warp_loss = 0.

    for i in range(8):
        base_mask = tf.expand_dims(open_prev_mask[:,:,:,i], 3)
        base_mask_w, ooi = bilinear_sampler(base_mask, xy2)
        warp_loss += tf.reduce_mean(input_tensor=tf.abs(tf.expand_dims(open_curr_mask[:,:,:,i], 3) - base_mask_w))

    warp_loss /= 8.

    return warp_loss


def roll_3dtensor(tensor, shift, axis):
    # 2d not including batch
    if axis == 1:
        result = tf.concat([tensor[:,shift:,:], tensor[:,:shift,:]], axis=axis)
    elif axis == 2:
        result = tf.concat([tensor[:,:,shift:], tensor[:,:,:shift]], axis=axis)

    return result

def calc_direct_loss(motion, mask, motion_gt):

    # assumes that motion_gt is batch x 2 (X,Z)

    motion_gt = tf.concat([tf.expand_dims(motion_gt[:,0], 1), tf.ones((tf.shape(input=motion_gt)[0], 1))*-1, tf.expand_dims(motion_gt[:,1], 1)], axis=1)

    motion_gt = tf.expand_dims(tf.expand_dims(motion_gt, 1), 1)

    expanded_gt = tf.tile(motion_gt, [1, tf.shape(input=motion)[1], tf.shape(input=motion)[2], 1])

    expanded_mask = tf.tile(mask, [1, 1, 1, 3])

    # Default empty for vcl values is -1
    expanded_mask = tf.compat.v1.where(tf.equal(expanded_gt, -1), tf.zeros_like(expanded_mask), expanded_mask)

    diff = tf.abs(expanded_gt - motion) * expanded_mask

    return diff, expanded_mask

def calc_one_mask_loss(mask, motion):

    # mask batch x height x width x 1
    # motion batch x height x width x 3

    mask_sz = tf.reduce_sum(input_tensor=mask, axis=[1,2,3])
    mask_sz = tf.maximum(mask_sz, tf.ones_like(mask_sz))
    mask_mean = tf.reduce_sum(input_tensor=motion*mask, axis=[1,2]) / tf.tile(tf.expand_dims(mask_sz, 1), [1,3])

    full_diffs = tf.abs(motion - tf.tile(tf.expand_dims(tf.expand_dims(mask_mean, 1), 1),
                                         [1, tf.shape(input=motion)[1], tf.shape(input=motion)[2], 1]))
    sum_diffs = tf.reduce_sum(input_tensor=full_diffs*mask, axis=[1,2,3])
    mean_diff = sum_diffs / mask_sz

    return mean_diff, tf.expand_dims(full_diffs[:,:,:,2], 3)*mask

def bilinear_sampler(imgs, coords):



    """Construct a new image by bilinear sampling from the input image.

    Points falling outside the source image boundary have value 0.

    Args:
      imgs: source image to be sampled from [batch, height_s, width_s, channels]
      coords: coordinates of source pixels to sample from [batch, height_t,
        width_t, 2]. height_t/width_t correspond to the dimensions of the output
        image (don't need to be the same as height_s/width_s). The two channels
        correspond to x and y coordinates respectively.
    Returns:
      A new sampled image [batch, height_t, width_t, channels]
    """

    def _repeat(x, n_repeats):
        rep = tf.transpose(
            a=tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
            ])), 1), perm=[1, 0])
        rep = tf.cast(rep, 'float32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    with tf.compat.v1.name_scope('image_sampling'):
        coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
        inp_size = imgs.get_shape()
        coord_size = coords.get_shape()
        # out_size = coords.get_shape().as_list()
        # out_size[3] = imgs.get_shape().as_list()[3]
        out_size = [-1,coords.get_shape().as_list()[1], coords.get_shape().as_list()[2], imgs.get_shape().as_list()[3]]

        coords_x = tf.cast(coords_x, 'float32')
        coords_y = tf.cast(coords_y, 'float32')

        x0 = tf.floor(coords_x)
        x1 = x0 + 1
        y0 = tf.floor(coords_y)
        y1 = y0 + 1

        y_max = tf.cast(tf.shape(input=imgs)[1] - 1, 'float32')
        x_max = tf.cast(tf.shape(input=imgs)[2] - 1, 'float32')
        zero = tf.zeros([1], dtype='float32')

        x0_safe = tf.clip_by_value(x0, zero, x_max)
        y0_safe = tf.clip_by_value(y0, zero, y_max)
        x1_safe = tf.clip_by_value(x1, zero, x_max)
        y1_safe = tf.clip_by_value(y1, zero, y_max)

        ## bilinear interp weights, with points outside the grid having weight 0
        # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
        # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
        # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
        # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

        wt_x0 = x1_safe - coords_x
        wt_x1 = coords_x - x0_safe
        wt_y0 = y1_safe - coords_y
        wt_y1 = coords_y - y0_safe

        ## indices in the flat image to sample from
        dim2 = tf.cast(inp_size[2], 'float32')
        dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
        # base = tf.reshape(
        #     _repeat(
        #         tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
        #         coord_size[1] * coord_size[2]),
        #     [out_size[0], out_size[1], out_size[2], 1])
        base = tf.reshape(
            _repeat(
                tf.cast(tf.range(tf.shape(input=coords)[0]), 'float32') * dim1,
                coord_size[1] * coord_size[2]),
            [-1, out_size[1], out_size[2], 1])

        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = tf.reshape(x0_safe + base_y0, [-1])
        idx01 = x0_safe + base_y1
        idx10 = x1_safe + base_y0
        idx11 = x1_safe + base_y1

        ## sample from imgs
        imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
        imgs_flat = tf.cast(imgs_flat, 'float32')
        im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
        im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
        im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
        im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        output = tf.add_n([
            w00 * im00, w01 * im01,
            w10 * im10, w11 * im11
        ])
        return output, tf.cast(tf.logical_not(tf.logical_and(tf.logical_and(tf.equal(x0, x0_safe), tf.equal(x1, x1_safe)), tf.logical_and(tf.equal(y0, y0_safe), tf.equal(y1, y1_safe)))),'float32')
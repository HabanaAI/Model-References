import tensorflow as tf
import os
import imp
import json
import numpy as np
from stereo.models.arch_utils import conv2d, deconv2d
from stereo.models.MOVE_utils import warp, prep_motion2, prep_motion_ims


def arch(curr_inp, prev_inp, next_inp, prev_curr2frame, next_curr2frame,
         curr_T_cntr_srnd, prev_T_cntr_srnd, next_T_cntr_srnd, curr_focal,
         curr_origin, x, y, steps=np.linspace(0.0, 3.0 ** -1, 45), batch_norm=0):

# def arch(curr_inp, prev_inp, prev_curr2frame,
#          curr_T_cntr_srnd, prev_T_cntr_srnd, curr_focal,
#          curr_origin, x, y, steps=np.linspace(0.0, 3.0 ** -1, 45), batch_norm=0):

    ####### LOAD DEPTH NET #######

    json_path = '/mobileye/algo_RP/jeff/MOVE/models/conf/fc_u_main.4.0.4.1.json'
    # view_names = "main_to_main,frontCornerLeft_to_main,frontCornerRight_to_main"
    # checkpoint_dir = '/mobileye/algo_STEREO2/stereo/models/fc_u_main.4.0.4.1/checkpoints'
    exec_dir = '/mobileye/algo_RP/jeff/stereo/stereo/models'
    # restore_iter = 271000

    with open(json_path, 'rb') as fp:
        depth_model_conf = json.load(fp)

    # make sure checkpoint dir exists
    model_name = os.path.splitext(os.path.split(json_path)[1])[0]
    model_dir = os.path.join(depth_model_conf['output_base_dir'], model_name)

    # dynamically import the arch and loss functions
    depth_arch_func_ = imp.load_source('arch', os.path.join(exec_dir, 'arch', depth_model_conf['arch']['name'] + '.py')).arch

    # input_placeholder = tf.placeholder(tf.float32, [None, 3, 310, 720], "depth_input_images")
    # T_cntr_srnd_placeholder = tf.placeholder(tf.float32, [None, 2, 3], "T_cntr_srnd")
    # focal_placeholder = tf.placeholder(tf.float32, [None, 2], "focal")
    # origin_placeholder = tf.placeholder(tf.float32, [None, 2], "origin")

    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=tf.compat.v1.AUTO_REUSE):
        # CURR
        # curr_depth = tf.identity(
        #     depth_arch_func_(curr_inp, curr_T_cntr_srnd, curr_focal, curr_origin, None, None),
        #     name="curr_depth")
        # curr_depth = tf.stop_gradient(curr_depth)

        combined_depth = tf.identity(depth_arch_func_(tf.concat([curr_inp, prev_inp], 0),
                                            tf.concat([curr_T_cntr_srnd, prev_T_cntr_srnd], 0),
                                            tf.concat([curr_focal, curr_focal], 0),
                                            tf.concat([curr_origin, curr_origin], 0),
                                            None, None),
                                 name="combined_depth")
        combined_depth = tf.stop_gradient(combined_depth)

        curr_depth, prev_depth = tf.split(combined_depth, 2)

        # PREV
        # prev_depth = tf.identity(
        #     depth_arch_func_(prev_inp, prev_T_cntr_srnd, curr_focal, curr_origin, None, None),
        #     name="prev_depth")
        # prev_depth = tf.stop_gradient(prev_depth)

        # prev_depth = tf.identity(depth_arch(prev_inp, prev_T_cntr_srnd, curr_focal, curr_origin, None, None),
        #                          name="prev_depth")
        # prev_depth = tf.stop_gradient(prev_depth)

        # # NEXT
        # next_depth = tf.identity(
        #     depth_arch_func_(next_inp, next_T_cntr_srnd, curr_focal, curr_origin, None, None),
        #     name="next_depth")
        # next_depth = tf.stop_gradient(next_depth)

    ##############################

    ######### MOVE ###############


    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=tf.compat.v1.AUTO_REUSE):
        # CURR
        curr_motion, curr_Z2_w_to1_inv, curr_Z1_inv, curr_im2, curr_im2_w = motion(prev_inp, curr_inp, prev_depth,
                                                                                   curr_depth, prev_curr2frame,
                                                                                   curr_focal, curr_origin, x, y,
                                                                                   batch_norm)
        curr_motion = tf.identity(curr_motion, "curr_motion")
        curr_Z2_w_to1_inv = tf.identity(curr_Z2_w_to1_inv, "motion_input1")
        curr_Z1_inv = tf.identity(curr_Z1_inv, "motion_input2")
        curr_im2 = tf.identity(curr_im2, "motion_input3")
        curr_im2_w = tf.identity(curr_im2_w, "motion_input4")


        # NEXT
        # next_base2curr = next_curr2frame
        # next_motion = motion(curr_inp, next_inp, curr_depth, next_depth, next_base2curr, curr_focal, curr_origin,
        #                      x, y, batch_norm)

    ##############################

    return [curr_motion, curr_depth, prev_depth, curr_Z2_w_to1_inv, curr_Z1_inv, curr_im2, curr_im2_w]

def motion(inp1, inp2, Z_inv_base, Z_inv_curr, RTcurr2base, focal, origin, x, y, batch_norm=0):

    with tf.compat.v1.name_scope("motion_"):

        with tf.compat.v1.name_scope("pre_warp"):

            focal = tf.reshape(focal[:,0], [-1,1,1,1])

            # x_ = x / tf.reshape(focal[:,0], [-1,1,1,1])
            # y_ = y / tf.reshape(focal[:,1], [-1,1,1,1])

            x_ = tf.tile(x, [tf.shape(input=Z_inv_curr)[0], 1, 1, 1]) / focal
            y_ = tf.tile(y, [tf.shape(input=Z_inv_curr)[0], 1, 1, 1]) / focal


            inp1 = tf.transpose(a=inp1, perm=[0,2,3,1])
            inp2 = tf.transpose(a=inp2, perm=[0,2,3,1])

            Z2_w_to1_inv, Z1_inv = prep_motion2(Z_inv_base, Z_inv_curr, focal, origin, tf.linalg.inv(RTcurr2base), x, y)
            im2, im2_w = prep_motion_ims(Z_inv_curr, inp1, inp2, focal, origin, RTcurr2base, x, y)
            #
            # input = tf.concat([im1, im2, Z_inv_curr, x_, y_], axis=3)

            input = tf.concat([Z2_w_to1_inv, Z1_inv, im2_w, im2, x_, y_], axis=3, name="motion_input_concat")

            # input = tf.concat([im1, im2, Z_inv_base, Z_inv_curr], axis=3)

        with tf.compat.v1.name_scope("unet_phase_a"):
            conv_0 = input

            with tf.compat.v1.name_scope("down_convs"):
                down_conv_width = 5
                with tf.compat.v1.name_scope("conv_1"):
                    conv_1_ = conv2d(conv_0, 10, down_conv_width, strides=(1, 1), batch_norm=batch_norm, name="conv_1_0")
                    conv_1 = conv2d(conv_1_, 20, down_conv_width, strides=(2, 2), batch_norm=batch_norm, name="conv_1_1")
                with tf.compat.v1.name_scope("conv_2"):
                    conv_2_ = conv2d(conv_1, 20, down_conv_width, strides=(1, 1), batch_norm=batch_norm, name="conv_2_0")
                    conv_2 = conv2d(conv_2_, 40, down_conv_width, strides=(2, 2), batch_norm=batch_norm, name="conv_2_1")
                with tf.compat.v1.name_scope("conv_3"):
                    conv_3_ = conv2d(conv_2, 40, down_conv_width, strides=(1, 1), batch_norm=batch_norm, name="conv_3_0")
                    conv_3 = conv2d(conv_3_, 80, down_conv_width, strides=(2, 2), batch_norm=batch_norm, name="conv_3_1")
                with tf.compat.v1.name_scope("conv_4"):
                    conv_4_ = conv2d(conv_3, 80, down_conv_width, strides=(1, 1), batch_norm=batch_norm, name="conv_4_0")
                    conv_4 = conv2d(conv_4_, 160, down_conv_width, strides=(2, 2), batch_norm=batch_norm, name="conv_4_1")
                with tf.compat.v1.name_scope("conv_5"):
                    conv_5_ = conv2d(conv_4, 160, down_conv_width, strides=(1, 1), batch_norm=batch_norm, name="conv_5_0")
                    conv_5 = conv2d(conv_5_, 320, down_conv_width, strides=(2, 2), batch_norm=batch_norm, name="conv_5_1")
                with tf.compat.v1.name_scope("conv_6"):
                    conv_6_ = conv2d(conv_5, 320, down_conv_width, strides=(1, 1), batch_norm=batch_norm, name="conv_6_0")
                    conv_6 = conv2d(conv_6_, 640, down_conv_width, strides=(2, 2), batch_norm=batch_norm, name="conv_6_1")

            with tf.compat.v1.name_scope("up_convs"):
                up_conv_width = 3
                up_deconv_width = 5
                up_deconv_strides = [1, 2, 2, 1]
                with tf.compat.v1.name_scope("deconv_6"):
                    deconv_6 = deconv2d(conv_6, up_deconv_width, conv_6_.shape[1:3], int(conv_5.shape[3]),
                                        int(conv_6.shape[3]),
                                        strides=up_deconv_strides, batch_norm=batch_norm, name="deconv_6")
                    deconv_6 = tf.concat([deconv_6, conv_6_], axis=3)
                    deconv_6 = conv2d(deconv_6, conv_5.shape[3], up_conv_width, strides=(1, 1), batch_norm=batch_norm, name="deconv_6_0")
                    deconv_6 = conv2d(deconv_6, conv_5.shape[3], up_conv_width, strides=(1, 1), batch_norm=batch_norm, name="deconv_6_1")
                with tf.compat.v1.name_scope("deconv_5"):
                    deconv_5 = deconv2d(deconv_6, up_deconv_width, conv_5_.shape[1:3], int(conv_4.shape[3]),
                                        int(conv_5.shape[3]),
                                        strides=up_deconv_strides, batch_norm=batch_norm, name="deconv_5")
                    deconv_5 = tf.concat([deconv_5, conv_5_], axis=3)
                    deconv_5 = conv2d(deconv_5, conv_4.shape[3], up_conv_width, strides=(1, 1), batch_norm=batch_norm, name="deconv_5_0")
                    deconv_5 = conv2d(deconv_5, conv_4.shape[3], up_conv_width, strides=(1, 1), batch_norm=batch_norm, name="deconv_5_1")
                with tf.compat.v1.name_scope("deconv_4"):
                    deconv_4 = deconv2d(deconv_5, up_deconv_width, conv_4_.shape[1:3], int(conv_3.shape[3]),
                                        int(conv_4.shape[3]),
                                        strides=up_deconv_strides, batch_norm=batch_norm, name="deconv_4")
                    deconv_4 = tf.concat([deconv_4, conv_4_], axis=3)
                    deconv_4 = conv2d(deconv_4, conv_3.shape[3], up_conv_width, strides=(1, 1), batch_norm=batch_norm, name="deconv_4_0")
                    deconv_4 = conv2d(deconv_4, conv_3.shape[3], up_conv_width, strides=(1, 1), batch_norm=batch_norm, name="deconv_4_1")
                with tf.compat.v1.name_scope("deconv_3"):
                    deconv_3 = deconv2d(deconv_4, up_deconv_width, conv_3_.shape[1:3], int(conv_2.shape[3]),
                                        int(conv_3.shape[3]),
                                        strides=up_deconv_strides, batch_norm=batch_norm, name="deconv_3")
                    deconv_3 = tf.concat([deconv_3, conv_3_], axis=3)
                    deconv_3 = conv2d(deconv_3, conv_2.shape[3], up_conv_width, strides=(1, 1), batch_norm=batch_norm, name="deconv_3_0")
                    deconv_3 = conv2d(deconv_3, conv_2.shape[3], up_conv_width, strides=(1, 1), batch_norm=batch_norm, name="deconv_3_1")
                with tf.compat.v1.name_scope("deconv_2"):
                    deconv_2 = deconv2d(deconv_3, up_deconv_width, conv_2_.shape[1:3], int(conv_1.shape[3]),
                                        int(conv_2.shape[3]),
                                        strides=up_deconv_strides, batch_norm=batch_norm, name="deconv_2")
                    deconv_2 = tf.concat([deconv_2, conv_2_], axis=3)
                    deconv_2 = conv2d(deconv_2, conv_1.shape[3], up_conv_width, strides=(1, 1), batch_norm=batch_norm, name="deconv_2_0")
                    deconv_2 = conv2d(deconv_2, conv_1.shape[3], up_conv_width, strides=(1, 1), batch_norm=batch_norm, name="deconv_2_1")
                with tf.compat.v1.name_scope("deconv_1"):
                    deconv_1 = deconv2d(deconv_2, up_deconv_width, conv_1_.shape[1:3], int(conv_0.shape[3]),
                                        int(conv_1.shape[3]),
                                        strides=up_deconv_strides, batch_norm=batch_norm, name="deconv_1")
                    deconv_1 = tf.concat([deconv_1, conv_1_], axis=3)
                    deconv_1 = conv2d(deconv_1, conv_0.shape[3], up_conv_width, strides=(1, 1), batch_norm=batch_norm, name="deconv_1_0")
                    deconv_1 = conv2d(deconv_1, conv_0.shape[3], up_conv_width, strides=(1, 1), batch_norm=batch_norm, name="deconv_1_1")

        with tf.compat.v1.name_scope("argmax_phase_a"):
            argmax = conv2d(deconv_1, 20, 1, strides=(1, 1), batch_norm=batch_norm, name="conv_arg1")
            argmax = conv2d(argmax, 10, 1, strides=(1, 1), batch_norm=batch_norm, name="conv_arg2")
            argmax = conv2d(argmax, 6, 1, strides=(1, 1), batch_norm=batch_norm, name="conv_arg_new", activation=None)

        # out = tf.tanh(argmax, name="motion_output")
        out = tf.identity(tf.clip_by_value(argmax, -1., 1.), name="motion_output")

        return out, Z2_w_to1_inv, Z1_inv, im2, im2_w
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def get_matrix(M00, M01, M02,
               M10, M11, M12,
               M20, M21, M22,
               dim0=0, dim1=1):

    M0 = tf.concat([tf.expand_dims(tf.expand_dims(M00, dim0), dim1),
                    tf.expand_dims(tf.expand_dims(M01, dim0), dim1),
                    tf.expand_dims(tf.expand_dims(M02, dim0), dim1)], dim1)
    M1 = tf.concat([tf.expand_dims(tf.expand_dims(M10, dim0), dim1),
                    tf.expand_dims(tf.expand_dims(M11, dim0), dim1),
                    tf.expand_dims(tf.expand_dims(M12, dim0), dim1)], dim1)
    M2 = tf.concat([tf.expand_dims(tf.expand_dims(M20, dim0), dim1),
                    tf.expand_dims(tf.expand_dims(M21, dim0), dim1),
                    tf.expand_dims(tf.expand_dims(M22,dim0), dim1)], dim1)
    M = tf.concat([M0, M1, M2], dim0)
    return M


def tf_euler_to_R(thetax, thetay, thetaz, name='R'):
    rotation_matrix_x = tf.stack(
        [tf.constant(1.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32),
         tf.constant(0.0, dtype=tf.float32), tf.cos(thetaz), -tf.sin(thetaz),
         tf.constant(0.0, dtype=tf.float32), tf.sin(thetaz), tf.cos(thetaz)])
    rotation_matrix_y = tf.stack([
        tf.cos(thetay), tf.constant(0.0, dtype=tf.float32), tf.sin(thetay),
        tf.constant(0.0, dtype=tf.float32), tf.constant(1.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32),
        -tf.sin(thetay), tf.constant(0.0, dtype=tf.float32), tf.cos(thetay)])
    rotation_matrix_z = tf.stack([
        tf.cos(thetax), -tf.sin(thetax), tf.constant(0.0, dtype=tf.float32),
        tf.sin(thetax), tf.cos(thetax), tf.constant(0.0, dtype=tf.float32),
        tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32), tf.constant(1.0, dtype=tf.float32)])
    rotation_matrix_x = tf.reshape(rotation_matrix_x, (3, 3))
    rotation_matrix_y = tf.reshape(rotation_matrix_y, (3, 3))
    rotation_matrix_z = tf.reshape(rotation_matrix_z, (3, 3))
    R = tf.matmul(rotation_matrix_z, tf.matmul(rotation_matrix_y, rotation_matrix_x))

    R = tf.transpose(a=tf.reverse(R, [0, 1]))
    R = tf.identity(R, name=name)
    return R


def tf_quaternion_to_R(q0, q1, q2, q3, name='R'):
    q = tf.concat([tf.expand_dims(tf.expand_dims(q0,0),1),
                   tf.expand_dims(tf.expand_dims(q1,0),1),
                   tf.expand_dims(tf.expand_dims(q2,0),1),
                   tf.expand_dims(tf.expand_dims(q3,0),1)], 1)
    n = tf.reduce_sum(input_tensor=q*q)
    q = q*tf.sqrt(2/n)
    q = q*tf.reshape(q,[4,1])
    R00 = 1.0 - q[2, 2] - q[3, 3]
    R01 = q[1, 2] - q[3, 0]
    R02 = q[1, 3] + q[2, 0]
    R10 = q[1, 2] + q[3, 0]
    R11 = 1.0 - q[1, 1] - q[3, 3]
    R12 = q[2, 3] - q[1, 0]
    R20 = q[1, 3] - q[2, 0]
    R21 = q[2, 3] + q[1, 0]
    R22 = 1.0 - q[1, 1] - q[2, 2]
    R = get_matrix(R00, R01, R02, R10, R11, R12, R20, R21, R22)
    R = tf.identity(R, name=name)
    return R


class ResidualRotationWarpLayer(layers.Layer):
    def __init__(self, name='residual_rotation_warp', stddev=0.0, trainable=True, **kwargs):
        super(ResidualRotationWarpLayer, self).__init__(name=name, **kwargs)
        init_zero = tf.compat.v1.keras.initializers.normal(dtype=tf.float32, mean=0., stddev=stddev)
        init_one = tf.compat.v1.keras.initializers.normal(dtype=tf.float32, mean=1., stddev=stddev)
        self.q0 = tf.compat.v1.get_variable(initializer=init_one, dtype=tf.float32, trainable=trainable, name='%s/q0' % self.name,
                                  shape=())
        self.q1 = tf.compat.v1.get_variable(initializer=init_zero, dtype=tf.float32, trainable=trainable, name='%s/q1' % self.name,
                                  shape=())
        self.q2 = tf.compat.v1.get_variable(initializer=init_zero, dtype=tf.float32, trainable=trainable, name='%s/q2' % self.name,
                                  shape=())
        self.q3 = tf.compat.v1.get_variable(initializer=init_zero, dtype=tf.float32, trainable=trainable, name='%s/q3' % self.name,
                                  shape=())

        self.R = tf_quaternion_to_R(self.q0, self.q1, self.q2, self.q3, name='%s/R' % self.name)

        # self.yaw = tf.identity(
        #     tf.get_variable(name='%s/yaw' % self.name, initializer=init_zero, dtype=tf.float32, trainable=trainable,
        #                     shape=()),
        #     name='%s/yaw' % self.name)
        # self.pitch = tf.identity(
        #     tf.get_variable(name='%s/pitch' % self.name, initializer=init_zero, dtype=tf.float32, trainable=trainable,
        #                     shape=()),
        #     name='%s/pitch' % self.name)
        # self.roll = tf.identity(
        #     tf.get_variable(name='%s/roll' % self.name, initializer=init_zero, dtype=tf.float32, trainable=trainable,
        #                     shape=()),
        #     name='%s/roll' % self.name)
        #
        # self.R = tf_euler_to_R(self.yaw, self.pitch, self.roll, name='%s/R' % self.name)

    def get_H(self, focal):
        R = tf.tile(tf.expand_dims(self.R, 0), [tf.shape(input=focal)[0], 1, 1])

        diag = tf.stack([focal[:, 0], focal[:, 1], tf.ones_like(focal[:, 0])], axis=1)
        K = tf.linalg.diag(diag)
        invK = tf.linalg.diag(1./diag)

        H = tf.matmul(K, tf.matmul(R, invK))

        return H

    def warp_H(self, images, focal):
        H = self.get_H(focal)

        im_sz = images.shape.as_list()[1:3]
        origin_x, origin_y = np.float32(im_sz[1]/2), np.float32(im_sz[0]/2)
        left, right = -origin_x, np.float32(im_sz[1]) - origin_x
        bottom, top = -origin_y, np.float32(im_sz[0]) - origin_y

        x, y = tf.meshgrid(tf.range(left, right), tf.range(bottom, top))

        x = tf.tile(tf.expand_dims(tf.reshape(x, [1, -1]), 0), [tf.shape(input=images)[0], 1, 1])
        y = tf.tile(tf.expand_dims(tf.reshape(y, [1, -1]), 0), [tf.shape(input=images)[0], 1, 1])

        ps = tf.concat([x, y, tf.ones_like(x)], 1)

        ps_w = tf.matmul(H, ps)

        x_w = tf.reshape(ps_w[:, 0, :]/ps_w[:, 2, :], [-1, im_sz[0], im_sz[1], 1]) + origin_x
        y_w = tf.reshape(ps_w[:, 1, :]/ps_w[:, 2, :], [-1, im_sz[0], im_sz[1], 1]) + origin_y

        coords = tf.concat([x_w, y_w], 3)

        return tf.contrib.resampler.resampler(images, coords)

    def call(self, inputs):
        images, focal = inputs[0], inputs[1]
        image_w = self.warp_H(images, focal)
        return image_w

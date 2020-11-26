import tensorflow as tf
import numpy as np


# def add_regularization(reg_dict, method='L2'):
#     """
#     Get decay dictionary of and add L2 regularization
#     :param reg_dict: mapping between name of decay to variable
#     :param method: L2, L1, *Currently only L2*
#     :return:
#     """
#     for (reg_name, var) in reg_dict.iteritems():
#         # Currently only L2
#         reg_value = tf.nn.l2_loss(var, name='reg_' + reg_name)
#         tf.add_to_collection("REGULARIZATION_" + reg_name, reg_value)


def conv2d(input_tensor, depth, kernel,
           name='conv2d', data_format='NHWC', strides=(1, 1), batch_norm=False, activation=tf.nn.relu,
           regularizer=None, training=False):
    with tf.compat.v1.name_scope(name):
        conv = tf.compat.v1.layers.conv2d(input_tensor, filters=depth, kernel_size=kernel,
                                strides=strides, padding="SAME", data_format='channels_last',
                                kernel_initializer=tf.compat.v1.keras.initializers.glorot_normal(),
                                bias_regularizer=regularizer,
                                activity_regularizer=regularizer, name=name)
        if batch_norm:
            conv = tf.contrib.layers.batch_norm(conv, is_training=training, decay=0.98, fused=True)
            # conv = tf.contrib.layers.batch_norm(conv, is_training=training, fused=True, decay=0.98,
            #                                     param_regularizers={"beta": regularizer, "gamma": regularizer},
            #                                     renorm=True, renorm_decay=0.98, scale=True)
        if activation:
            conv = activation(conv)
    return conv


def deconv2d(input_tensor, filter_size, output_size, out_channels, in_channels,
             name='deconv2d', data_format='NHWC', strides=[1, 1, 1, 1], batch_norm=False, regularizer=None,
             training=False):
    with tf.compat.v1.name_scope(name):
        dyn_input_shape = tf.shape(input=input_tensor)
        batch_size = dyn_input_shape[0]
        out_shape = tf.stack([batch_size, output_size[0], output_size[1], out_channels])
        filter_shape = [filter_size, filter_size, out_channels, in_channels]
        w = tf.compat.v1.get_variable(name, shape=filter_shape, initializer=tf.compat.v1.keras.initializers.glorot_normal())
        if regularizer is not None:
            tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, regularizer(w))
        deconv = tf.nn.conv2d_transpose(input_tensor, w, out_shape, strides, padding='SAME', data_format=data_format)
        if batch_norm:
            deconv = tf.contrib.layers.batch_norm(deconv, is_training=training, decay=0.98, fused=True)

            # deconv = tf.contrib.layers.batch_norm(deconv, is_training=training, fused=True, decay=0.98,
            #                                       param_regularizers={"beta": regularizer, "gamma": regularizer},
            #                                       renorm=True, renorm_decay=0.98, scale=True)
    return deconv


def correlation_map(x, y, origin, focal, T12, steps, interpolation='BILINEAR', name="corr_map"):
    with tf.compat.v1.name_scope(name):
        corr_tensors = []
        if type(steps) is list or type(steps) is np.ndarray:
            n_steps = len(steps)
        else:
            n_steps = steps.shape.as_list()[1]
        for i in np.arange(n_steps):
            if type(steps) is list or type(steps) is np.ndarray:
                d = steps[i]
                if d == 0:
                    D = tf.zeros(tf.shape(input=T12[:, 2]))
                else:
                    D = tf.compat.v1.div(tf.ones(tf.shape(input=T12[:, 2])),
                               tf.add(tf.scalar_mul(d ** -1, tf.ones(tf.shape(input=T12[:, 2]))), T12[:, 2]))
            else:
                D = tf.compat.v1.where(steps[:, i] > 0,
                             tf.compat.v1.div(tf.ones_like(T12[:, 2]),
                                    tf.add(tf.multiply(steps[:, i] ** -1, tf.ones_like(T12[:, 2])), T12[:, 2])),
                             tf.zeros_like(T12[:, 2])
                             )
            alpha = tf.subtract(tf.ones(tf.shape(input=T12[:, 2])), tf.multiply(D, T12[:, 2]))
            beta = tf.add(tf.multiply(tf.multiply(D, T12[:, 2]), origin[:, 0]),
                          tf.multiply(D, tf.multiply(focal[:, 0], T12[:, 0])))
            gamma = tf.add(tf.multiply(tf.multiply(D, T12[:, 2]), origin[:, 1]),
                           tf.multiply(D, tf.multiply(focal[:, 1], T12[:, 1])))

            transforms = tf.transpose(a=tf.stack([alpha, tf.zeros(tf.shape(input=alpha)), beta,
                                                tf.zeros(tf.shape(input=alpha)), alpha, gamma,
                                                tf.zeros(tf.shape(input=alpha)), tf.zeros(tf.shape(input=alpha))]),
                                      perm=[1, 0])

            shifted = tf.contrib.image.transform(images=x,
                                                 transforms=transforms,
                                                 interpolation=interpolation,
                                                 name="corr_transform")

            corr = tf.reduce_mean(input_tensor=tf.multiply(shifted, y), axis=3)
            corr_tensors.append(corr)

        return tf.transpose(a=tf.stack(corr_tensors), perm=[1, 2, 3, 0])


def split_interleave(tensor, cycle_legth):
    return [tensor[i::cycle_legth, ...] for i in np.arange(cycle_legth)]


def stack_interleave(tens):
    shape = tens[0].shape
    tens = [tf.expand_dims(ten, 1) for ten in tens]
    return tf.reshape(tf.stack(tens, axis=1), [-1, shape[1], shape[2], shape[3]])


def corr_steps(num_steps, min_Z, max_Z, min_delta_Z, return_poly=False, poly_deg=5):

    Z = np.zeros((num_steps,))
    for i in range(num_steps):
        Z[i] = min_Z + i*min_delta_Z
        potential_inv_Z_delta = (Z[i] ** -1 - max_Z ** -1) / (num_steps - i)
        if (Z[i]**-1 - potential_inv_Z_delta)**-1 - Z[i] > min_delta_Z:
            break
    remaining_steps = num_steps - i
    Z[i:] = np.linspace(Z[i]**-1, max_Z**-1, remaining_steps)**-1
    steps = Z**-1
    if return_poly:
        x = (np.arange(num_steps, dtype=np.float32) - num_steps/2.)/(num_steps/2.)
        return np.polyfit(x, steps, poly_deg)
    return steps


def corr_ind_2_invZ(x, poly):
    out = tf.zeros_like(x)
    h, w = x.shape.as_list()[1], x.shape.as_list()[2]
    poly_deg = poly.shape.as_list()[1]-1
    for i in np.arange(poly_deg+1):
        coef = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(poly[:,i], 1),2),3), [1,h,w,1])
        out += coef * x**(poly_deg-i)
    return out

def batchNorm_activation(inp, name='batchNorm_activation', activation='relu', batch_norm=False, training=False, bn_first=False):
    assert activation is None or activation == 'relu'
    if bn_first:
        out = batchNorm(inp, name=name+'_bn', training=training) if batch_norm else inp
        out = tf.keras.layers.ReLU(name=name+'_relu').apply(out) if activation else out
    else:
        out = tf.keras.layers.ReLU(name=name+'_relu').apply(inp) if activation else inp
        out = batchNorm(out, name=name+'_bn', training=training) if batch_norm else out
    return out

def batchNorm(inp, name, training=False):
    bn = tf.keras.layers.BatchNormalization(name=name, renorm=True, fused=True, trainable=training)
    res = bn(inp, training=training)
    for x in bn.updates:
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, x)
    return res

def down_conv_layer(inp, kernel_size, ch_out, activation='relu', name='down_conv_layer',
                    batch_norm=False, training=False, bn_first=True, activation_and_bn_output=True, regularizer=None):
    with tf.keras.backend.name_scope(name):
        dcl1 = tf.keras.layers.Conv2D(filters=ch_out, kernel_size=kernel_size, name='dcl1',
                                      strides=[1, 1], padding="same", data_format='channels_last',
                                      kernel_initializer=tf.compat.v1.keras.initializers.glorot_normal(),
                                      kernel_regularizer=regularizer).apply(inp)
        dcl1 = batchNorm_activation(dcl1, name='dcl1_bn_activation', activation=activation,
                                    batch_norm=batch_norm, training=training, bn_first=bn_first)
        dcl2 = tf.keras.layers.Conv2D(filters=ch_out, kernel_size=kernel_size, name='dcl2',
                                      strides=[2, 2], padding="same", data_format='channels_last',
                                      kernel_initializer=tf.compat.v1.keras.initializers.glorot_normal(),
                                      kernel_regularizer=regularizer).apply(dcl1)
        if activation_and_bn_output:
            dcl2 = batchNorm_activation(dcl2, name='dcl2_bn_activation', activation=activation,
                                        batch_norm=batch_norm, training=training, bn_first=bn_first)

    out, to_skip = dcl2, dcl1
    return out, to_skip


def up_conv_layer(inp, from_skip, ch_out, activation='relu', name='up_conv_layer',
                  batch_norm=False, training=False, bn_first=True, activation_and_bn_output=True, regularizer=None):
    with tf.keras.backend.name_scope(name):
        ucl1 = tf.keras.layers.Conv2DTranspose(filters=ch_out, kernel_size=5, name='ucl1',
                                      strides=[2, 2], padding="same", data_format='channels_last',
                                      kernel_initializer=tf.compat.v1.keras.initializers.glorot_normal(),
                                      kernel_regularizer=regularizer).apply(inp)
        ucl1 = batchNorm_activation(ucl1, name='ucl1_bn_activation', activation=activation,
                                    batch_norm=batch_norm, training=training, bn_first=bn_first)

        concat = tf.keras.layers.concatenate([ucl1, from_skip], axis=3, name='concat')
        ucl2 = tf.keras.layers.Conv2D(filters=ch_out, kernel_size=3, name='ucl2',
                                      strides=[1, 1], padding="same", data_format='channels_last',
                                      kernel_initializer=tf.compat.v1.keras.initializers.glorot_normal(),
                                      kernel_regularizer=regularizer).apply(concat)
        if activation_and_bn_output:
            ucl2 = batchNorm_activation(ucl2, name='ucl2_bn_activation', activation=activation,
                                        batch_norm=batch_norm, training=training, bn_first=bn_first)

    return ucl2


def image_size_of_format(I, data_format='channels_last'):
    if isinstance(I, tf.Tensor):
        shape = I.shape.as_list()
        ndims = I.shape.ndims
    else:
        shape = I.shape
        ndims = I.ndim

    if ndims == 3:
        im_sz = np.array(shape[:2] if data_format=='channels_last' else shape[1:3])
    else:  # ndims == 4
        im_sz = np.array(shape[1:3] if data_format == 'channels_last' else shape[2:4])
    return im_sz


def pad_image(I_, origin, im_sz_new, data_format='channels_last'):
    im_sz = image_size_of_format(I_, data_format=data_format)
    delta_sz = im_sz_new - im_sz
    pad_beg = delta_sz // 2
    pad_end = delta_sz - pad_beg
    padding = ((pad_beg[0], pad_end[0]), (pad_beg[1], pad_end[1]))
    I_new = tf.keras.layers.ZeroPadding2D(padding, data_format)(I_)
    origin_new = origin + [[pad_beg[1], pad_beg[0]]]
    return I_new, origin_new, padding


def unpad_image(I_, cropping, data_format='channels_last'):
    I_new = tf.keras.layers.Cropping2D(cropping, data_format)(I_)
    return I_new


def pad_im_sz_to_levels(im_sz, num_levels):
    """ Return the smallest new image size (larger or equal to im_sz)  that can be pyramid reduced num_levels times """
    im_sz_padded = np.ceil(np.array(im_sz) / (2.0**num_levels)).astype('int32') * (2**num_levels)
    return im_sz_padded


def im_padding(im, num_levels, data_format='channels_last'):
    im_sz = image_size_of_format(im, data_format=data_format)
    im_sz_new = pad_im_sz_to_levels(im_sz, num_levels=num_levels)
    delta_sz = im_sz_new - im_sz
    pad_beg = delta_sz // 2
    pad_end = delta_sz - pad_beg
    padding = ((pad_beg[0], pad_end[0]), (pad_beg[1], pad_end[1]))
    return padding


from scipy import signal

def gkern(kernlen=11, std=2):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return np.reshape(gkern2d / np.sum(gkern2d), [kernlen, kernlen, 1, 1])

def blur_kernel_conv2d(input, kernel_rad=2, name="blur_kernel_conv2d"):

    diam = 2 * kernel_rad + 1

    filt = gkern(kernlen=diam)

    init = tf.compat.v1.constant_initializer(value=filt,
                                   dtype=tf.float32)

    return tf.compat.v1.layers.conv2d(input, filters=1, kernel_size=kernel_rad*2+1,
                             strides=(1, 1), padding="SAME", data_format='channels_first',
                             kernel_initializer=init,
                             use_bias=False,
                             activity_regularizer=None, name=name)


def color_categories(img):
    color_map = np.array([
        [0x66, 0x00, 0xcc],
        [0x00, 0x66, 0x66],
        [0xff, 0xff, 0x00],
        [0xff, 0x80, 0x00],
        [0xff, 0x00, 0x00],
        [0x00, 0x00, 0xcc],
        [0x00, 0xcc, 0xff],
    ])
    rgb_img = tf.gather(params=color_map, indices=img)
    return tf.cast(rgb_img, tf.uint8)


def color_confidence(confidence_img):
    indices = tf.cast(tf.round(confidence_img * 255), dtype=tf.int32)
    colors = np.array([
        [0.0416, 0., 0.],
        [0.05189484, 0., 0.],
        [0.06218969, 0., 0.],
        [0.07248453, 0., 0.],
        [0.08277938, 0., 0.],
        [0.09307422, 0., 0.],
        [0.10336906, 0., 0.],
        [0.11366391, 0., 0.],
        [0.12395875, 0., 0.],
        [0.1342536 , 0., 0.],
        [0.14454844, 0., 0.],
        [0.15484328, 0., 0.],
        [0.16513813, 0., 0.],
        [0.17543297, 0., 0.],
        [0.18572782, 0., 0.],
        [0.19602266, 0., 0.],
        [0.2063175 , 0., 0.],
        [0.21661235, 0., 0.],
        [0.22690719, 0., 0.],
        [0.23720204, 0., 0.],
        [0.24749688, 0., 0.],
        [0.25779173, 0., 0.],
        [0.26808657, 0., 0.],
        [0.27838141, 0., 0.],
        [0.28867626, 0., 0.],
        [0.2989711 , 0., 0.],
        [0.30926595, 0., 0.],
        [0.31956079, 0., 0.],
        [0.32985563, 0., 0.],
        [0.34015048, 0., 0.],
        [0.35044532, 0., 0.],
        [0.36074017, 0., 0.],
        [0.37103501, 0., 0.],
        [0.38132985, 0., 0.],
        [0.3916247 , 0., 0.],
        [0.40191954, 0., 0.],
        [0.41221439, 0., 0.],
        [0.42250923, 0., 0.],
        [0.43280407, 0., 0.],
        [0.44309892, 0., 0.],
        [0.45339376, 0., 0.],
        [0.46368861, 0., 0.],
        [0.47398345, 0., 0.],
        [0.48427829, 0., 0.],
        [0.49457314, 0., 0.],
        [0.50486798, 0., 0.],
        [0.51516283, 0., 0.],
        [0.52545767, 0., 0.],
        [0.53575251, 0., 0.],
        [0.54604736, 0., 0.],
        [0.5563422 , 0., 0.],
        [0.56663705, 0., 0.],
        [0.57693189, 0., 0.],
        [0.58722673, 0., 0.],
        [0.59752158, 0., 0.],
        [0.60781642, 0., 0.],
        [0.61811127, 0., 0.],
        [0.62840611, 0., 0.],
        [0.63870096, 0., 0.],
        [0.6489958 , 0., 0.],
        [0.65929064, 0., 0.],
        [0.66958549, 0., 0.],
        [0.67988033, 0., 0.],
        [0.69017518, 0., 0.],
        [0.70047002, 0., 0.],
        [0.71076486, 0., 0.],
        [0.72105971, 0., 0.],
        [0.73135455, 0., 0.],
        [0.7416494 , 0., 0.],
        [0.75194424, 0., 0.],
        [0.76223908, 0., 0.],
        [0.77253393, 0., 0.],
        [0.78282877, 0., 0.],
        [0.79312362, 0., 0.],
        [0.80341846, 0., 0.],
        [0.8137133 , 0., 0.],
        [0.82400815, 0., 0.],
        [0.83430299, 0., 0.],
        [0.84459784, 0., 0.],
        [0.85489268, 0., 0.],
        [0.86518752, 0., 0.],
        [0.87548237, 0., 0.],
        [0.88577721, 0., 0.],
        [0.89607206, 0., 0.],
        [0.9063669 , 0., 0.],
        [0.91666174, 0., 0.],
        [0.92695659, 0., 0.],
        [0.93725143, 0., 0.],
        [0.94754628, 0., 0.],
        [0.95784112, 0., 0.],
        [0.96813596, 0., 0.],
        [0.97843081, 0., 0.],
        [0.98872565, 0., 0.],
        [0.9990205 , 0., 0.],
        [1., 0.00931467, 0.],
        [1., 0.01960877, 0.],
        [1., 0.02990287, 0.],
        [1., 0.04019697, 0.],
        [1., 0.05049107, 0.],
        [1., 0.06078517, 0.],
        [1., 0.07107927, 0.],
        [1., 0.08137338, 0.],
        [1., 0.09166748, 0.],
        [1., 0.10196158, 0.],
        [1., 0.11225568, 0.],
        [1., 0.12254978, 0.],
        [1., 0.13284388, 0.],
        [1., 0.14313798, 0.],
        [1., 0.15343208, 0.],
        [1., 0.16372618, 0.],
        [1., 0.17402028, 0.],
        [1., 0.18431438, 0.],
        [1., 0.19460849, 0.],
        [1., 0.20490259, 0.],
        [1., 0.21519669, 0.],
        [1., 0.22549079, 0.],
        [1., 0.23578489, 0.],
        [1., 0.24607899, 0.],
        [1., 0.25637309, 0.],
        [1., 0.26666719, 0.],
        [1., 0.27696129, 0.],
        [1., 0.28725539, 0.],
        [1., 0.29754949, 0.],
        [1., 0.3078436, 0.],
        [1., 0.3181377, 0.],
        [1., 0.3284318, 0.],
        [1., 0.3387259, 0.],
        [1., 0.34902, 0.],
        [1., 0.3593141, 0.],
        [1., 0.3696082, 0.],
        [1., 0.3799023, 0.],
        [1., 0.3901964, 0.],
        [1., 0.4004905, 0.],
        [1., 0.4107846, 0.],
        [1., 0.42107871, 0.],
        [1., 0.43137281, 0.],
        [1., 0.44166691, 0.],
        [1., 0.45196101, 0.],
        [1., 0.46225511, 0.],
        [1., 0.47254921, 0.],
        [1., 0.48284331, 0.],
        [1., 0.49313741, 0.],
        [1., 0.50343151, 0.],
        [1., 0.51372561, 0.],
        [1., 0.52401971, 0.],
        [1., 0.53431382, 0.],
        [1., 0.54460792, 0.],
        [1., 0.55490202, 0.],
        [1., 0.56519612, 0.],
        [1., 0.57549022, 0.],
        [1., 0.58578432, 0.],
        [1., 0.59607842, 0.],
        [1., 0.60637252, 0.],
        [1., 0.61666662, 0.],
        [1., 0.62696072, 0.],
        [1., 0.63725482, 0.],
        [1., 0.64754893, 0.],
        [1., 0.65784303, 0.],
        [1., 0.66813713, 0.],
        [1., 0.67843123, 0.],
        [1., 0.68872533, 0.],
        [1., 0.69901943, 0.],
        [1., 0.70931353, 0.],
        [1., 0.71960763, 0.],
        [1., 0.72990173, 0.],
        [1., 0.74019583, 0.],
        [1., 0.75048993, 0.],
        [1., 0.76078404, 0.],
        [1., 0.77107814, 0.],
        [1., 0.78137224, 0.],
        [1., 0.79166634, 0.],
        [1., 0.80196044, 0.],
        [1., 0.81225454, 0.],
        [1., 0.82254864, 0.],
        [1., 0.83284274, 0.],
        [1., 0.84313684, 0.],
        [1., 0.85343094, 0.],
        [1., 0.86372504, 0.],
        [1., 0.87401915, 0.],
        [1., 0.88431325, 0.],
        [1., 0.89460735, 0.],
        [1., 0.90490145, 0.],
        [1., 0.91519555, 0.],
        [1., 0.92548965, 0.],
        [1., 0.93578375, 0.],
        [1., 0.94607785, 0.],
        [1., 0.95637195, 0.],
        [1., 0.96666605, 0.],
        [1., 0.97696016, 0.],
        [1., 0.98725426, 0.],
        [1., 0.99754836, 0.],
        [1., 1., 0.01176372],
        [1., 1., 0.02720491],
        [1., 1., 0.0426461],
        [1., 1., 0.05808729],
        [1., 1., 0.07352849],
        [1., 1., 0.08896968],
        [1., 1., 0.10441087],
        [1., 1., 0.11985206],
        [1., 1., 0.13529325],
        [1., 1., 0.15073444],
        [1., 1., 0.16617564],
        [1., 1., 0.18161683],
        [1., 1., 0.19705802],
        [1., 1., 0.21249921],
        [1., 1., 0.2279404],
        [1., 1., 0.2433816],
        [1., 1., 0.25882279],
        [1., 1., 0.27426398],
        [1., 1., 0.28970517],
        [1., 1., 0.30514636],
        [1., 1., 0.32058756],
        [1., 1., 0.33602875],
        [1., 1., 0.35146994],
        [1., 1., 0.36691113],
        [1., 1., 0.38235232],
        [1., 1., 0.39779352],
        [1., 1., 0.41323471],
        [1., 1., 0.4286759],
        [1., 1., 0.44411709],
        [1., 1., 0.45955828],
        [1., 1., 0.47499947],
        [1., 1., 0.49044067],
        [1., 1., 0.50588186],
        [1., 1., 0.52132305],
        [1., 1., 0.53676424],
        [1., 1., 0.55220543],
        [1., 1., 0.56764663],
        [1., 1., 0.58308782],
        [1., 1., 0.59852901],
        [1., 1., 0.6139702],
        [1., 1., 0.62941139],
        [1., 1., 0.64485259],
        [1., 1., 0.66029378],
        [1., 1., 0.67573497],
        [1., 1., 0.69117616],
        [1., 1., 0.70661735],
        [1., 1., 0.72205855],
        [1., 1., 0.73749974],
        [1., 1., 0.75294093],
        [1., 1., 0.76838212],
        [1., 1., 0.78382331],
        [1., 1., 0.79926451],
        [1., 1., 0.8147057],
        [1., 1., 0.83014689],
        [1., 1., 0.84558808],
        [1., 1., 0.86102927],
        [1., 1., 0.87647046],
        [1., 1., 0.89191166],
        [1., 1., 0.90735285],
        [1., 1., 0.92279404],
        [1., 1., 0.93823523],
        [1., 1., 0.95367642],
        [1., 1., 0.96911762],
        [1., 1., 0.98455881],
        [1., 1., 1.]])
    colors = tf.constant(colors)
    img = tf.gather(colors, indices)

    return img

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# %matplotlib inline

def interp2(z, x, y, ooi_val=0):
    """
    Perform bilinear interpolation of image z (with shape [h,w,c] at image points x, y. 
    Output has same shape as x and y in the image domain, and channels like z's 
    OOI pixels will get ooi_val (also during the interpolation)
    """
    assert(x.shape == y.shape)
    channels = z.shape[2]

    ooi = np.logical_or.reduce([y < 0, y >= z.shape[0],
                                x < 0, x >= z.shape[1]])


    x_ = np.floor(x).astype(np.int32)
    y_ = np.floor(y).astype(np.int32)

    y_0 = y_
    y_0_ooi = np.logical_or(y_0 < 0, y_0 >= z.shape[0])
    y_0[ y_0_ooi] = 0
    y_1 = y_+1
    y_1_ooi = np.logical_or(y_1 < 0, y_1 >= z.shape[0])
    y_1[y_1_ooi] = 0
    x_0 = x_
    x_0_ooi = np.logical_or(x_0 < 0, x_0 >= z.shape[1])
    x_0[x_0_ooi] = 0
    x_1 = x_ + 1
    x_1_ooi = np.logical_or(x_1 < 0, x_1 >= z.shape[1])
    x_1[x_1_ooi] = 0

    dx = x - x_
    dy = y - y_
    
    y_0_ooi = np.tile(np.expand_dims(y_0_ooi, 2), [1, 1, channels])
    y_1_ooi = np.tile(np.expand_dims(y_1_ooi, 2), [1, 1, channels])
    x_0_ooi = np.tile(np.expand_dims(x_0_ooi, 2), [1, 1, channels])
    x_1_ooi = np.tile(np.expand_dims(x_1_ooi, 2), [1, 1, channels])
    
    z00 = z[y_0, x_0, :]
    z00[np.logical_or(y_0_ooi, x_0_ooi)] = ooi_val
    z10 = z[y_1, x_0, :]
    z10[np.logical_or(y_1_ooi, x_0_ooi)] = ooi_val
    z01 = z[y_0, x_1, :]
    z01[np.logical_or(y_0_ooi, x_1_ooi)] = ooi_val
    z11 = z[y_1, x_1, :]
    z11[np.logical_or(y_1_ooi, x_1_ooi)] = ooi_val

    w00 = (1 - dy) * (1 - dx)
    w10 =     (dy) * (1 - dx)
    w01 = (1 - dy) *     (dx)
    w11 =     (dy) *     (dx)

    w00 = np.tile(np.expand_dims(w00 ,2), [1, 1, channels])
    w10 = np.tile(np.expand_dims(w10, 2), [1, 1, channels])
    w01 = np.tile(np.expand_dims(w01, 2), [1, 1, channels])
    w11 = np.tile(np.expand_dims(w11, 2), [1, 1, channels])
    ooi = np.tile(np.expand_dims(ooi, 2), [1, 1, channels])

    zi = z00*w00 + z10*w10 + z01*w01 + z11*w11
    zi[ooi] = ooi_val
    return zi


def generate_scale_translate_params(origin, focal, T12, invZ_steps):
    if isinstance(origin, np.ndarray):
        origin = np.expand_dims(origin, 0)
        focal = np.expand_dims(focal, 0)
        T12 = np.expand_dims(T12, 0)
    scale, trans_x, trans_y = [], [], []
    for i,d in enumerate(invZ_steps):
        D = np.float32(1.)/ (d ** -1 + T12[:, 2]) if d != 0 else 0
        scale.append(np.float32(1.) - D*T12[:, 2] )
        trans_x.append(D * T12[:, 2] * origin[:, 0] + D*focal[:, 0] * T12[:, 0])
        trans_y.append(D * T12[:, 2] * origin[:, 1] + D*focal[:, 1] * T12[:, 1])
    if isinstance(origin, np.ndarray):
        scale = [np.asscalar(scale_) for scale_ in scale]
        trans_x = [np.asscalar(trans_x_) for trans_x_ in trans_x]
        trans_y = [np.asscalar(trans_y_) for trans_y_ in trans_y]
    return scale, trans_x, trans_y


# +
def scale_translate(image, scale, trans_x, trans_y):
    mesh_x, mesh_y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    x_w = scale*mesh_x + trans_x
    y_w = scale*mesh_y + trans_y
    return interp2(image, x_w, y_w)


def correlation_map(cntr, srnd, origin, focal, T12, invZ_steps):
    """
    NUMPY reference code for the correlation layer
    """
    scale, trans_x, trans_y = generate_scale_translate_params(origin, focal, T12, invZ_steps)
    corrs = np.zeros((cntr.shape[0], cntr.shape[1], len(invZ_steps)) ,dtype=cntr.dtype)
    for i in range(len(scale)):
        srnd_w = scale_translate(srnd, scale[i], trans_x[i], trans_y[i]) # scale_translate
        corr = np.sum(cntr*srnd_w, axis=2)                               # dot product
        corrs[:,:,i] = corr
    return corrs


# +
def scale_translate_tf(image, scale, trans_x, trans_y):
    transforms = tf.transpose(a=tf.stack([scale, tf.zeros(tf.shape(input=scale)), trans_x,
                                        tf.zeros(tf.shape(input=scale)), scale, trans_y,
                                        tf.zeros(tf.shape(input=scale)), tf.zeros(tf.shape(input=scale))]),
                              perm=[1, 0])

    return tf.contrib.image.transform(images=image, transforms=transforms, interpolation='BILINEAR')


def correlation_map_tf(cntr, srnd, origin, focal, T12, invZ_steps):
    """
    TENSORFLOW code of the correlation layer
    """
    scale, trans_x, trans_y = generate_scale_translate_params(origin, focal, T12, invZ_steps)
    corrs = []
    for i in range(len(invZ_steps)):
        srnd_w = scale_translate_tf(srnd, scale[i], trans_x[i], trans_y[i]) # scale_translate
        corr = tf.reduce_sum(input_tensor=cntr*srnd_w, axis=3)                           # dot product
        corrs.append(corr)
    return tf.transpose(a=tf.stack(corrs), perm=[1, 2, 3, 0])


# -

data = np.load('./correlation_data.npz')

# +
use_np_implementaion = True

corrs = []

if use_np_implementaion: 
    for i in [0, 1, 2]:
        corrs.append(correlation_map(data['corr_features_cntr'],
                                     data['corr_features_srnd_%d' % i],
                                     data['origin'],  
                                     data['focal'], 
                                     data['T_cntr_srnd'][i,:], 
                                     data['steps']))
    
else: # tf implementaion
    for i in [0, 1, 2]:
        corrs.append(correlation_map_tf(tf.expand_dims(tf.convert_to_tensor(value=data['corr_features_cntr']), 0),   
                                        tf.expand_dims(tf.convert_to_tensor(value=data['corr_features_srnd_%d' % i]), 0),
                                        tf.expand_dims(tf.convert_to_tensor(value=data['origin']), 0),  
                                        tf.expand_dims(tf.convert_to_tensor(value=data['focal']), 0), 
                                        tf.expand_dims(tf.convert_to_tensor(value=data['T_cntr_srnd'][i,:]), 0), 
                                        data['steps']))
    with tf.compat.v1.Session() as sess:
        corrs = [tens.squeeze() for tens in sess.run(corrs)]
corr = np.max(np.stack(corrs, -1), -1)
# -

fig = plt.figure(figsize=(11,9))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax1.imshow(data['im_cntr'], cmap='gray', origin='lower')
ax2.imshow(-np.squeeze(np.argmax(corr, axis=2)), origin='lower')
ax3.imshow(data['out'], origin='lower')





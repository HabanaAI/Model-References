import tensorflow as tf

cfg=tf.compat.v1.ConfigProto(device_count={'GPU':0})
tf.compat.v1.enable_eager_execution(cfg)

from train_example import input_fn
from stereo.models.layers.correlation_map_dlo import correlation_map
from stereo.models.layers.translate_from_calibration import TranslateFromCalibration
from stereo.models.layers.correlation_layer import Correlation as CorrelationDLO


# random feature maps
corr_features_cntr_l2=tf.random.normal((1,192,96,64))
corr_features_srnd_0_l2=tf.random.normal((1,192,96,64))

## extract sample data
ds = input_fn('train',1)
keys=['I_cntr',
 'I_srnd_0',
 'I_srnd_1',
 'I_srnd_2',
 'T_cntr_srnd',
 'clip_name',
 'cntr_cam_name',
 'focal',
 'gi',
 'im_lidar_inv',
 'im_lidar_short_inv',
 'im_mask',
 'origin',
 'photo_loss_im_mask',
 'speed',
 'origin_l2',
 'focal_l2',
 'x',
 'y',
 'blur_kernels',
 'deeplab']
data_point_dict=next(tf.compat.v1.data.make_one_shot_iterator(ds))[0]


# parameters
I_srnd_0 = data_point_dict['I_srnd_0']
T_cntr_srnd0 = data_point_dict['T_cntr_srnd'][:,0:1,:]
origin_l2 = data_point_dict['origin_l2']
focal_l2 = data_point_dict['focal_l2']

steps = tf.constant([0.5 , 0.47619048, 0.45454545, 0.43478261, 0.41666667,
                 0.4, 0.38461538, 0.37037037, 0.35714286, 0.34482759,
                 0.33333333, 0.32258065, 0.3125    , 0.3030303 , 0.29411765,
                 0.2852656 , 0.27641355, 0.2675615 , 0.25870945, 0.2498574 ,
                 0.24100535, 0.2321533 , 0.22330125, 0.2144492 , 0.20559715,
                 0.1967451 , 0.18789305, 0.179041  , 0.17018895, 0.1613369 ,
                 0.15248485, 0.1436328 , 0.13478075, 0.1259287 , 0.11707665,
                 0.1082246 , 0.09937255, 0.0905205 , 0.08166845, 0.0728164 ,
                 0.06396435, 0.0551123 , 0.04626025, 0.0374082 , 0.02855615,
                 0.0197041 , 0.01085205, 0.002 ])

## the next line aim to reproduce the following call:
# corr_scores_srnd_0_l2 = correlation_map(corr_features_cntr_l2, corr_features_srnd_0_l2, origin_l2, focal_l2,
#                                         T_cntr_srnd0, steps, name='corr_scores_srnd_0_l2')

# correlation map creates a Correlation object which creates CorrelationDLO call with the folowing trasformation params
srnd = corr_features_srnd_0_l2
cntr = corr_features_cntr_l2
steps_cnt = len(steps)

# extract transform parameters per sample per srnd input (only steps is shared)
TFC_keras_layer = TranslateFromCalibration(steps)
# focal origin and camera relation change the transformation used according to the sample and reference camera used
transform_params = TFC_keras_layer([origin_l2,focal_l2,T_cntr_srnd0])
scales = transform_params[0]
trans_x = transform_params[1]
trans_y = transform_params[2]

# in a for loop within CorrelationDLO we compose the transformations and apply them to the warp the srnd input
# corrs_concat = CorrelationDLO()([cntr, srnd, scale, trans_x, trans_y])
scale_split = tf.split(scales, steps_cnt, axis=-1)
trans_x_split = tf.split(trans_x, steps_cnt, axis=-1)
trans_y_split = tf.split(trans_x, steps_cnt, axis=-1)
corrs = []
## we want to fuse all transformations to a single kernel for fw and backward pass (take params trans_x trans_y and scales)
for i in range(steps_cnt):
    scale = scale_split[i]
    trans_x = trans_x_split[i]
    trans_y = trans_y_split[i]

    transform = tf.transpose(a=tf.stack([scale, tf.zeros(tf.shape(input=scale)), trans_x,
                                        tf.zeros(tf.shape(input=scale)), scale, trans_y,
                                        tf.zeros(tf.shape(input=scale)), tf.zeros(tf.shape(input=scale))])[:, :, 0],
                              perm=[1, 0])
    # no gradients passed to transform params,
    warped_srnd = tf.contrib.image.transform(images=srnd,
                                             transforms=transform,
                                             interpolation='NEAREST')
    corr = tf.reduce_sum(input_tensor=cntr * warped_srnd, axis=-1, keepdims=True)
    corrs.append(corr)
pass

# this function applies an affine transform on the input feature map ids (ind mapping only,no interpolation)
def my_packed_transform(feature_maps,transform_x,transform_y):
    # list all image pixel ids
    size_x,size_y = feature_maps.shape[1],feature_maps.shape[2]
    ids = tf.reshape(tf.stack(tf.meshgrid(tf.range(size_x), tf.range(size_y)), -1), (-1, 2))
    # append 1 for translation
    ids_affine = tf.concat([ids, tf.ones((ids.shape[0], 1), dtype=tf.int32)], 1)
    xt = tf.matmul(tf.cast(ids_affine, tf.float32), tf.transpose(a=transform_x, perm=[0, 2, 1]))
    yt = tf.matmul(tf.cast(ids_affine, tf.float32), tf.transpose(a=transform_y, perm=[0, 2, 1]))
    xy_t = tf.stack([xt,yt],2)
    return xy_t


def transform_scale_translate(feature_maps,scales,translate_x,translate_y):
    transform_x = tf.stack([scales, tf.zeros(tf.shape(input=scales)), translate_x], -1)
    transform_y = tf.stack([scales, tf.zeros(tf.shape(input=scales)), translate_y], -1)
    return my_packed_transform(feature_maps,transform_x, transform_y)


def transform_scale_translate_inv(feature_maps,scales,translate_x,translate_y):
    return transform_scale_translate(feature_maps,1/scales,-translate_x/scales,-translate_y/scales)
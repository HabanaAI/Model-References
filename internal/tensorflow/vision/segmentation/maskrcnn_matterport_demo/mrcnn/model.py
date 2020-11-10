"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow.compat.v1 as tf
tf.enable_resource_variables()
import tensorflow.compat.v1.keras as keras
import tensorflow.compat.v1.keras.backend as K
import tensorflow.compat.v1.keras.layers as KL
KE=KL
import tensorflow.compat.v1.keras.models as KM
from tensorflow.python.keras.saving import hdf5_format as saving
from tensorflow.compat.v1.keras.utils import get_file
from tensorflow.python.util import nest
from tensorflow.python.keras import backend
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.client import timeline
from mrcnn import visualize
from mrcnn import utils
tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()

INIT_SEED=0
INIT_CLS_BIAS=0.0
#fcs
init_detectron_xavier = keras.initializers.VarianceScaling(scale=2., mode="fan_in", distribution="uniform",seed=INIT_SEED)
#convs
init_detectron_msra = keras.initializers.VarianceScaling(scale=2., mode="fan_out", distribution="normal",seed=INIT_SEED)
tf_he_normal = keras.initializers.he_normal(INIT_SEED)

INIT_FC_KERNEL = init_detectron_xavier
INIT_CONV_KERNEL = init_detectron_msra

INIT_BBOX_KERNEL= keras.initializers.normal(stddev=0.001,seed=INIT_SEED)
INIT_CLS_KERNEL= keras.initializers.normal(stddev=0.01,seed=INIT_SEED)

### some global configs
RPN_STEM_C = 256
USE_BG_BIAS=False
USE_L1_LOSS=False

SCALE_BY_NUM_GT = None #3
AVOID_LOSS_SWITCH = True
############################################################
#  Utility Functions
############################################################
def partial_class(cls, *args, **kwargs):
    from functools import partialmethod
    class PartialClass(cls):
        source_class = cls
        __name__ = cls.__name__
        __init__ = partialmethod(cls.__init__, *args, **kwargs)
    return PartialClass

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)


class FrozenBatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def __init__(self, trainable=False,fused=True,**kwargs):
        super().__init__(trainable=False,fused=fused,**kwargs)

    def call(self, inputs, training=False):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=False)


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = FrozenBatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = FrozenBatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = FrozenBatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = FrozenBatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = FrozenBatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = FrozenBatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = FrozenBatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = FrozenBatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(KE.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, dev_str, config=None, combined_nms=False, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.dev_str = dev_str
        self.combined_nms = combined_nms

    def call(self, inputs):
        with tf.device(self.dev_str):
            # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
            scores = inputs[0][:, :, 1]
            # Box deltas [batch, num_rois, 4]
            deltas = inputs[1]
            deltas = deltas * np.reshape(np.array(self.config.RPN_BBOX_STD_DEV), [1, 1, 4])
            # Anchors
            anchors = inputs[2]
            input_image_meta=inputs[3]

            # Improve performance by trimming to top anchors by score
            # and doing the rest on the smaller subset.
            pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
            ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                            name="top_anchors").indices
            scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                self.config.IMAGES_PER_GPU)
            deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                self.config.IMAGES_PER_GPU)
            pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                    self.config.IMAGES_PER_GPU,
                                    names=["pre_nms_anchors"])

            # Apply deltas to anchors to get refined anchors.
            # [batch, N, (y1, x1, y2, x2)]
            boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                      lambda x, y: apply_box_deltas_graph(x, y),
                                      self.config.IMAGES_PER_GPU,
                                      names=["refined_anchors"])

            # Clip to image boundaries. Since we're in normalized coordinates,
            # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
            # each image has its own window, make sure croping is done on valid pixels
            windows = input_image_meta[:, 7:11]
            image_shapes = input_image_meta[:, 4:6]
            normalized_windows = utils.batch_slice([windows, image_shapes], norm_boxes_graph, self.config.IMAGES_PER_GPU)
            boxes = utils.batch_slice([boxes, normalized_windows],
                                      clip_boxes_graph,
                                      self.config.IMAGES_PER_GPU,
                                      names=["refined_anchors_clipped"])

            # Filter out small boxes
            # According to Xinlei Chen's paper, this reduces detection accuracy
            # for small objects, so we're skipping it.

            # Non-max suppression
            # NonMaxSuppressionV4 is not supported on GPU for TF1.15 and TF2.2
            def nms(boxes, scores):
                with tf.device(self.dev_str):
                    indices_padded, num_valid = tf.image.non_max_suppression_padded(
                        boxes, scores, self.proposal_count,
                        self.nms_threshold, pad_to_max_output_size=True,
                        name="rpn_non_max_suppression")
                    proposals =  tf.gather(boxes, indices_padded)
                    # proposal_range  will be a constant
                    proposal_range = tf.range(self.proposal_count)
                    proposal_range = tf.stack([proposal_range, proposal_range, proposal_range, proposal_range], axis=1)
                    mask = tf.cast(tf.greater(num_valid, proposal_range),  dtype=tf.float32)
                    proposals = tf.multiply(proposals, mask)
                    return proposals

            if self.combined_nms:
                boxes = tf.expand_dims(boxes, axis=2)
                scores = tf.expand_dims(scores, axis=2)
                proposals, nmsed_scores, nmsed_classes, valid_detections = tf.image.combined_non_max_suppression(
                    boxes, scores, self.proposal_count, self.proposal_count, iou_threshold=self.nms_threshold,
                    clip_boxes=False, name="rpn_combined_non_max_suppression")
            else:
                proposals = utils.batch_slice([boxes, scores], nms,
                                      self.config.IMAGES_PER_GPU)
            return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


############################################################
#  ROIAlign Layer
############################################################

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.math.log(x) / tf.math.log(2.0)


from collections import namedtuple
class PyramidROIAlign(KE.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    _META_REG = namedtuple('RoiAlignMeta',['arrange_BxRxlevels','arrange_BxRxlevels_stacked','flatten_all_box_ids','out_shape'])
    def __init__(self, pool_shape, dev_str, B=None, R=None, C=None, custom_op=None, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)
        self.pool_intermediate_shape = [j * 2 for j in self.pool_shape]
        self.dev_str = dev_str
        self.B = B
        self.R = R
        self.C = C
        self.custom_op = custom_op
        self._use_where = False
        self._use_memory_for_roi_id_recovery = True
        self._set_meta_once = False
        # tensor caching for order manipulation in case of shape changes
        self._meta_cache = {}
        if all(dim is not None for dim in [self.B,self.R,self.C]):
            # shapes are fixed
            self._set_multi_level_roi_pool_meta()
            self._set_meta_once = True

    def _roi_align(self, featurs, boxes, img_ref):
        feats = tf.image.crop_and_resize(featurs, boxes, img_ref, self.pool_intermediate_shape, method="bilinear")
        feats = tf.nn.avg_pool2d(feats, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        return feats

    def _set_multi_level_roi_pool_meta(self):
        if not self._set_meta_once and (self.B, self.R, self.C) not in self._meta_cache:
            arrange_BxRxlevels = tf.tile(tf.range(self.R)[tf.newaxis, :], (self.B, 4))
            arrange_BxRxlevels_stacked = tf.tile(tf.range(self.R*4)[tf.newaxis, :], (self.B, 1))
            flatten_all_box_ids = tf.repeat(tf.range(self.B), self.R, 0)
            out_shape = tf.stack((self.B, self.R) + self.pool_shape + (self.C,),0)
            self._meta_cache[(self.B, self.R, self.C)] = type(self)._META_REG(arrange_BxRxlevels,
                                                                              arrange_BxRxlevels_stacked,
                                                                              flatten_all_box_ids, out_shape)
        self.meta = self._meta_cache[(self.B, self.R, self.C)]

    def _proluge(self,inputs):
        ## input processing and handling
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        # stop ROI gradients
        boxes = tf.stop_gradient(inputs[0])
        self.B = self.B or tf.shape(boxes)[0]
        self.R = self.R or tf.shape(boxes)[1]
        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:]
        self.C = self.C or tf.shape(feature_maps[0])[-1]
        self._set_multi_level_roi_pool_meta()
        return boxes, image_meta, feature_maps

    def _epiluge(self, pooled,box_to_level=None):
        return pooled

    def _get_box2level_matching(self,boxes,image_meta):
        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)
        return roi_level

    def _pyramid_pools_by_level(self,roi_level,boxes,feature_maps):
        with tf.device(self.dev_str):
            # Loop through levels and apply ROI pooling to each. P2 to P5.
            all_box_roi_feats = []
            box_to_level = []
            # for crop&resize
            flattend_boxes = tf.reshape(boxes, (self.B * self.R, 4))

            ## to avoid FPN data marshall to host, collect all ROI features from all levels, static output:
            # Nlevels x [Nrois,pool_h,pool_w,Nfeatures]
            for i, level in enumerate(range(2, 6)):
                # Crop and Resize
                # Result: [batch * num_boxes, pool_height, pool_width, channels]
                ## use box_ids to gather on all boxes
                feats = self._roi_align(feature_maps[i], flattend_boxes, self.meta.flatten_all_box_ids)
                # [batch, num_boxes_lvl_i, pool_height, pool_width, channels]
                all_box_roi_feats.append(tf.reshape(feats,self.meta.out_shape))
                box_to_level.append(tf.equal(roi_level, level))

            # feat_c contains all rois from all levels, shape: [batch, num_boxes_lvl_i, pool_height, pool_width, channels]
            feats_c = tf.concat(all_box_roi_feats,1)
            # ids_bool_c contains the dense boolean indicator for roi matching per level
            ids_bool_c = tf.concat(box_to_level,1)
            ## Recover ROI ids and reorder to match the original ROI order.:
            # ids are ordered in groupes matching their corresponding fpn level (i.e. original order is not
            # preserved between groups).
            with tf.device("/device:CPU:0"):
                if self._use_where:
                    ## This op may need to run on host: tf.where introduces dynamic shapes in the graph althogh in this case shapes
                    # are well defined. However since ids are boolean they are much smaller than the data itself

                    ## find which roi features to gather from each level in the concatinated tensor
                    ids = tf.reshape(tf.cast(tf.where(ids_bool_c),dtype=tf.int32),(self.R*self.B,2))
                    stacked_lvl_roi_mapping = tf.reshape(ids[:, 1], (self.B, self.R))
                    if self._use_memory_for_roi_id_recovery:
                        order_ids_to_invert = tf.reshape(tf.gather_nd(self.meta.arrange_BxRxlevels,ids), (self.B, self.R))
                else:
                    ## bool_ids version
                    stacked_lvl_roi_mapping = tf.reshape(self.meta.arrange_BxRxlevels_stacked[ids_bool_c], (self.B, self.R))
                    if self._use_memory_for_roi_id_recovery:
                        order_ids_to_invert = tf.reshape(self.meta.arrange_BxRxlevels[ids_bool_c], (self.B, self.R))

                if not self._use_memory_for_roi_id_recovery:
                    order_ids_to_invert = stacked_lvl_roi_mapping % self.R

                ## calculate the inverse permutation to for gathering rois in correct order
                order = []
                for b in range(self.B):
                    order.append(tf.reshape(tf.math.invert_permutation(order_ids_to_invert[b]), (1, order_ids_to_invert.shape[1])))
                order = tf.concat(order, 0)
                ## reorder ids (instead of data)
                stacked_level_mapping_reordered = tf.gather(stacked_lvl_roi_mapping, order, batch_dims=1)

            final_feats = tf.gather(feats_c, stacked_level_mapping_reordered, batch_dims=1)

            return final_feats,None

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes, image_meta, feature_maps = self._proluge(inputs)
        roi_level = self._get_box2level_matching(boxes,image_meta)
        if self.custom_op:
            import tf_falsh
            roi_level = tf.subtract(roi_level, 2)
            roi_level = tf.expand_dims(tf.cast(roi_level, tf.uint32), axis=2)
            num_boxes = tf.cast(tf.fill([self.B,1], self.R), tf.uint32)
            pooled_feature = tf_falsh.ops.pyramid_roi_align(feature_maps, num_boxes, boxes, roi_level,
                            roi_size = [self.pool_shape[0]*2, self.pool_shape[1]*2], post_roi_downscale_factor = 2)
            return pooled_feature
        else:
            pooled_list, box2lvl_list = self._pyramid_pools_by_level(roi_level,boxes,feature_maps)
            return self._epiluge(pooled_list, box2lvl_list)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )


############################################################
#  Detection Target Layer
############################################################

def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
    masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
           boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.cast(tf.where(non_zeros),tf.int32)[:, 0], axis=2,
                         name="trim_gt_masks")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.cast(tf.where(gt_class_ids < 0),tf.int32)[:, 0]
    non_crowd_ix = tf.cast(tf.where(gt_class_ids > 0),tf.int32)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Adding a noisy version of the GT bbox to the generated proposals allows the
    # detector to train on high quality proposals without relying on rpn quality.
    # i.e. rpn generates bg samples while fg samples are drawn from the
    # union set of proposals (rpn w/iou>0.5 + noisy_gt). noise std proportional to the gt size
    if config.GT_NOISE_STD >= 0:
        if config.GT_NOISE_STD > 0 :
            #replicated_gt = tf.repeat(gt_boxes,2,axis=0)
            replicated_gt = gt_boxes
            area_gt = (gt_boxes[:,3]-gt_boxes[:,1])*(gt_boxes[:,2]-gt_boxes[:,0])
            noisy_gt = replicated_gt + tf.truncated_normal(tf.shape(replicated_gt), stddev=tf.expand_dims(area_gt,1)*config.GT_NOISE_STD)
            noisy_gt = clip_boxes_graph(noisy_gt, np.array([0, 0, 1, 1], dtype=np.float32))
        else:
            noisy_gt = gt_boxes
        proposals = tf.stop_gradient(tf.concat((proposals,noisy_gt),axis=0))
    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [proposals, crowd_boxes]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.cast(tf.where(positive_roi_bool),tf.int32)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.cast(tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool)),tf.int32)[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / (1.0 - config.ROI_POSITIVE_RATIO)
    #negative_count = tf.maximum(tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32),1)
    if config.BIN_PADDING:
        negative_count = tf.maximum(tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32),1)
        ## try to fill with negative samples instead of padding - todo add positive instead?
        bins = tf.cast(tf.ceil(tf.cast(negative_count + positive_count,tf.float32) / config.BIN_PADDING),tf.int32)
        negative_count = tf.minimum(bins * config.BIN_PADDING - positive_count,tf.shape(negative_indices)[0])
        #([negative_count,positive_count],[negative_count,positive_count],'negetive/positive count')
    else:
        negative_count = config.TRAIN_ROIS_PER_IMAGE-positive_count
    #TODO replace with sample with replacemnet?
    negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn = lambda: tf.argmax(positive_overlaps, axis=1,output_type=tf.int32),
        false_fn = lambda: tf.constant([],dtype=tf.int32)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= np.array(config.BBOX_STD_DEV)

    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids,
                                     config.MASK_SHAPE)
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    if config.BIN_PADDING:
        # if not enought negative samples pad with zeros
        tot = tf.shape(rois)[0]
        P =  tf.maximum(tf.cast(tf.math.ceil( tf.cast(tot,tf.float32) / config.BIN_PADDING)*config.BIN_PADDING,tf.int32) - tot,0)
    else:
        P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    ## the following lines adds Reshape ops after Pads
    ## this is to call off dynamic shape after detection target layer
    shape = [config.TRAIN_ROIS_PER_IMAGE, 4]
    rois = tf.reshape(rois, shape)
    deltas = tf.reshape(deltas, shape)
    shape = [config.TRAIN_ROIS_PER_IMAGE]
    roi_gt_class_ids = tf.reshape(roi_gt_class_ids, shape)
    shape = [config.TRAIN_ROIS_PER_IMAGE] + config.MASK_SHAPE
    masks = tf.reshape(masks, shape)

    return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KE.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(
                w, x, y, z, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        if config.BIN_PADDING:
            return [(None, None, 4),  # rois
            (None, None),  # class_ids
            (None, None, 4),  # deltas
            (None, None, self.config.MASK_SHAPE[0],
             self.config.MASK_SHAPE[1])  # masks
             ]
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
             self.config.MASK_SHAPE[1])  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]


############################################################
#  Detection Layer
############################################################

def refine_detections_graph(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(tf.shape(probs)[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific * np.array(config.BBOX_STD_DEV))
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.cast(tf.where(class_ids > 0),tf.int32)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.cast(tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE),tf.int32)[:, 0]
        keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse.to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.cast(tf.where(tf.equal(pre_nms_class_ids, class_id)),tf.int32)[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=config.DETECTION_MAX_INSTANCES,
                iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int32)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.cast(tf.where(nms_keep > -1),tf.int32)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse.to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.dtypes.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


class DetectionLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        windows = image_meta[:, 7:11]
        image_shapes = image_meta[:, 4:6]
        normalized_windows = utils.batch_slice([windows, image_shapes], norm_boxes_graph, self.config.IMAGES_PER_GPU)

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, normalized_windows],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)


############################################################
#  Region Proposal Network (RPN)
############################################################

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    # TODO: check if stride of 2 causes alignment issues if the feature map
    # is not even.
    # Shared convolutional base of the RPN
    ### used to be 512 changed to 256 to match detectron
    shared = KL.Conv2D(RPN_STEM_C, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,kernel_initializer=INIT_CONV_KERNEL,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    if USE_BG_BIAS:
        bias_cls_init = np.array([1,0] * anchors_per_location)
    else:
        bias_cls_init = INIT_CLS_BIAS
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw',use_bias=True,
                  bias_initializer=keras.initializers.Constant(bias_cls_init),
                  kernel_initializer=INIT_FC_KERNEL)(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred',use_bias=True,
                  bias_initializer=keras.initializers.Constant(0),
                  kernel_initializer=INIT_BBOX_KERNEL)(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = KL.Input(shape=[None, None, depth],
                             name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  Feature Pyramid Network Heads
############################################################

def fpn_classifier_graph(dev_str, rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True, use_bn=True,
                         fc_layers_size=1024, train=False, build_roi_pool_fn=PyramidROIAlign):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    """
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    with tf.device(dev_str):
        pooled_features = build_roi_pool_fn([pool_size, pool_size], dev_str, name="roi_align_classifier")([rois, image_meta] + feature_maps)

    with tf.device(dev_str):
        # Two 1024 FC layers (implemented with Conv2D for consistency)
        x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid",
                                    use_bias=True,
                                    bias_initializer=keras.initializers.Constant(0),
                                    kernel_initializer=INIT_FC_KERNEL),
                               name="mrcnn_class_conv1")(pooled_features)
        if use_bn:
            x = KL.TimeDistributed(FrozenBatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
        x = KL.Activation('relu')(x)
        x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1),use_bias=True,
                                    bias_initializer=keras.initializers.Constant(0),
                                    kernel_initializer=INIT_FC_KERNEL),
                               name="mrcnn_class_conv2")(x)
        if use_bn:
            x = KL.TimeDistributed(FrozenBatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
        x = KL.Activation('relu')(x)

        shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                           name="pool_squeeze")(x)

        # Classifier head
        if USE_BG_BIAS:
            bias_cls_init = np.zeros(num_classes)
            bias_cls_init[0]=1
        else:
            bias_cls_init = INIT_CLS_BIAS

        mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes,use_bias=True,
                                    bias_initializer=keras.initializers.Constant(bias_cls_init),
                                    kernel_initializer=INIT_CLS_KERNEL),
                                                name='mrcnn_class_logits')(shared)
        if train:
            mrcnn_probs = None
        else:
            mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                        name="mrcnn_class")(mrcnn_class_logits)

        # BBox head
        # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
        x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear', use_bias=True,
                                        bias_initializer=keras.initializers.Constant(0),
                                        kernel_initializer=INIT_BBOX_KERNEL),
                               name='mrcnn_bbox_fc')(shared)
        # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
        mrcnn_bbox = KL.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(dev_str, num_classes, image_meta ,rois=None, feature_maps=None,pool_size=None,pooled_features=None,
                          train_bn=True,use_bn=True,build_roi_pool_fn=PyramidROIAlign):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers

    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    with tf.device(dev_str):
        if pooled_features is not None:
            x = pooled_features
        else:
            x = build_roi_pool_fn([pool_size, pool_size], dev_str, name="roi_align_mask")([rois, image_meta] + feature_maps)

    with tf.device(dev_str):
        # Conv layers
        x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same",
                      kernel_initializer=INIT_CONV_KERNEL),
                               name="mrcnn_mask_conv1")(x)
        if use_bn:
            x = KL.TimeDistributed(FrozenBatchNorm(),
                                   name='mrcnn_mask_bn1')(x, training=train_bn)
        x = KL.Activation('relu')(x)

        x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same",
                      kernel_initializer=INIT_CONV_KERNEL),
                               name="mrcnn_mask_conv2")(x)
        if use_bn:
            x = KL.TimeDistributed(FrozenBatchNorm(),
                                   name='mrcnn_mask_bn2')(x, training=train_bn)
        x = KL.Activation('relu')(x)
        x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same",
                      kernel_initializer=INIT_CONV_KERNEL),
                               name="mrcnn_mask_conv3")(x)
        if use_bn:
            x = KL.TimeDistributed(FrozenBatchNorm(),
                                   name='mrcnn_mask_bn3')(x, training=train_bn)
        x = KL.Activation('relu')(x)
        x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same",
                      kernel_initializer=INIT_CONV_KERNEL),
                               name="mrcnn_mask_conv4")(x)
        if use_bn:
            x = KL.TimeDistributed(FrozenBatchNorm(),
                                   name='mrcnn_mask_bn4')(x, training=train_bn)
        x = KL.Activation('relu')(x)
        x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu",
                      kernel_initializer=INIT_CONV_KERNEL),
                               name="mrcnn_mask_deconv")(x)
        x = KL.TimeDistributed(
                            KL.Conv2D(
                                    num_classes, (1, 1), strides=1, activation="sigmoid",
                                    use_bias=True,
                                    bias_initializer=keras.initializers.Constant(0),
                                    kernel_initializer=INIT_BBOX_KERNEL
                                ),name="mrcnn_mask"
                               )(x)
    return x


############################################################
#  Loss Functions
############################################################

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    if USE_L1_LOSS:
        return diff
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.cast(tf.where(K.not_equal(rpn_match, 0)),tf.int32)
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=False)
    if AVOID_LOSS_SWITCH:
        loss = tf.reduce_sum(loss) / tf.maximum(tf.cast(tf.size(loss),tf.float32),1.0)
    else:
        loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.cast(tf.where(K.equal(rpn_match, 1)),tf.int32)

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)
    if AVOID_LOSS_SWITCH:
        loss = tf.reduce_sum(loss) / tf.maximum(tf.cast(tf.size(loss),tf.float32), 1.0)
    else:
        loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))

    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """
    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    #target_class_ids = tf.cast(target_class_ids, 'int64')

    # Find predictions of classes that are not in the dataset.
    #pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    #pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    #loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    if AVOID_LOSS_SWITCH:
        return tf.reduce_sum(loss) / tf.maximum(tf.cast(tf.size(loss),tf.float32), 1.0)
    else:
        return  K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    is_fg_bool = target_class_ids > 0
    positive_roi_ix = tf.cast(tf.where(is_fg_bool),tf.int32)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int32)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    if AVOID_LOSS_SWITCH:
        loss = smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox)
        ## normalize bbox according to total number of proposals
        loss = tf.reduce_sum(loss)
        if SCALE_BY_NUM_GT:
            normalizer = SCALE_BY_NUM_GT * tf.maximum(tf.reduce_sum(tf.cast(is_fg_bool,tf.float32)),1.0)
        else:
            normalizer = tf.maximum(tf.cast(tf.size(target_class_ids),tf.float32),1.0)
        loss = loss / normalizer
    else:
        loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
        loss = K.mean(loss)

    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.cast(tf.where(target_class_ids > 0),tf.int32)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int32)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    if AVOID_LOSS_SWITCH:
        loss = K.binary_crossentropy(target=y_true, output=y_pred)
        loss = tf.reduce_sum(loss) / tf.maximum(tf.cast(tf.size(y_true),tf.float32), 1.0)
    else:
        loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
        loss = K.mean(loss)

    return loss


############################################################
#  Data Generator
############################################################
DEBUG=False
def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False,validation=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape

    # Random horizontal flips.
    # TODO: will be removed in a future update in favor of augmentation
    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad","Crop",
                           "Affine", "PiecewiseAffine","KeepSizeByResize"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        if DEBUG:
            print(f'\tid {image_id} size before and after augment:'
                  f' {image_shape[0]}x{image_shape[1]},{image.shape[0]}x{image.shape[1]}')
        # # Verify that shapes didn't change
        assert image.shape[2] == image_shape[2], "Augmentation shouldn't change image size"
        assert mask.shape[2] == mask_shape[2], "Augmentation shouldn't change mask size"
        assert mask.shape[:2] == image.shape[:2]
        # Change mask back to bool
        mask = mask.astype(np.bool)
    ## size manipulations
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM_VAL if validation else config.IMAGE_MIN_DIM_TRAIN,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(np.where(mask > 0, 1, 0), axis=(0, 1)) > 0
    assert len(_idx)>0, 'no masks left in resized sample'
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)
    if DEBUG:
        print(f'\tid {image_id} final shape and padding window: {image.shape[0]}x{image.shape[1]}\t {window}')
    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask


def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
    """Generate targets for training Stage 2 classifier and mask heads.
    This is not used in normal training. It's useful for debugging or to train
    the Mask RCNN heads without using the RPN head.

    Inputs:
    rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
    gt_class_ids: [instance count] Integer class IDs
    gt_boxes: [instance count, (y1, x1, y2, x2)]
    gt_masks: [height, width, instance count] Ground truth masks. Can be full
              size or mini-masks.

    Returns:
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Class-specific
            bbox refinements.
    masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Class specific masks cropped
           to bbox boundaries and resized to neural network output size.
    """
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(
        gt_class_ids.dtype)
    assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(
        gt_boxes.dtype)
    assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(
        gt_masks.dtype)

    # It's common to add GT Boxes to ROIs but we don't do that here because
    # according to XinLei Chen's paper, it doesn't help.

    # Trim empty padding in gt_boxes and gt_masks parts
    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, "Image must contain instances."
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]
    gt_masks = gt_masks[:, :, instance_ids]

    # Compute areas of ROIs and ground truth boxes.
    rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * \
        (rpn_rois[:, 3] - rpn_rois[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
        (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Compute overlaps [rpn_rois, gt_boxes]
    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i]
        overlaps[:, i] = utils.compute_iou(
            gt, rpn_rois, gt_box_area[i], rpn_roi_area)

    # Assign ROIs to GT boxes
    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
    rpn_roi_iou_max = overlaps[np.arange(
        overlaps.shape[0]), rpn_roi_iou_argmax]
    # GT box assigned to each ROI
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

    # Positive ROIs are those with >= 0.5 IoU with a GT box.
    fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

    # Negative ROIs are those with max IoU 0.1-0.5 (hard example mining)
    # TODO: To hard example mine or not to hard example mine, that's the question
    # bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

    # Subsample ROIs. Aim for 33% foreground.
    # FG
    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids
    # BG
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    # Combine indices of ROIs to keep
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    # Need more?
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        # Looks like we don't have enough samples to maintain the desired
        # balance. Reduce requirements and fill in the rest. This is
        # likely different from the Mask RCNN paper.

        # There is a small chance we have neither fg nor bg samples.
        if keep.shape[0] == 0:
            # Pick bg regions with easier IoU threshold
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            # Fill the rest with repeated bg rois.
            keep_extra_ids = np.random.choice(
                keep_bg_ids, remaining, replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
        "keep doesn't match ROI batch size {}, {}".format(
            keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

    # Reset the gt boxes assigned to BG ROIs.
    rpn_roi_gt_boxes[keep_bg_ids, :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0

    # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
    roi_gt_assignment = rpn_roi_iou_argmax[keep]

    # Class-aware bbox deltas. [y, x, log(h), log(w)]
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE,
                       config.NUM_CLASSES, 4), dtype=np.float32)
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(
        rois[pos_ids], roi_gt_boxes[pos_ids, :4])
    # Normalize bbox refinements
    bboxes /= np.array(config.BBOX_STD_DEV)

    # Generate class-specific target masks
    masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES),
                     dtype=np.float32)
    for i in pos_ids:
        class_id = roi_gt_class_ids[i]
        assert class_id > 0, "class id must be greater than 0"
        gt_id = roi_gt_assignment[i]
        class_mask = gt_masks[:, :, gt_id]

        if config.USE_MINI_MASK:
            # Create a mask placeholder, the size of the image
            placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
            # GT box
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1
            # Resize mini mask to size of GT box
            placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = \
                np.round(utils.resize(class_mask, (gt_h, gt_w))).astype(bool)
            # Place the mini batch in the placeholder
            class_mask = placeholder

        # Pick part of the mask and resize it
        y1, x1, y2, x2 = rois[i].astype(np.int32)
        m = class_mask[y1:y2, x1:x2]
        mask = utils.resize(m, config.MASK_SHAPE)
        masks[i, :, :, class_id] = mask

    return rois, roi_gt_class_ids, bboxes, masks


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:,0]
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= np.array(config.RPN_BBOX_STD_DEV)
        ix += 1

    return rpn_match, rpn_bbox


def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    """Generates ROI proposals similar to what a region proposal network
    would generate.

    image_shape: [Height, Width, Depth]
    count: Number of ROIs to generate
    gt_class_ids: [N] Integer ground truth class IDs
    gt_boxes: [N, (y1, x1, y2, x2)] Ground truth boxes in pixels.

    Returns: [count, (y1, x1, y2, x2)] ROI boxes in pixels.
    """
    # placeholder
    rois = np.zeros((count, 4), dtype=np.int32)

    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(0.9 * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # random boundaries
        r_y1 = max(gt_y1 - h, 0)
        r_y2 = min(gt_y2 + h, image_shape[0])
        r_x1 = max(gt_x1 - w, 0)
        r_x2 = min(gt_x2 + w, image_shape[1])

        # To avoid generating boxes with zero area, we generate double what
        # we need and filter out the extra. If we get fewer valid boxes
        # than we need, we loop and try again.
        while True:
            y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
            x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
            # Filter out zero area boxes
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                        threshold][:rois_per_box]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                        threshold][:rois_per_box]
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
        # into x1, y1, x2, y2 order
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        box_rois = np.hstack([y1, x1, y2, x2])
        rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

    # Generate random ROIs anywhere in the image (10% of count)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # To avoid generating boxes with zero area, we generate double what
    # we need and filter out the extra. If we get fewer valid boxes
    # than we need, we loop and try again.
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        # Filter out zero area boxes
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                    threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                    threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break

    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
    # into x1, y1, x2, y2 order
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois
    return rois


class AnchorsLayer(tf.keras.layers.Layer):
    def __init__(self, name="anchors", **kwargs):
        super(AnchorsLayer, self).__init__(name=name, **kwargs)

    def call(self, anchor):
        return anchor

    def get_config(self):
        config = super(AnchorsLayer, self).get_config()
        return config


class DataGenerator(keras.utils.Sequence):
    def __init__(self, dataset, config, shuffle=True, augment=False, augmentation=None,
                random_rois=0, batch_size=1, detection_targets=False,validation=False,
                no_augmentation_sources=None,anchor_generator=None):
        """A generator that returns images and corresponding target class ids,
        bounding box deltas, and masks.

        dataset: The Dataset object to pick data from
        config: The model config object
        shuffle: If True, shuffles the samples before every epoch
        augment: (deprecated. Use augmentation instead). If true, apply random
            image augmentation. Currently, only horizontal flipping is offered.
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
            For example, passing imgaug.augmenters.Fliplr(0.5) flips images
            right/left 50% of the time.
        random_rois: If > 0 then generate proposals to be used to train the
                    network classifier and mask heads. Useful if training
                    the Mask RCNN part without the RPN.
        batch_size: How many images to return in each call
        detection_targets: If True, generate detection targets (class IDs, bbox
            deltas, and masks). Typically for debugging or visualizations because
            in trainig detection targets are generated by DetectionTargetLayer.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.

        Returns a Python generator. Upon calling next() on it, the
        generator returns two lists, inputs and outputs. The contents
        of the lists differs depending on the received arguments:
        inputs list:
        - images: [batch, H, W, C]
        - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
        - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
        - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
        - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
        - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                    are those of the image unless use_mini_mask is True, in which
                    case they are defined in MINI_MASK_SHAPE.

        outputs list: Usually empty in regular training. But if detection_targets
            is True then the outputs list contains target class_ids, bbox deltas,
            and masks.
        """
        self.rpn_only=getattr(config,'RPN_ONLY',False)
        self.image_ids = np.copy(dataset.image_ids)
        if shuffle == True:
            np.random.shuffle(self.image_ids)
        self.dataset = dataset
        self.config = config
        self.error_count = 0
        #function that takes the input shape
        self.anchors = anchor_generator
        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        if anchor_generator is None:
            self.backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
            self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                config.RPN_ANCHOR_RATIOS,
                                                self.backbone_shapes,
                                                config.BACKBONE_STRIDES,
                                                config.RPN_ANCHOR_STRIDE)

        self.shuffle = shuffle
        self.augment = augment
        self.augmentation = augmentation
        self.random_rois = random_rois
        self.batch_size = batch_size
        self.detection_targets = detection_targets
        self.no_augmentation_sources = no_augmentation_sources or []
        self.validation = validation

    def __len__(self):
        return len(self.image_ids) // self.batch_size

    def __getitem__(self, idx):
        idx_start=idx*self.batch_size + self.error_count
        idx_stop=(idx+1)*self.batch_size + self.error_count  #adding a sample to make sure
        assert idx_stop >= idx_start+self.batch_size
        buffer_for_loading_errors=min(len(self.image_ids)-idx_stop,5)
        batch_img_ids=self.image_ids[idx_start:idx_stop+buffer_for_loading_errors]
        if buffer_for_loading_errors < 5:
            batch_img_ids = np.concatenate((batch_img_ids, self.image_ids[:5-buffer_for_loading_errors]))

        return self.data_generator(batch_img_ids)

    def data_generator(self,image_ids):
        b=0
        while b < self.batch_size:
            try:
                # Get GT bounding boxes and masks for image.
                image_id = image_ids[b]
                # If the image source is not to be augmented pass None as augmentation
                if self.dataset.image_info[image_id]['source'] in self.no_augmentation_sources:
                    image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                    load_image_gt(self.dataset, self.config, image_id, augment=self.augment,
                              augmentation=None,
                                use_mini_mask=self.config.USE_MINI_MASK,validation=self.validation)
                else:
                    image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                        load_image_gt(self.dataset, self.config, image_id, augment=self.augment,
                                    augmentation=self.augmentation,
                                    use_mini_mask=self.config.USE_MINI_MASK,validation=self.validation)

                # Skip images that have no instances. This can happen in cases
                # where we train on a subset of classes and the image doesn't
                # have any of the classes we care about.
                if not np.any(gt_class_ids > 0):
                    assert 0,'no gt class'
                    continue

                # RPN Targets
                if callable(self.anchors):
                    ## TODO anchors should be shared between all batch elements this currently works for batch size 1
                    anchors = self.anchors(image.shape,normalized=False)
                else:
                    anchors = self.anchors
                rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors, gt_class_ids, gt_boxes, self.config)

                # Mask R-CNN Targets
                if self.random_rois:
                    rpn_rois = generate_random_rois(
                            image.shape, self.random_rois, gt_class_ids, gt_boxes)
                    if self.detection_targets:
                        rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
                            build_detection_targets(
                                    rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)

                # Init batch arrays
                if b == 0:
                    batch_image_meta = np.zeros(
                            (self.batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                    batch_rpn_match = np.zeros(
                            #[self.batch_size, self.anchors.shape[0], 1], dtype=rpn_match.dtype)
                        [self.batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                    batch_rpn_bbox = np.zeros(
                            [self.batch_size, self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                    batch_images = np.zeros(
                            (self.batch_size,) + image.shape, dtype=np.float32)
                    batch_gt_class_ids = np.zeros(
                            (self.batch_size, self.config.MAX_GT_INSTANCES), dtype=np.int32)
                    batch_gt_boxes = np.zeros(
                            (self.batch_size, self.config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                    batch_gt_masks = np.zeros(
                            (self.batch_size, gt_masks.shape[0], gt_masks.shape[1],
                            self.config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
                    if self.random_rois:
                        batch_rpn_rois = np.zeros(
                            (self.batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                        if self.detection_targets:
                            batch_rois = np.zeros(
                                    (self.batch_size,) + rois.shape, dtype=rois.dtype)
                            batch_mrcnn_class_ids = np.zeros(
                                    (self.batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                            batch_mrcnn_bbox = np.zeros(
                                    (self.batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                            batch_mrcnn_mask = np.zeros(
                                    (self.batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

                # If more instances than fits in the array, sub-sample from them.
                if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
                    ids = np.random.choice(
                        np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
                    gt_class_ids = gt_class_ids[ids]
                    gt_boxes = gt_boxes[ids]
                    gt_masks = gt_masks[:, :, ids]

                # Add to batch
                batch_image_meta[b] = image_meta
                batch_rpn_match[b] = rpn_match[:, np.newaxis]
                batch_rpn_bbox[b] = rpn_bbox
                batch_images[b] = mold_image(image, self.config)
                batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
                batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
                batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
                if self.random_rois:
                    batch_rpn_rois[b] = rpn_rois
                    if self.detection_targets:
                        batch_rois[b] = rois
                        batch_mrcnn_class_ids[b] = mrcnn_class_ids
                        batch_mrcnn_bbox[b] = mrcnn_bbox
                        batch_mrcnn_mask[b] = mrcnn_mask

                b += 1

                # Batch full?
                if b >= self.batch_size:
                    inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox]
                    if callable(self.anchors):
                        inputs += [np.expand_dims(self.anchors(image.shape,normalized=True),0)]
                    if not self.rpn_only:
                        inputs += [batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                    outputs = []

                    if self.random_rois:
                        inputs.extend([batch_rpn_rois])
                        if self.detection_targets:
                            inputs.extend([batch_rois])
                            # Keras requires that output and targets have the same number of dimensions
                            batch_mrcnn_class_ids = np.expand_dims(batch_mrcnn_class_ids, -1)
                            outputs.extend([batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

                    ## @TF_Dataset_from_generator
                    #return tuple(inputs), tuple(outputs)
                    return inputs, outputs

            except (GeneratorExit, KeyboardInterrupt):
                raise
            except:
                # Log it and skip the image
                logging.exception("Error processing image {}".format(self.dataset.image_info[image_id]))
                self.error_count += 1
                if self.error_count > 5:
                    raise

    def on_epoch_end(self):
        self.error_count=0
        if self.shuffle == True:
            np.random.shuffle(self.image_ids)

class SGD_with_colocate_grad(keras.optimizers.SGD):
    def __init__(self,
               momentum_const=False,
               **kwargs):
        super(SGD_with_colocate_grad, self).__init__(**kwargs)
        self.momentum_const=momentum_const

    def get_gradients(self, loss, params):
        """Returns gradients of `loss` with respect to `params`.
        Arguments:
          loss: Loss tensor.
          params: List of variables.
        Returns:
          List of gradient tensors.
        Raises:
          ValueError: In case any gradient cannot be computed (e.g. if gradient
          function not implemented).
        """
        params = nest.flatten(params)
        with backend.get_graph().as_default(), backend.name_scope(self._name +
                                                                "/gradients"):
            grads = gradients.gradients(loss, params, colocate_gradients_with_ops=True)
            for grad, param in zip(grads, params):
                if grad is None:
                    raise ValueError("Variable {} has `None` for gradient. "
                                     "Please make sure that all of your ops have a "
                                     "gradient defined (i.e. are differentiable). "
                                     "Common ops without gradient: "
                                     "K.argmax, K.round, K.eval.".format(param))
            if hasattr(self, "clipnorm"):
                grads = [clip_ops.clip_by_norm(g, self.clipnorm) for g in grads]
            if hasattr(self, "clipvalue"):
                if self.clipvalue:
                    grads = [
                        clip_ops.clip_by_value(g, -self.clipvalue, self.clipvalue)
                        for g in grads
                    ]
        return grads
    def _create_hypers(self):
      if self._hypers_created:
        return
      # Iterate hyper values deterministically.
      for name, value in sorted(self._hyper.items()):
        if isinstance(
            value, (ops.Tensor, tf_variables.Variable)) or callable(value):
          continue
        else:
          if self.momentum_const == True and name == 'momentum':
              self._hyper[name] = backend.constant(value)
          else:
              self._hyper[name] = self.add_weight(
                  name,
                  shape=[],
                  trainable=False,
                  initializer=value,
                  aggregation=tf_variables.VariableAggregation.ONLY_FIRST_REPLICA)
      self._hypers_created = True



############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, dev_str, mode, config, model_dir, hvd=None):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.dynamic_shapes = config.IMAGE_RESIZE_MODE in ['no_pad','pad64']
        self.dynamic_anchors = config.DYNAMIC_ANCHORS
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(dev_str, mode=mode, config=config)
        self.hvd = hvd

    def build(self, dev_str, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']
        rpn_only=getattr(self.config,'RPN_ONLY',False)

        ## select specific ROI pooling function
        build_roi_pool_fn = PyramidROIAlign
        if mode == 'training':
            build_roi_pool_fn = partial_class(build_roi_pool_fn,B=self.config.IMAGES_PER_GPU,
                                              R=self.config.TRAIN_ROIS_PER_IMAGE,C=self.config.TOP_DOWN_PYRAMID_SIZE,
                                              custom_op=self.config.PYRAMID_ROI_CUSTOM_OP)
        # Image size must be dividable by 2 multiple times
        if self.dynamic_shapes :
            h, w = None, None
        else:
            h, w = config.IMAGE_SHAPE[:2]

            if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
                raise Exception("Image size must be dividable by 2 at least 6 times "
                                "to avoid fractions when downscaling and upscaling."
                                "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(
            batch_shape=[config.BATCH_SIZE,h, w, config.IMAGE_SHAPE[2]], name="input_image")
        input_image_meta = KL.Input(batch_shape=[config.BATCH_SIZE, config.IMAGE_META_SIZE],
                                    name="input_image_meta")

        if mode == "training":
            if self.dynamic_anchors:
                n_anchors = None
            else:
                anchors = anchors = self.get_anchors(np.array(config.IMAGE_SHAPE))
                n_anchors = anchors.shape[0]
            # RPN GT
            input_rpn_match = KL.Input(
                batch_shape=[config.BATCH_SIZE, n_anchors, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(
                batch_shape=[config.BATCH_SIZE,config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                batch_shape=[config.BATCH_SIZE, config.MAX_GT_INSTANCES], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                batch_shape=[config.BATCH_SIZE, config.MAX_GT_INSTANCES, 4], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(
                x, K.shape(input_image)[1:3]))(input_gt_boxes)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    batch_shape=(config.BATCH_SIZE,) + config.MINI_MASK_SHAPE + (config.MAX_GT_INSTANCES,),
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    batch_shape=(config.BATCH_SIZE,) +config.IMAGE_SHAPE+(config.MAX_GT_INSTANCES,),
                    name="input_gt_masks", dtype=bool)

        if self.dynamic_anchors or mode == "inference":
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")
 
        with tf.device(dev_str):
            # Build the shared convolutional layers.
            # Bottom-up Layers
            # Returns a list of the last layers of each stage, 5 in total.
            # Don't create the thead (stage 5), so we pick the 4th item in the list.
            if 'kapp_' in config.BACKBONE:
                ## used to integrate keras model zoo backbone
                KL.BatchNormalization = FrozenBatchNorm
                base_model = getattr(keras.applications,config.BACKBONE.split('_')[-1])(include_top=False,
                                                                                        input_tensor=input_image,
                                                                                        layers=KL,
                                                                                        weights=None if self.mode == "inference" else "imagenet")
                # currently assuming block outputs are always marked with _out
                layers = [l for l in base_model.layers if '_out' in l.name]
                outs = []
                # remove intermidiate outputs, keep last 4 blocks
                for i in range(4):
                    last= layers.pop(-1)
                    while any(re.fullmatch(last.name[:5] + '.*', l.name) for l in layers):
                        layers.pop(-1)
                    outs.append(last.output)

                C5, C4, C3, C2 =  outs[:4]
            elif callable(config.BACKBONE):
                _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
                                                    train_bn=config.TRAIN_BN)
            else:
                _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE,
                                                 stage5=True, train_bn=config.TRAIN_BN)
            # Top-down Layers
            # TODO: add assert to varify feature map sizes match what's in config
            P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
            P4 = KL.Add(name="fpn_p4add")([
                KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
            P3 = KL.Add(name="fpn_p3add")([
                KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
            P2 = KL.Add(name="fpn_p2add")([
                KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
            # Attach 3x3 conv to all P layers to get the final feature maps.
            P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
            P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
            P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
            P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
            # P6 is used for the 5th anchor scale in RPN. Generated by
            # subsampling from P5 with stride of 2.
            P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training" and not self.dynamic_anchors:
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchor_layer = AnchorsLayer(name="anchors")
            anchors = anchor_layer(anchors)
        else:
            anchors = input_anchors

        # RPN Model
        with tf.device(dev_str):
            rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                                  len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
            # Loop through pyramid layers
            layer_outputs = []  # list of lists
            for p in rpn_feature_maps:
                layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE
        if not rpn_only:
            rpn_rois = ProposalLayer(
                proposal_count=proposal_count,
                nms_threshold=config.RPN_NMS_THRESHOLD,
                dev_str=dev_str,
                name="ROI",
                config=config, combined_nms=config.COMBINED_NMS_OP)([rpn_class, rpn_bbox, anchors, input_image_meta])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = KL.Lambda(
                lambda x: parse_image_meta_graph(x)["active_class_ids"]
                )(input_image_meta)
            if not rpn_only:
                if not config.USE_RPN_ROIS:
                    # Ignore predicted ROIs and use ROIs provided as an input.
                    input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                        name="input_roi", dtype=np.int32)
                    # Normalize coordinates
                    target_rois = KL.Lambda(lambda x: norm_boxes_graph(
                        x, K.shape(input_image)[1:3]))(input_rois)
                else:
                    target_rois = rpn_rois

                # Generate detection targets
                # Subsamples proposals and generates target outputs for training
                # Note that proposal class IDs, gt_boxes, and gt_masks are zero
                # padded. Equally, returned rois and targets are zero padded.
                rois, target_class_ids, target_bbox, target_mask =\
                    DetectionTargetLayer(config, name="proposal_targets")([
                        target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])
                # Network Heads
                # TODO: verify that this handles zero padded ROIs
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                    fpn_classifier_graph(dev_str, rois, mrcnn_feature_maps, input_image_meta,
                                        config.POOL_SIZE, config.NUM_CLASSES,
                                        train_bn=config.TRAIN_BN,use_bn=config.BN_ON_DETECTOR,
                                        fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE,train=False,build_roi_pool_fn=build_roi_pool_fn)

                mrcnn_mask = build_fpn_mask_graph(dev_str, config.NUM_CLASSES, input_image_meta,rois, mrcnn_feature_maps,
                                                config.MASK_POOL_SIZE, train_bn=config.TRAIN_BN,use_bn=config.BN_ON_MASK,build_roi_pool_fn=build_roi_pool_fn)

                # TODO: clean up (use tf.identify if necessary)
                output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class])
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            inputs = [input_image, input_image_meta,
                      input_rpn_match, input_rpn_bbox]
            if self.dynamic_anchors:
                inputs += [input_anchors]
            if not rpn_only:
                class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                    [target_class_ids, mrcnn_class_logits, active_class_ids])
                bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                    [target_bbox, target_class_ids, mrcnn_bbox])
                mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                    [target_mask, target_class_ids, mrcnn_mask])

                inputs+=[input_gt_class_ids, input_gt_boxes, input_gt_masks]

            # Model
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            if rpn_only:
                outputs = [rpn_class, rpn_bbox, rpn_class_loss, rpn_bbox_loss]
            else:
                outputs = [rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            _, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(dev_str, rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,use_bn=config.BN_ON_DETECTOR,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE,train=False,build_roi_pool_fn=build_roi_pool_fn)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = build_fpn_mask_graph(dev_str, config.NUM_CLASSES, input_image_meta,
                            detection_boxes, mrcnn_feature_maps,
                            config.MASK_POOL_SIZE, train_bn=config.TRAIN_BN,use_bn=config.BN_ON_MASK,build_roi_pool_fn=build_roi_pool_fn)

            model = KM.Model([input_image, input_image_meta, input_anchors],
                             [detections, mrcnn_class, mrcnn_bbox,
                                 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                             name='mask_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def save(self,path=None):
        keras_model = self.keras_model.inner_model if hasattr(self.keras_model, "inner_model") else self.keras_model
        keras_model.save_weights(path if path else os.path.join(self.log_dir,f'manual_checkpoint_{self.epoch:04}.h5'))

    def load_weights(self, filepath, by_name=False, exclude=None,resume=True,verbose=False):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            by_name = True
            layers = filter(lambda l: not bool(re.match(exclude, l.name)), layers)
            if verbose:
                print('excluded layers:', [layer.name for layer in filter(lambda l: bool(re.match(exclude, l.name)), layers)])
        if verbose:
            print('included layers:', [layer.name for layer in layers])

        if by_name:
            load_weights=saving.load_weights_from_hdf5_group_by_name
        else:
            load_weights=saving.load_weights_from_hdf5_group
        load_weights(f, layers)

        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        if resume:
            self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
                                 'releases/download/v0.2/'\
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path

    def compile(self, dev_str, learning_rate, momentum, loss_weights=None, run_options=None, run_metadata=None):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        with tf.device(dev_str):
            rpn_only=getattr(self.config,'RPN_ONLY',False)
            loss_weights=loss_weights or self.config.LOSS_WEIGHTS
            # Optimizer object
            if self.config.OPTIMIZER == 'SGD':
                opt_kwargs = dict(lr=learning_rate, momentum=momentum, momentum_const=self.config.LEARNING_MOMENTUM_CONST)
                opt = SGD_with_colocate_grad
            elif self.config.OPTIMIZER == 'Adam':
                opt_kwargs = dict(lr=learning_rate, beta_1=momentum)
                opt = keras.optimizers.Adam
            elif self.config.OPTIMIZER == 'RAdam':
                opt_kwargs = dict(lr=learning_rate, beta_1=momentum)
                from radam import RectifiedAdam
                opt = RectifiedAdam
            if self.config.GRADIENT_CLIP_NORM and not self.config.MIXED_PRECISION:
                opt_kwargs.update({'clipnorm': self.config.GRADIENT_CLIP_NORM})

            if self.config.MIXED_PRECISION:
                assert 0, 'need some work, see PR in https://github.com/tensorflow/tensorflow/pull/31578'
                # grad clipnorm not supported
                opt_ = opt
                opt = lambda **kwargs: tf.train.experimental.enable_mixed_precision_graph_rewrite(opt_(**kwargs))

            optimizer = opt(**opt_kwargs)

            if self.hvd:
                optimizer = self.hvd.DistributedOptimizer(optimizer)

            # Add Losses
            # First, clear previously set losses to avoid duplication
            loss_names = ["rpn_class_loss",  "rpn_bbox_loss"]
            if not rpn_only:
                loss_names+= ["mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
            losses = []
            for name in loss_names:
                layer = self.keras_model.get_layer(name)
                if layer.output in self.keras_model.losses:
                    continue
                loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * loss_weights.get(name, 1.))
                losses.append(loss)
            # Add L2 Regularization
            # Skip gamma and beta weights of batch normalization layers.
            reg_losses = [
                keras.regularizers.l2(self.config.WEIGHT_DECAY * 0.5)(w)
                for w in self.keras_model.trainable_weights
                if not any([ex in w.name for ex in ['gamma', 'beta']]) ]
            reg = tf.add_n(reg_losses,name='wd')
            losses.append(reg)
            self.keras_model.add_metric(reg, name='wd', aggregation='mean')
            self.keras_model.add_loss(tf.add_n(losses,name='global_loss'))
            # Compile
            self.keras_model.compile(
                optimizer=optimizer, experimental_run_tf_function=False, options=run_options, run_metadata=run_metadata)

            # Add metrics for losses
            for name in loss_names:
                if name in self.keras_model.metrics_names:
                    continue
                layer = self.keras_model.get_layer(name)
                loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * loss_weights.get(name, 1.))
                self.keras_model.add_metric(loss, name=name, aggregation='mean')

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model' or layer.__class__.__name__ == 'Sequential':
                if verbose:
                    print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4,verbose=verbose)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, dev_str, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None,
              loss_weights=None, dump_tf_timeline=False, profile=False):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
	    custom_callbacks: Optional. Add custom callbacks to be called
	        with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # only rpn:
            "rpn": r"(rpn\_.*)|(fpn\_.*)",
            # only mask:
            "mrcnn": r"(mrcnn\_.*)",
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "2+": r"(res2.*)|(bn2.*)|(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(conv[2-5].*)",
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(conv[3-5].*)",
            "3+nobn": r"(res3.*)|(res4.*)|(res5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(conv[3-5].*conv)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = DataGenerator(train_dataset, self.config, shuffle=self.config.DETERMINISTIC,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE,
                                         no_augmentation_sources=no_augmentation_sources,
                                         anchor_generator = self.get_anchors if self.dynamic_anchors else None)
        val_generator = DataGenerator(val_dataset, self.config, shuffle=self.config.DETERMINISTIC,
                                       batch_size=self.config.BATCH_SIZE,validation=True,
                                       anchor_generator = self.get_anchors if self.dynamic_anchors else None)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = []
        is_master = True
        if self.config.DETERMINISTIC:
            workers =  1
        else:
            workers = multiprocessing.cpu_count()
        use_multiprocessing = False # hanging issue when set to True
        if self.hvd:
            callbacks += [
                self.hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                self.hvd.callbacks.MetricAverageCallback()
                ]
            is_master = self.hvd.local_rank() == 0
            workers=workers//self.hvd.local_size()

        if is_master:
            if not profile and not dump_tf_timeline:
                callbacks += [
                    keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0,
                                                write_graph=True, write_images=False, write_grads=False)
                ]
            callbacks += [
                keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                                save_weights_only=True, period=5),
                keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                                save_weights_only=True, save_best_only=True,
                                                monitor='val_loss', mode='min'),
            ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        if is_master:
            log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
            log("Checkpoint Path: {}".format(self.checkpoint_path))
            experiment_dir = os.path.abspath(os.path.join(self.checkpoint_path,os.path.pardir))
            self.config.dump(experiment_dir,'train_config.json')
        self.set_trainable(layers,verbose=int(is_master))

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        if dump_tf_timeline:
            self.compile(dev_str, learning_rate, self.config.LEARNING_MOMENTUM, loss_weights=loss_weights, run_options=run_options, run_metadata=run_metadata)
        else:
            self.compile(dev_str, learning_rate, self.config.LEARNING_MOMENTUM, loss_weights=loss_weights)

        with tf.device(dev_str):
            from functools import partial
            keras_model_fit = partial(self.keras_model.fit,
                train_generator,
                verbose=int(is_master),
                initial_epoch=self.epoch,
                epochs=epochs,
                steps_per_epoch=self.config.STEPS_PER_EPOCH,
                callbacks=callbacks,
                max_queue_size=5*workers,
                workers=workers,
                use_multiprocessing=use_multiprocessing)

            if not profile:
                keras_model_fit(validation_data=val_generator, validation_steps=self.config.VALIDATION_STEPS)
            else:
                keras_model_fit()

            if dump_tf_timeline:
                tl = timeline.Timeline(step_stats=run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open('timeline_mrcnn.json', 'w') as f:
                    f.write(ctf)

        self.epoch = max(self.epoch, epochs)
        if is_master:
            self.save()

    def mold_inputs(self, images,validation=False):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM_VAL if validation else self.config.IMAGE_MIN_DIM_TRAIN,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images,validation=True)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def detect_molded(self, molded_images, image_metas, verbose=0):
        """Runs the detection pipeline, but expect inputs that are
        molded already. Used mostly for debugging and inspecting
        the model.

        molded_images: List of images loaded using load_image_gt()
        image_metas: image meta data, also returned by load_image_gt()

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(molded_images) == self.config.BATCH_SIZE,\
            "Number of images must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(molded_images)))
            for image in molded_images:
                log("image", image)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, "Images must have the same size"

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(molded_images):
            window = [0, 0, image.shape[0], image.shape[1]]
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       window)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def get_anchors(self, image_shape, normalized=True):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            unnormalized_anchors = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            # Normalize coordinates
            normalized_anchors = utils.norm_boxes(unnormalized_anchors, image_shape[:2])
            self._anchor_cache[tuple(image_shape)] = unnormalized_anchors,normalized_anchors

        return self._anchor_cache[tuple(image_shape)][1] if normalized else self._anchor_cache[tuple(image_shape)][0]

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

    def run_graph(self, images, outputs, image_metas=None):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        image_metas: If provided, the images are assumed to be already
            molded (i.e. resized, padded, and normalized)

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        # Prepare inputs
        if image_metas is None:
            molded_images, image_metas, _ = self.mold_inputs(images)
        else:
            molded_images = images
        image_shape = molded_images[0].shape
        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        model_in = [molded_images, image_metas, anchors]

        # Run inference
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta


def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed values.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    if config.BGR:
        images = images[:, :, ::-1]
    return images.astype(np.float32) - np.array(config.MEAN_PIXEL)


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    unnormalized_images = (normalized_images + np.array(config.MEAN_PIXEL)).astype(np.uint8)
    if config.BGR:
        # return to rgb
        unnormalized_images = unnormalized_images[:, :, ::-1]
    return unnormalized_images


############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)

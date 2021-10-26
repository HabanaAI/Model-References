# Copyright 2018 Google. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - Ported to TF 2.2
# - Removed mlp_log
# - Removed topk_mask
# - Renamed ssd_constants to constants
# - Renamed ssd_architecture to architecture
# - Removed non_max_suppression
# - Added global_batch_size parameter
# - Removed hparams
# - Removed support for TPU
# - Updated _filter_scores to use tf.math.greater_equal
# - Refactored concat_outputs function
# - Added custom _softmax_cross_entropy_mme optimized for HPU
# - Refactored _classification_loss function and added cast to float to neg_sorted_cross_indices
# - transpose_input flag removed from model
# - Updated features calculation to use multiplication instead of division
# - use_bfloat16 parameter changed to dtype
# - Added multi-node training on Horovod
# - Removed obsolete todos
# - Added box_loss and class_loss to TF summary
# - Added support for saving TF summary
# - Removed weight_decay loss from total loss calculation (fwd only)
# - Formatted with autopep8
# - Removed __future__ imports
# - Added absolute paths in imports

"""Model defination for the SSD Model.

Defines model_fn of SSD for TF Estimator. The model_fn includes SSD
model architecture, loss function, learning rate schedule, and evaluation
procedure.

T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar
Focal Loss for Dense Object Detection. arXiv:1708.02002
"""

import tensorflow.compat.v1 as tf

from TensorFlow.computer_vision.SSD_ResNet34.object_detection import box_coder
from TensorFlow.computer_vision.SSD_ResNet34.object_detection import box_list
from TensorFlow.computer_vision.SSD_ResNet34.object_detection import faster_rcnn_box_coder

from tensorflow.python.estimator import model_fn as model_fn_lib


from TensorFlow.computer_vision.SSD_ResNet34 import dataloader
from TensorFlow.computer_vision.SSD_ResNet34 import architecture
from TensorFlow.computer_vision.SSD_ResNet34 import constants


def select_top_k_scores(scores_in, pre_nms_num_detections=5000):
    """Select top_k scores and indices for each class.

    Args:
      scores_in: a Tensor with shape [batch_size, N, num_classes], which stacks
        class logit outputs on all feature levels. The N is the number of total
        anchors on all levels. The num_classes is the number of classes predicted
        by the model.
      pre_nms_num_detections: Number of candidates before NMS.

    Returns:
      scores and indices: Tensors with shape [batch_size, pre_nms_num_detections,
        num_classes].
    """
    scores_trans = tf.transpose(scores_in, perm=[0, 2, 1])

    top_k_scores, top_k_indices = tf.nn.top_k(
        scores_trans, k=pre_nms_num_detections, sorted=True)

    return tf.transpose(top_k_scores, [0, 2, 1]), tf.transpose(
        top_k_indices, [0, 2, 1])


def _filter_scores(scores, boxes, min_score=constants.MIN_SCORE):
    mask = tf.math.greater_equal(scores, min_score)
    scores = tf.where(mask, scores, tf.zeros_like(scores))
    boxes = tf.where(
        tf.tile(tf.expand_dims(mask, 2), (1, 1, 4)), boxes, tf.zeros_like(boxes))
    return scores, boxes


def concat_outputs(cls_outputs, box_outputs, transpose):
    """Concatenate predictions into a single tensor.

    This function takes the dicts of class and box prediction tensors and
    concatenates them into a single tensor for comparison with the ground truth
    boxes and class labels.
    Args:
      cls_outputs: an OrderDict with keys representing levels and values
        representing logits in [batch_size, height, width,
        num_anchors * num_classses].
      box_outputs: an OrderDict with keys representing levels and values
        representing box regression targets in
        [batch_size, height, width, num_anchors * 4].
      transpose: it is required to transpose boxes and classes in order to match
        order from FasterRcnnBoxCoder. If dataloader is transposing the ground
        truth then we must not transpose here. Always required for PREDICT.
    Returns:
      concatenanted cls_outputs and box_outputs.
    """
    assert set(cls_outputs.keys()) == set(box_outputs.keys())

    # This sort matters. The labels assume a certain order based on
    # constants.FEATURE_SIZES, and this sort matches that convention.
    keys = sorted(cls_outputs.keys())
    batch_size = int(cls_outputs[keys[0]].shape[0])

    flat_cls = []
    flat_box = []

    with tf.name_scope("concat_cls_outputs") as scope:
        for i, k in enumerate(keys):
            scale = constants.FEATURE_SIZES[i]

            split_shape = (constants.NUM_DEFAULTS[i], constants.NUM_CLASSES)
            assert cls_outputs[k].shape[3] == split_shape[0] * split_shape[1]

            intermediate_shape = (batch_size, scale, scale) + split_shape
            final_shape = (batch_size, scale ** 2 *
                           split_shape[0], split_shape[1])

            cls_outputs_k = cls_outputs[k]

            if transpose:
                cls_reshape = tf.reshape(
                    cls_outputs_k, intermediate_shape, name="reshape_k"+str(k))
                cls_transpose = tf.transpose(
                    cls_reshape, (0, 3, 1, 2, 4), name="transpose_k"+str(k))
                cls_reshape2 = tf.reshape(
                    cls_transpose, final_shape, name="reshape2_k"+str(k))
            else:
                cls_reshape2 = tf.reshape(
                    cls_outputs_k, final_shape, name="reshape_k"+str(k))

            flat_cls.append(cls_reshape2)

        concat_cls_outputs = tf.concat(flat_cls, axis=1, name=scope)

    with tf.name_scope("concat_box_outputs") as scope:
        for i, k in enumerate(keys):
            scale = constants.FEATURE_SIZES[i]

            split_shape = (constants.NUM_DEFAULTS[i], 4)
            assert box_outputs[k].shape[3] == split_shape[0] * split_shape[1]

            intermediate_shape = (batch_size, scale, scale) + split_shape
            final_shape = (batch_size, scale ** 2 *
                           split_shape[0], split_shape[1])

            box_outputs_k = box_outputs[k]

            if transpose:
                box_reshape = tf.reshape(
                    box_outputs_k, intermediate_shape, name="reshape_k"+str(k))
                box_transpose = tf.transpose(
                    box_reshape, (0, 3, 1, 2, 4), name="transpose_k"+str(k))
                box_reshape2 = tf.reshape(
                    box_transpose, final_shape, name="reshape2_k"+str(k))
            else:
                box_reshape2 = tf.reshape(
                    box_outputs_k, final_shape, name="reshape_k"+str(k))

            flat_box.append(box_reshape2)

        concat_box_outputs = tf.concat(flat_box, axis=1, name=scope)

    return concat_cls_outputs, concat_box_outputs


def _localization_loss(pred_locs, gt_locs, gt_labels, num_matched_boxes):
    """Computes the localization loss.

    Computes the localization loss using smooth l1 loss.
    Args:
      pred_locs: a dict from index to tensor of predicted locations. The shape
        of each tensor is [batch_size, num_anchors, 4].
      gt_locs: a list of tensors representing box regression targets in
        [batch_size, num_anchors, 4].
      gt_labels: a list of tensors that represents the classification groundtruth
        targets. The shape is [batch_size, num_anchors, 1].
      num_matched_boxes: the number of anchors that are matched to a groundtruth
        targets, used as the loss normalizater. The shape is [batch_size].
    Returns:
      box_loss: a float32 representing total box regression loss.
    """
    keys = sorted(pred_locs.keys())
    box_loss = 0
    for i, k in enumerate(keys):
        gt_label = gt_labels[i]
        gt_loc = gt_locs[i]
        pred_loc = tf.reshape(pred_locs[k], gt_loc.shape)
        mask = tf.greater(gt_label, 0)
        float_mask = tf.cast(mask, tf.float32)

        smooth_l1 = tf.reduce_sum(
            tf.losses.huber_loss(
                gt_loc, pred_loc, reduction=tf.losses.Reduction.NONE),
            axis=-1)
        smooth_l1 = tf.multiply(smooth_l1, float_mask)
        box_loss = box_loss + tf.reduce_sum(
            smooth_l1, axis=list(range(1, smooth_l1.shape.ndims)))

    # TODO(taylorrobie): Confirm that normalizing by the number of boxes matches
    # reference
    return tf.reduce_mean(box_loss / num_matched_boxes)


@tf.custom_gradient
def _softmax_cross_entropy_mme(logits, label):
    """Helper function to compute softmax cross entropy loss."""
    with tf.name_scope("softmax_cross_entropy"):
        batch_size = tf.shape(logits)[0]
        num_boxes = tf.shape(logits)[1]
        num_classes = tf.shape(logits)[2]
        reduce_sum_filter = tf.fill(
            [1, 1, num_classes, 1], 1.0, name="reduce_sum_filter")

        logits = tf.reshape(logits, [1, batch_size, num_boxes, num_classes])
        logits_t = tf.transpose(logits, perm=(0, 1, 3, 2), name="logits_t")
        reduce_max = tf.reduce_max(logits_t, 2, name="reduce_max")

        max_logits = tf.reshape(
            reduce_max, [1, batch_size, num_boxes, 1], name="max_logits")

        shifted_logits = tf.subtract(logits, max_logits, name="shifted_logits")
        exp_shifted_logits = tf.math.exp(
            shifted_logits, name="exp_shifted_logits")

        # MME was idle during classification_loss computation.
        # In this case, conv2d is equvalent to reduce_sum but reduce_sum is executed on TPC while conv2d on MME.
        sum_exp = tf.nn.conv2d(
            exp_shifted_logits, reduce_sum_filter, strides=1, padding="VALID", name="sum_exp")

        log_sum_exp = tf.math.log(sum_exp, name="log_sum_exp")
        one_hot_label = tf.one_hot(label, num_classes, name="one_hot_label")

        # MME was idle during classification_loss computation.
        # In this case, conv2d is equvalent to reduce_sum but reduce_sum is executed on TPC while conv2d on MME.
        shifted_logits2 = tf.nn.conv2d(
            shifted_logits * one_hot_label, reduce_sum_filter, strides=1, padding="VALID", name="shifted_logits2")

        loss = tf.subtract(log_sum_exp, shifted_logits2, name="loss/sub")
        loss = tf.reshape(loss, [batch_size, -1], name="loss")

    def grad(dy):
        with tf.name_scope("gradients/softmax_cross_entropy"):
            dy_reshaped = tf.reshape(
                dy, [1, batch_size, num_boxes, 1], name="dy/Reshape")
            div = tf.math.truediv(exp_shifted_logits, sum_exp, name="div")
            sub = tf.math.subtract(div, one_hot_label, name="sub")
            ret = tf.math.multiply(sub, dy_reshaped, name="mul")
            reshaped_ret = tf.reshape(
                ret, [batch_size, num_boxes, num_classes], name="Reshape")
        return reshaped_ret, dy

    return loss, grad


@tf.custom_gradient
def _softmax_cross_entropy(logits, label):
    """Helper function to compute softmax cross entropy loss."""
    shifted_logits = logits - tf.expand_dims(tf.reduce_max(logits, -1), -1)
    exp_shifted_logits = tf.math.exp(shifted_logits)
    sum_exp = tf.reduce_sum(exp_shifted_logits, -1)
    log_sum_exp = tf.math.log(sum_exp)
    one_hot_label = tf.one_hot(label, constants.NUM_CLASSES)
    shifted_logits = tf.reduce_sum(shifted_logits * one_hot_label, -1)
    loss = log_sum_exp - shifted_logits

    def grad(dy):
        return (exp_shifted_logits / tf.expand_dims(sum_exp, -1) -
                one_hot_label) * tf.expand_dims(dy, -1), dy

    return loss, grad


def _classification_loss(pred_labels, gt_labels, num_matched_boxes):
    """Computes the classification loss.

    Computes the classification loss with hard negative mining.
    Args:
      pred_labels: a dict from index to tensor of predicted class. The shape
      of the tensor is [batch_size, num_anchors, num_classes].
      gt_labels: a list of tensor that represents the classification groundtruth
      targets. The shape is [batch_size, num_anchors, 1].
      num_matched_boxes: the number of anchors that are matched to a groundtruth
      targets. This is used as the loss normalizater.
    Returns:
      box_loss: a float32 representing total box regression loss.
    """
    with tf.name_scope("class_loss_scope"):
        keys = sorted(pred_labels.keys())
        batch_size = gt_labels[0].shape[0]

        assert(len(keys) == 1)
        assert(keys[0] == 'flatten')
        gt_label = gt_labels[0]
        pred_label = pred_labels['flatten']

        cross_entropy = tf.reshape(
            _softmax_cross_entropy_mme(pred_label, gt_label), [batch_size, -1], name="cross_entropy")

        mask = tf.greater(gt_label, 0, name="mask")
        float_mask = tf.cast(mask, tf.float32, name="float_mask")

        # Hard example mining
        neg_masked_cross_entropy = cross_entropy * (1 - float_mask)

        num_neg_boxes = tf.expand_dims(tf.minimum(
            tf.cast(num_matched_boxes, tf.int32) * constants.NEGS_PER_POSITIVE,
            constants.NUM_SSD_BOXES), -1)
        _, neg_sorted_cross_indices = tf.nn.top_k(
            neg_masked_cross_entropy, tf.shape(neg_masked_cross_entropy)[1])  # descending order

        _, neg_sorted_cross_rank = tf.nn.top_k(
            -1*neg_sorted_cross_indices, tf.shape(neg_sorted_cross_indices)[1])  # ascending order
        topk_neg_mask = tf.cast(tf.math.less(
            neg_sorted_cross_rank, num_neg_boxes), tf.float32)

        add = tf.add(float_mask, topk_neg_mask, name="add")

        class_loss = tf.reduce_sum(
            tf.multiply(cross_entropy, add, name="mul"), axis=1, name="reduce_sum")

        normalized_class_loss = tf.truediv(
            class_loss, num_matched_boxes, name="normalized_class_loss")

    return tf.reduce_mean(normalized_class_loss, name="class_loss_1")


def detection_loss(cls_outputs, box_outputs, labels):
    """Computes total detection loss.

    Computes total detection loss including box and class loss from all levels.
    Args:
      cls_outputs: an OrderDict with keys representing levels and values
        representing logits in [batch_size, height, width, num_anchors].
      box_outputs: an OrderDict with keys representing levels and values
        representing box regression targets in
        [batch_size, height, width, num_anchors * 4].
      labels: the dictionary that returned from dataloader that includes
        groundturth targets.
    Returns:
      total_loss: a float32 representing total loss reducing from class and box
        losses from all levels.
      cls_loss: a float32 representing total class loss.
      box_loss: a float32 representing total box regression loss.
    """
    if isinstance(labels[constants.BOXES], dict):
        gt_boxes = list(labels[constants.BOXES].values())
        gt_classes = list(labels[constants.CLASSES].values())
    else:
        gt_boxes = [labels[constants.BOXES]]
        gt_classes = [labels[constants.CLASSES]]
        cls_outputs, box_outputs = concat_outputs(
            cls_outputs, box_outputs, False)
        cls_outputs = {'flatten': cls_outputs}
        box_outputs = {'flatten': box_outputs}

    box_loss = _localization_loss(box_outputs, gt_boxes, gt_classes,
                                  labels[constants.NUM_MATCHED_BOXES])
    class_loss = _classification_loss(cls_outputs, gt_classes,
                                      labels[constants.NUM_MATCHED_BOXES])

    return class_loss + box_loss, class_loss, box_loss


def update_learning_rate_schedule_parameters(params):
    """Updates params that are related to the learning rate schedule.

    Args:
      params: a parameter dictionary that includes learning_rate, lr_warmup_epoch,
        first_lr_drop_epoch, and second_lr_drop_epoch.
    """
    # Learning rate is proportional to the batch size
    steps_per_epoch = params['num_examples_per_epoch'] / \
        params['global_batch_size']
    params['lr_warmup_step'] = int(params['lr_warmup_epoch'] * steps_per_epoch)
    params['first_lr_drop_step'] = int(
        params['first_lr_drop_epoch'] * steps_per_epoch)
    params['second_lr_drop_step'] = int(
        params['second_lr_drop_epoch'] * steps_per_epoch)


def learning_rate_schedule(params, global_step):
    """Handles learning rate scaling, linear warmup, and learning rate decay.

    Args:
      params: A dictionary that defines hyperparameters of model.
      global_step: A tensor representing current global step.

    Returns:
      A tensor representing current learning rate.
    """
    base_learning_rate = params['base_learning_rate']
    lr_warmup_step = params['lr_warmup_step']
    first_lr_drop_step = params['first_lr_drop_step']
    second_lr_drop_step = params['second_lr_drop_step']
    scaling_factor = params['global_batch_size'] / constants.DEFAULT_BATCH_SIZE
    adjusted_learning_rate = base_learning_rate * scaling_factor

    with tf.colocate_with(global_step):
        learning_rate = (tf.cast(global_step, dtype=tf.float32) /
                         lr_warmup_step) * adjusted_learning_rate
        learning_rate = tf.where(global_step < lr_warmup_step, learning_rate,
                                 adjusted_learning_rate * 1.0, name="learning_rate_schedule_1")
        learning_rate = tf.where(global_step < first_lr_drop_step, learning_rate,
                                 adjusted_learning_rate * 0.1, name="learning_rate_schedule_2")
        learning_rate = tf.where(global_step < second_lr_drop_step, learning_rate,
                                 adjusted_learning_rate * 0.01, name="learning_rate_1")

    return learning_rate


def _model_fn(features, labels, mode, params, model):
    """Model defination for the SSD model based on ResNet-50.

    Args:
      features: the input image tensor with shape [batch_size, height, width, 3].
        The height and width are fixed and equal.
      labels: the input labels in a dictionary. The labels include class targets
        and box targets which are dense label maps. The labels are generated from
        get_input_fn function in data/dataloader.py
      mode: the mode of TPUEstimator including TRAIN, EVAL, and PREDICT.
      params: the dictionary defines hyperparameters of model. The default
        settings are in default_hparams function in this file.
      model: the SSD model outputs class logits and box regression outputs.

    Returns:
      spec: the EstimatorSpec or TPUEstimatorSpec to run training, evaluation,
        or prediction.
    """
    if mode == tf.estimator.ModeKeys.PREDICT:
        labels = features
        features = labels.pop('image')

    features -= tf.constant(
        constants.NORMALIZATION_MEAN, shape=[1, 1, 3], dtype=features.dtype)
    COEF_STD = 1.0 / tf.constant(
        constants.NORMALIZATION_STD, shape=[1, 1, 3], dtype=features.dtype)
    features *= COEF_STD

    def _model_outputs():
        return model(
            features, params, is_training_bn=(mode == tf.estimator.ModeKeys.TRAIN))

    if params['dtype'] == 'bf16':
        with tf.compat.v1.tpu.bfloat16_scope():
            cls_outputs, box_outputs = _model_outputs()
            levels = cls_outputs.keys()
            for level in levels:
                cls_outputs[level] = tf.cast(cls_outputs[level], tf.float32)
                box_outputs[level] = tf.cast(box_outputs[level], tf.float32)
    else:
        cls_outputs, box_outputs = _model_outputs()
        levels = cls_outputs.keys()

    # First check if it is in PREDICT mode.
    if mode == tf.estimator.ModeKeys.PREDICT:
        flattened_cls, flattened_box = concat_outputs(
            cls_outputs, box_outputs, True)
        ssd_box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
            scale_factors=constants.BOX_CODER_SCALES)

        anchors = box_list.BoxList(
            tf.convert_to_tensor(dataloader.DefaultBoxes()('ltrb')))

        decoded_boxes = box_coder.batch_decode(
            encoded_boxes=flattened_box, box_coder=ssd_box_coder, anchors=anchors)

        pred_scores = tf.nn.softmax(flattened_cls, axis=2)

        pred_scores, indices = select_top_k_scores(pred_scores,
                                                   constants.MAX_NUM_EVAL_BOXES)
        predictions = dict(
            labels,
            indices=indices,
            pred_scores=pred_scores,
            pred_box=decoded_boxes,
        )

        if params['visualize_dataloader']:
            # this is for inference visualization.
            predictions['image'] = features

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Load pretrained model from checkpoint.
    if params['resnet_checkpoint'] and mode == tf.estimator.ModeKeys.TRAIN:

        def scaffold_fn():
            """Loads pretrained model through scaffold function."""
            tf.train.init_from_checkpoint(params['resnet_checkpoint'], {
                '/': 'resnet%s/' % constants.RESNET_DEPTH,
            })
            return tf.train.Scaffold()
    else:
        scaffold_fn = None

    # Set up training loss and learning rate.
    update_learning_rate_schedule_parameters(params)
    global_step = tf.train.get_or_create_global_step()
    learning_rate = learning_rate_schedule(params, global_step)
    # cls_loss and box_loss are for logging. only total_loss is optimized.
    loss, cls_loss, box_loss = detection_loss(
        cls_outputs, box_outputs, labels)

    total_loss = loss + params['weight_decay'] * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(
            learning_rate, momentum=constants.MOMENTUM)

        if params['distributed_optimizer']:
            optimizer = params['distributed_optimizer'](optimizer)

        # Batch norm requires update_ops to be added as a train_op dependency.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(optimizer.minimize(total_loss, global_step),
                            update_ops)

        save_summary_steps = params['save_summary_steps']
        if save_summary_steps is not None:
            tf.summary.scalar("learning_rate", learning_rate)
            tf.summary.scalar("class_loss", cls_loss)
            tf.summary.scalar("box_loss", box_loss)

        return model_fn_lib.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            training_hooks=None,
            scaffold=scaffold_fn())

    if mode == tf.estimator.ModeKeys.EVAL:
        raise NotImplementedError


def ssd_model_fn(features, labels, mode, params):
    """SSD model."""
    return _model_fn(features, labels, mode, params, model=architecture.ssd)

import numpy as np
import tensorflow as tf
from tensorflow import identity
from stereo.interfaces.implements import implements_format
from stereo.models.loss.top_view.top_view_loss_utils import get_combined_label_image, is_positive_min, \
    mpl_get_boxes_image


def get_default_boxes(out_resolution, inp_resolution, origin, inp_focal):
    out_length, out_width = out_resolution
    default_boxes = np.zeros((out_length * out_width, 4), dtype=np.float32)  # 4 box params
    inp_length, inp_width = np.array(inp_resolution) / inp_focal
    box_length = inp_length / out_length
    box_width = inp_width / out_width
    default_boxes[:, 0] = box_length  # 1. length in meters
    default_boxes[:, 1] = box_width  # 2. width in meters
    j, i = np.meshgrid(np.arange(out_width), np.arange(out_length))
    i = 1. * inp_resolution[0] / out_resolution[0] * i - origin[0]
    j = 1. * inp_resolution[1] / out_resolution[1] * j - origin[1]
    default_boxes[:, 2] = i.flatten() / inp_focal  # 3. y location in meters
    default_boxes[:, 3] = j.flatten() / inp_focal  # 4. x location in meters
    return default_boxes


def get_matching_labels(labels, default_boxes, max_distance):
    labels_centers = labels[..., 2:4]  # xy locations in labels (?, 50, 2)
    broadcast_labels_centers = tf.expand_dims(labels_centers, 2)  # shape: (?, 50, 1, 2)
    default_boxes_centers = default_boxes[:, 2:4]  # xy locations in default boxes (40X16 , 2)
    broadcast_boxes_centers = tf.expand_dims(default_boxes_centers, 0)  # shape: (1, 40X16, 2)
    diff_centers = np.abs(broadcast_boxes_centers - broadcast_labels_centers)  # shape: (?, 50, 40X16, 2)
    diff_xs = diff_centers[..., 0]  # shape: (?, 50, 40X16)
    diff_ys = diff_centers[..., 1]  # shape: (?, 50, 40X16)
    matching = tf.logical_and(diff_xs < max_distance[0], diff_ys < max_distance[1])  # shape: (?, 50, 40X16)
    closest_matching = is_positive_min(diff_xs**2 + diff_ys**2, axis=1)  # shape: (?, 50, 40X16)
    matching = tf.logical_and(matching, closest_matching)  # shape: (?, 50, 40X16)
    is_real_label = labels[..., 5]  # shape: (?, 50)
    is_real_label = tf.expand_dims(is_real_label, -1)  # shape: (?, 50, 1)
    _ = identity(tf.reduce_sum(input_tensor=is_real_label), "real_num_of_labels")
    is_real_label = tf.cast(is_real_label, tf.bool)
    matching = tf.logical_and(matching, is_real_label)  # remove fake labels <==> fake match. shape: (?, 50, 40X16)
    return matching


def get_boxes_to_labels_loss(pred_boxes, labels):
    labels = labels[:, :, :5]  # remove is_real flag, shape: (?, 50, 5)
    labels = angle_box_to_sin_cos_box(labels)  # shape: (?, 50, 6)
    broadcast_labels = tf.expand_dims(labels, 2)  # shape: (?, 50, 1, 6)
    broadcast_boxes = tf.expand_dims(pred_boxes, 1)  # shape: (?, 1, 40X16, 6)
    diff_boxes = np.abs(broadcast_labels - broadcast_boxes)  # shape: (?, 50, 40X16, 6)
    return diff_boxes


def angle_box_to_sin_cos_box(five_box):
    return tf.concat([five_box[..., :4],
                      tf.expand_dims(tf.sin(five_box[..., 4]), -1),
                      tf.expand_dims(tf.cos(five_box[..., 4]), -1)], axis=-1)


def get_pred_boxes(default_boxes, boxes_params_preds):
    orientation_params = boxes_params_preds[..., 4:6]
    normalized_orientation_vector = tf.math.l2_normalize(orientation_params, axis=-1)
    pred_boxes = tf.concat([default_boxes + boxes_params_preds[..., :4], normalized_orientation_vector], axis=-1)
    return pred_boxes


loss_format = ("loss", "yolo_sim_top_view")
@implements_format(*loss_format)
def loss(out, label, top_view_cam, vehicles_boxes, inp_resolution=(700, 300), origin=(300, 150), input_focal=10.0,
         max_distance=None, reg_constant=1.0, categories_from_top_view=[], conf_thresh=[0.0]):  # labels shape: (?, 50, 6)

    out_resolution = out.shape.as_list()[1:3]  # [40, 16]
    out = tf.reshape(out, [-1, np.prod(out_resolution), 7])
    boxes_params_preds, confidence_preds = out[..., :6], out[..., 6]  # shapes: (?, 40X16, 6), (?, 40X16)
    default_boxes = get_default_boxes(out_resolution, inp_resolution, origin, input_focal)  # shape: (40X16, 4)
    if max_distance is None:
        max_distance = 0.5 * (inp_resolution / (input_focal * np.array(out_resolution)))
    matching_labels = get_matching_labels(vehicles_boxes, default_boxes, max_distance=max_distance)  # shape: (?, 50, 40X16)
    matching_labels = tf.cast(matching_labels, tf.float32)
    num_of_matching_labels = tf.reduce_sum(input_tensor=matching_labels)
    are_matching_boxes = tf.cast(tf.reduce_sum(input_tensor=matching_labels, axis=1) > 0, tf.float32)  # shape: (?, 40X16)

    # Locations loss
    pred_boxes = get_pred_boxes(default_boxes, boxes_params_preds)  # shape: (?, 40X16, 6)
    all_boxes_loss = get_boxes_to_labels_loss(pred_boxes, vehicles_boxes)  # shape: (?, 50, 40X16, 6)
    matching_boxes_loss = tf.expand_dims(matching_labels, -1) * all_boxes_loss  # shape: (? ,50, 40X16, 6)
    locations_loss = tf.reduce_sum(input_tensor=matching_boxes_loss, axis=[1, 2]) / (tf.reduce_sum(input_tensor=matching_labels) + 1e-6)  # shape: (?, 6)
    _ = identity(tf.reduce_mean(input_tensor=locations_loss[:, 0]), "scalar_loss_length")
    _ = identity(tf.reduce_mean(input_tensor=locations_loss[:, 1]), "scalar_loss_width")
    _ = identity(tf.reduce_mean(input_tensor=locations_loss[:, 2]), "scalar_loss_y")
    _ = identity(tf.reduce_mean(input_tensor=locations_loss[:, 3]), "scalar_loss_x")
    _ = identity(tf.reduce_mean(input_tensor=locations_loss[:, 4]), "scalar_loss_sin")
    _ = identity(tf.reduce_mean(input_tensor=locations_loss[:, 5]), "scalar_loss_cos")
    locations_loss = tf.reduce_mean(input_tensor=locations_loss)

    # Confidence loss
    confidence_loss = tf.keras.losses.binary_crossentropy(are_matching_boxes, confidence_preds, from_logits=True)
    confidence_loss = tf.reduce_mean(input_tensor=confidence_loss)

    combined_loss = locations_loss + confidence_loss

    reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
    loss_reg = reg_constant * tf.reduce_sum(input_tensor=reg_losses)

    total_loss = combined_loss + loss_reg

    # Visualization

    top_view_cam = tf.cast(top_view_cam, dtype=tf.int32)
    label = tf.cast(label, dtype=tf.int32)
    label_image = get_combined_label_image(top_view_cam, label, categories_from_top_view)

    sigmoid_confidence = tf.nn.sigmoid(confidence_preds)  # shape: (?, 40X16)
    boxes_image = mpl_get_boxes_image(pred_boxes[0], sigmoid_confidence[0], origin, input_focal,
                                      tf.squeeze(label_image, -1)[0], confidence_thresh=0.1)
    _ = identity(boxes_image, 'boxes_image')
    matching_confidence_preds = sigmoid_confidence * are_matching_boxes
    # matching_boxes_image = mpl_get_boxes_image(pred_boxes[0], matching_confidence_preds[0], origin, input_focal,
    #                                            tf.squeeze(label_image, -1)[0], confidence_thresh=0.0)
    # _ = identity(matching_boxes_image, 'matching_boxes_image')

    _ = identity(locations_loss, 'scalar_loss_locations')
    _ = identity(confidence_loss, 'scalar_loss_confidence')
    _ = identity(combined_loss, 'scalar_loss_combined')
    _ = identity(num_of_matching_labels, 'num_of_matching_labels')
    _ = identity(loss_reg, 'loss_reg')
    _ = identity(total_loss, 'loss')

    return total_loss

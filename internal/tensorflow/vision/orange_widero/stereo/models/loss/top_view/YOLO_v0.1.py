import numpy as np
import tensorflow as tf
from tensorflow import identity
from stereo.interfaces.implements import implements_format
from stereo.models.arch_utils import color_categories
from stereo.models.loss.top_view.top_view_loss_utils import get_combined_label_image
import tfmpl
from cv2 import fillPoly


def get_box_pixels(box, origin, focal):
    length, width, x, y, orientation = box
    orientation = np.pi - orientation
    bb = 0.5 * np.array([
        (length, width),
        (-length, width),
        (-length, -width),
        (length, -width)
    ])
    rotation_matrix = np.array([
        [np.cos(orientation), -np.sin(orientation)],
        [np.sin(orientation), np.cos(orientation)]
    ])
    center = np.reshape([x, y], (2,))
    oriented_bb = np.dot(bb, rotation_matrix) + center
    pixels_oriented_bb = np.flip(focal * oriented_bb + origin).astype(np.int32)
    return pixels_oriented_bb


@tfmpl.figure_tensor
def mpl_get_boxes_image(boxes, confidence, inp_resolution, origin, focal, confidence_thresh=0.0):
    boxes = boxes[confidence > confidence_thresh]
    confidence = confidence[confidence > confidence_thresh]
    image = np.zeros(inp_resolution)
    for i, box in enumerate(boxes):
        box_pixels = get_box_pixels(box, origin, focal)
        fillPoly(image, pts=[box_pixels], color=float(confidence[i]))
    fig = tfmpl.create_figure(figsize=(5, 12))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.imshow(image, cmap='gray')
    fig.tight_layout()
    return fig


def get_positive_min(t, axis):
    """
    :param t: tensor to run on. all values must be non negative
    :param axis: run minimum along axis
    :return: Returns the positive minimum along axis. If all values along axis equals 0 return 0
    """
    large_num = 50000.0
    to_add = tf.cast(tf.equal(t, 0), tf.float32) * large_num
    fake_t = t + to_add
    fake_min = tf.reduce_min(input_tensor=fake_t, axis=axis)
    to_sub = tf.cast(tf.equal(fake_min, large_num), tf.float32) * large_num
    true_min = fake_min - to_sub
    return true_min


def flatten(t):
    if type(t) == np.ndarray:
        original_shape = list(t.shape)
        prod_shape = np.prod(original_shape[:-1])
        return np.reshape(t, [prod_shape, original_shape[-1]])
    else:
        original_shape = list(t.shape.as_list())
        prod_shape = np.prod(original_shape[1:-1])
        return tf.reshape(t, [-1, prod_shape, original_shape[-1]])


def get_default_boxes(out_resolution, inp_resolution, origin, inp_focal):
    out_length, out_width = out_resolution
    default_boxes = np.zeros((out_length, out_width, 5), dtype=np.float32)  # 5 box params
    inp_length, inp_width = np.array(inp_resolution) / inp_focal
    box_length = inp_length / out_length
    box_width = inp_width / out_width
    default_boxes[:, :, 0] = box_length  # 1. length in meters
    default_boxes[:, :, 1] = box_width  # 2. width in meters
    j, i = np.meshgrid(np.arange(out_width), np.arange(out_length))
    i = 1. * inp_resolution[0] / out_resolution[0] * i - origin[0]
    j = 1. * inp_resolution[1] / out_resolution[1] * j - origin[1]
    default_boxes_cntrs = np.stack([i, j], -1) / inp_focal
    default_boxes[:, :, 2:4] = default_boxes_cntrs  # 3-4. yx locations in meters
    # 5. orientation = always 0
    return default_boxes


def get_matching_labels(labels, default_boxes, max_distance):
    labels_centers = labels[:, :, 2:4]  # xy locations in labels (?, 50, 2)
    broadcast_labels_centers = tf.expand_dims(labels_centers, 2)  # shape: (?, 50, 1, 2)
    default_boxes_centers = default_boxes[:, :, 2:4]  # xy locations in default boxes (40, 16 , 2)
    flatten_boxes_centers = flatten(default_boxes_centers)  # shape: (40X16, 2)
    broadcast_boxes_centers = tf.expand_dims(flatten_boxes_centers, 0)  # shape: (1, 40X16, 2)
    diff_centers = np.abs(broadcast_boxes_centers - broadcast_labels_centers)  # shape: (?, 50, 16X40, 2)
    diff_xs = diff_centers[:, :, :, 0]  # shape: (?, 50, 16X40)
    diff_ys = diff_centers[:, :, :, 1]  # shape: (?, 50, 16X40)
    num_of_labels = labels.shape[1]
    boxes_shape = list(default_boxes.shape[:2])
    flatten_matching = tf.logical_and(diff_xs < max_distance[0], diff_ys < max_distance[1])  # shape: (?, 50, 40X16)
    matching = tf.reshape(flatten_matching, [-1, num_of_labels] + boxes_shape)  # shape: (?, 50, 40, 16)
    is_real_label = labels[:, :, 5]  # shape: (?, 50)
    is_real_label = tf.reshape(is_real_label, [-1, num_of_labels] + [1, 1])  # shape: (?, 50, 1, 1)
    _ = identity(tf.reduce_sum(input_tensor=is_real_label), "real_num_of_labels")
    is_real_label = tf.cast(is_real_label, tf.bool)
    matching = tf.logical_and(matching, is_real_label)  # remove fake labels <==> fake match
    return matching


def get_boxes_to_labels_loss(pred_boxes, labels):
    labels = labels[:, :, :5]  # remove is_real flag, shape: (?, 50, 5)
    broadcast_labels_centers = tf.expand_dims(labels, 2)  # shape: (?, 50, 1, 5)
    _, out_length, out_width, _ = pred_boxes.shape.as_list()
    broadcast_boxes_centers = tf.reshape(pred_boxes, (-1, 1, out_length * out_width, 5))  # shape: (?, 1, 40X16, 5)
    diff_boxes = np.abs(broadcast_labels_centers - broadcast_boxes_centers)  # shape: (?, 50, 40X16, 5)
    angles_diff = diff_boxes[:, :, :, 4]  # shape: (?, 50, 40X16)
    angles_diff = tf.abs(tf.atan2(tf.sin(angles_diff), tf.cos(angles_diff)))  # shape: (?, 50, 40X16)
    angles_diff = tf.expand_dims(angles_diff, -1)  # shape: (?, 50, 40X16, 1)
    distances_diff = diff_boxes[:, :, :, :4]  # shape: (?, 50, 40X16, 4)
    all_diff = tf.concat([distances_diff, angles_diff], -1)  # shape: (?, 50, 40X16, 5)
    num_of_labels = labels.shape.as_list()[1]
    reshaped_all_diff = tf.reshape(all_diff, (-1, num_of_labels, out_length, out_width, 5))  # shape: (?, 50, 40, 16, 5)
    return reshaped_all_diff


loss_format = ("loss", "yolo_sim_top_view")
@implements_format(*loss_format)
def loss(out, label, top_view_cam, vehicles_boxes, inp_resolution=(700, 300), origin=(300, 150), input_focal=10.0,
         max_distance=(1.0, 1.0), reg_constant=1.0, categories_from_top_view=[], conf_thresh=[0.0]):  # labels shape: (?, 50, 6)

    boxes_params_preds, confidence_preds = out[:, :, :, :5], out[:, :, :, 5]  # shapes - (?, 40, 16, 5) , (?, 40, 16)
    out_resolution = boxes_params_preds.shape.as_list()[1:3]  # [40, 16]
    default_boxes = get_default_boxes(out_resolution, inp_resolution, origin, input_focal)  # shape: (40, 16, 5)
    matching_labels = get_matching_labels(vehicles_boxes, default_boxes, max_distance=max_distance)  # shape: (?, 50, 40, 16)
    not_matching_labels = tf.logical_not(matching_labels)
    matching_labels = tf.cast(matching_labels, tf.float32)
    num_of_matching_labels = tf.reduce_sum(input_tensor=matching_labels)
    not_matching_labels = tf.cast(not_matching_labels, tf.float32)

    # Locations loss
    pred_boxes = default_boxes + boxes_params_preds  # shape: (?, 40, 16, 5)
    all_boxes_loss = get_boxes_to_labels_loss(pred_boxes, vehicles_boxes)  # shape: (?, 50, 40, 16, 5)
    matching_boxes_loss = tf.expand_dims(matching_labels, -1) * all_boxes_loss  # shape: (? ,50, 40, 16, 5)
    # min_matches_loss = get_positive_min(matching_boxes_loss, axis=1)  # shape: (?, 40, 16)
    # expanded_min_matches_loss = tf.expand_dims(min_matches_loss, 1)  # shape: (?, 1, 40, 16)
    # is_min_box_loss = tf.cast(tf.equal(matching_boxes_loss, expanded_min_matches_loss), tf.float32)  # shape: (? ,50, 40, 16)
    # best_labels_matching = is_min_box_loss * matching_labels  # shape: (? ,50, 40, 16)
    # locations_loss = tf.reduce_sum(matching_boxes_loss, axis=[1, 2, 3]) / (tf.reduce_sum(best_labels_matching) + 0.0001)
    locations_loss = tf.reduce_sum(input_tensor=matching_boxes_loss, axis=[1, 2, 3]) / (tf.reduce_sum(input_tensor=matching_labels) + 1e-6)  # shape: (?, 5)
    _ = identity(tf.reduce_mean(input_tensor=locations_loss[:, 0]), "scalar_loss_length")
    _ = identity(tf.reduce_mean(input_tensor=locations_loss[:, 1]), "scalar_loss_width")
    _ = identity(tf.reduce_mean(input_tensor=locations_loss[:, 2]), "scalar_loss_y")
    _ = identity(tf.reduce_mean(input_tensor=locations_loss[:, 3]), "scalar_loss_x")
    _ = identity(tf.reduce_mean(input_tensor=locations_loss[:, 4]), "scalar_loss_orientation")
    locations_loss = tf.reduce_mean(input_tensor=locations_loss)

    # Confidence loss
    num_of_labels = vehicles_boxes.shape.as_list()[1]
    tiled_confidence_preds = tf.tile(tf.expand_dims(confidence_preds, 1), [1, num_of_labels, 1, 1])  # shape: (?, 50, 40, 16)
    confidence_loss = tf.reduce_mean(input_tensor=-tf.math.log(tiled_confidence_preds + 1e-6) * matching_labels) + \
                      tf.reduce_mean(input_tensor=-tf.math.log(1 - tiled_confidence_preds + 1e-6) * not_matching_labels)

    combined_loss = locations_loss + confidence_loss

    reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
    loss_reg = reg_constant * tf.reduce_sum(input_tensor=reg_losses)

    total_loss = combined_loss + loss_reg

    reshaped_confidence_preds = tf.reshape(confidence_preds, [-1, np.prod(out_resolution)])
    for thresh in conf_thresh:
        boxes_image = mpl_get_boxes_image(flatten(pred_boxes)[0], reshaped_confidence_preds[0], inp_resolution, origin,
                                          input_focal, confidence_thresh=thresh)
        _ = identity(boxes_image, 'boxes_image_%s' % thresh)

    top_view_cam = tf.cast(top_view_cam, dtype=tf.int32)
    label = tf.cast(label, dtype=tf.int32)
    label_image = get_combined_label_image(top_view_cam, label, categories_from_top_view)

    are_matching_boxes = tf.cast(tf.reduce_sum(input_tensor=matching_labels, axis=1) > 0, tf.float32)  # shape: (?, 40, 16)
    matching_confidence_preds = confidence_preds * are_matching_boxes
    reshaped_matching_confidence_preds = tf.reshape(matching_confidence_preds, [-1, np.prod(out_resolution)])
    matching_boxes_image = mpl_get_boxes_image(flatten(pred_boxes)[0], reshaped_matching_confidence_preds[0],
                                               inp_resolution, origin, input_focal, confidence_thresh=0.0)
    _ = identity(matching_boxes_image, 'matching_boxes_image')
    _ = identity(color_categories(tf.squeeze(label_image, [3])), "rgb_label")

    _ = identity(locations_loss, 'scalar_loss_locations')
    _ = identity(confidence_loss, 'scalar_loss_confidence')
    _ = identity(combined_loss, 'scalar_loss_combined')
    _ = identity(num_of_matching_labels, 'num_of_matching_labels')
    _ = identity(loss_reg, 'loss_reg')
    _ = identity(total_loss, 'loss')

    return total_loss

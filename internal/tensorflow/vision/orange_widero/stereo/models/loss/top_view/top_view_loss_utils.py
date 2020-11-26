import tensorflow as tf
import numpy as np
import tfmpl
import matplotlib.patches as patches
from matplotlib import cm


# Categories:
UNKNOWN = 0
AGENT = 1
VEHICLES = 2
PEDS = 3
ROAD = 4
GROUND = 5
OBJECT = 6

categories_names = {
    VEHICLES: "vehicles",
    PEDS: "peds",
    ROAD: "road",
    GROUND: "ground",
    OBJECT: "object"
}

CARLA_LABEL_TO_VIDAR_LABEL = {
        0: UNKNOWN,    # None
        1: OBJECT,     # Buildings
        2: OBJECT,     # Fences
        3: OBJECT,     # Other
        4: PEDS,       # Pedestrians
        5: UNKNOWN,    # Poles
        6: ROAD,       # RoadLines
        7: ROAD,       # Roads
        8: GROUND,     # Sidewalks
        9: UNKNOWN,    # Vegetation
        10: VEHICLES,  # Vehicles
        11: OBJECT,    # Walls
        12: UNKNOWN,   # TrafficSigns
    }


def map_to_categories(categories_img_inp, categories_map):
    return tf.gather(params=categories_map, indices=tf.cast(categories_img_inp, dtype=tf.int32))


def mark_category_in_label(label, top_view_cam, category):
    label_unknown = tf.equal(label, UNKNOWN)
    label_known = tf.not_equal(label, UNKNOWN)
    category_pixels = tf.cast(tf.logical_and(tf.equal(top_view_cam, category),label_unknown), label.dtype)
    not_category_pixels = tf.cast(tf.logical_or(tf.not_equal(label, category), label_known), top_view_cam.dtype)
    label = not_category_pixels * label + category_pixels * category
    return label


def get_combined_label_image(top_view_cam, label, categories_from_top_view):
    top_view_cam = map_to_categories(top_view_cam, list(CARLA_LABEL_TO_VIDAR_LABEL.values()))
    for category in categories_from_top_view:
        label = mark_category_in_label(label, top_view_cam, category)
    return label


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


def is_positive_min(t, axis):
    return tf.equal(t, tf.expand_dims(get_positive_min(t, axis), axis))


def get_box_pixels(box, origin, focal):
    length, width, x, y, sin, cos = box
    orientation = np.arctan2(sin, cos)
    bb = 0.5 * np.array([
        (-length, -width),
        (length, -width),
        (length, width),
        (-length, width)
    ])
    rotation_matrix = np.array([
        [np.cos(orientation), np.sin(orientation)],
        [-np.sin(orientation), np.cos(orientation)]
    ])
    center = np.reshape([x, y], (2,))
    oriented_bb = np.dot(bb, rotation_matrix) + center
    pixels_oriented_bb = np.flip(focal * oriented_bb + origin)
    return pixels_oriented_bb


@tfmpl.figure_tensor
def mpl_get_boxes_image(boxes, confidence, origin, focal, label_image, confidence_thresh=0.0,
                        label_cmap='rainbow', boxes_cmap='hot'):
    boxes = boxes[confidence > confidence_thresh]
    confidence = confidence[confidence > confidence_thresh]
    boxes = boxes[confidence.argsort()]
    confidence = confidence[confidence.argsort()]
    fig = tfmpl.create_figure(figsize=(5, 12))
    ax = fig.add_subplot(111)
    fig.tight_layout()
    ax.axis('off')
    ax.imshow(label_image, cmap=label_cmap)
    boxes_colormap = cm.get_cmap(boxes_cmap)
    for i, box in enumerate(boxes):
        box_pixels = get_box_pixels(box, origin, focal)
        box_corner = box_pixels[0]
        box_angle = -np.rad2deg(np.arctan2(box[4], box[5]))
        rect = patches.Rectangle(box_corner, -box[1] * focal, box[0] * focal, angle=box_angle,
                                 linewidth=2, edgecolor=boxes_colormap(confidence[i]), facecolor='none')
        middle_top = (0.5 * (box_pixels[1] + box_pixels[2]))
        middle_bottom = (0.5 * (box_pixels[0] + box_pixels[3]))
        arrow_direction = 0.5 * (middle_top - middle_bottom)
        arrow = patches.Arrow(middle_top[0], middle_top[1], arrow_direction[0], arrow_direction[1],
                              linewidth=2, edgecolor=boxes_colormap(confidence[i]))
        ax.add_patch(rect)
        ax.add_patch(arrow)
    return fig
import tensorflow as tf
from tensorflow import identity
from stereo.interfaces.implements import implements_format
from stereo.models.arch_utils import color_categories
from stereo.models.loss.top_view.top_view_loss_utils import *


loss_format = ("loss", "sim_top_view")
@implements_format(*loss_format)
def loss(out, label, top_view_cam, num_of_categories=3, categories_weights=None, categories_map=[], reg_constant=1.0,
         categories_from_top_view=[]):
    top_view_cam = tf.cast(top_view_cam, dtype=tf.int32)
    label = tf.cast(label, dtype=tf.int32)
    label = get_combined_label_image(top_view_cam, label, categories_from_top_view)
    if categories_weights is None:
        categories_weights = [1.0] * num_of_categories

    # We don't want to punish on unknown or agent
    mask_unknown = tf.logical_and(tf.not_equal(label, UNKNOWN), tf.not_equal(label, AGENT))
    mask_unknown = tf.cast(mask_unknown, tf.float32)
    mask_unknown = tf.squeeze(mask_unknown, [3])

    mapped_label = map_to_categories(label, categories_map)
    reversed_categories_map = [i for i, v in sorted(enumerate(categories_map), key=lambda iv: iv[1]) if v != -1]
    categories_loss_list = []
    for category in range(num_of_categories):
        category_label = tf.cast(tf.equal(mapped_label, category), tf.float32)
        category_out = tf.slice(out, [0, 0, 0, category], [-1, -1, -1, 1])
        _ = identity(category_out, name="%s_out" % categories_names[reversed_categories_map[category]])
        category_loss = tf.keras.losses.binary_crossentropy(category_label, category_out)
        category_loss *= mask_unknown
        category_loss *= categories_weights[category]
        _ = identity(category_loss, name="image_%s_loss" % categories_names[reversed_categories_map[category]])
        _ = identity(tf.reduce_mean(input_tensor=category_loss), name="scalar_loss_%s" %
                                                         categories_names[reversed_categories_map[category]])
        categories_loss_list.append(category_loss)
    loss_categories = tf.reduce_mean(input_tensor=categories_loss_list)

    # regularization
    reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
    loss_reg = reg_constant * tf.reduce_sum(input_tensor=reg_losses)

    total_loss = loss_categories + loss_reg
    _ = identity(loss_categories, name="scalar_loss_categories")
    _ = identity(total_loss, name="loss")
    _ = identity(color_categories(tf.squeeze(label, [3])), "rgb_label")
    _ = identity(tf.reduce_sum(input_tensor=categories_loss_list, axis=0), name='image_all_loss')
    _ = identity(loss_reg, name="loss_reg")
    return total_loss

import tensorflow as tf
from tensorflow import identity
from stereo.interfaces.implements import implements_format
from stereo.models.arch_utils import color_categories


# Categories:
UNKNOWN = 0
AGENT = 1
VEHICLES = 2
PEDS = 3
ROAD = 4
OBJECT = 5


def map_to_categories(categories_img_inp, categories_map):
    return tf.gather(params=categories_map, indices=categories_img_inp)


loss_format = ("loss", "top_view")
@implements_format(*loss_format)
def loss(out, label, reg_constant=1.0, alpha=1.0, beta=1.0):
    vehicles_label = tf.cast(tf.equal(label, VEHICLES), tf.float32)
    vehicles_out = tf.slice(out, [0, 0, 0, 0], [-1, -1, -1, 1])
    _ = identity(vehicles_out,"vehicles_out")
    loss_vehicles = tf.keras.losses.binary_crossentropy(vehicles_label, vehicles_out)
    peds_labels = tf.cast(tf.equal(label, PEDS), tf.float32)
    peds_out = tf.slice(out, [0, 0, 0, 1], [-1, -1, -1, 1])
    _ = identity(peds_out, "peds_out")
    loss_peds = tf.keras.losses.binary_crossentropy(peds_labels, peds_out)
    loss_both = alpha * loss_vehicles + beta * loss_peds

    # We don't want to punish on unknown or agent
    mask_unknown = tf.logical_and(tf.not_equal(label, UNKNOWN), tf.not_equal(label, AGENT))
    mask_unknown = tf.cast(mask_unknown, tf.float32)
    mask_unknown = tf.squeeze(mask_unknown, [3])
    masked_loss = mask_unknown * loss_both

    _ = identity(tf.reduce_mean(input_tensor=mask_unknown * loss_vehicles), "loss_vehicles")
    _ = identity(tf.reduce_mean(input_tensor=mask_unknown * loss_peds), "loss_peds")
    loss_categories = tf.reduce_mean(input_tensor=masked_loss)

    # regularization
    reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
    loss_reg = reg_constant * tf.reduce_sum(input_tensor=reg_losses)

    total_loss = loss_categories + loss_reg
    _ = tf.identity(loss_categories, name="loss_categories")
    _ = tf.identity(total_loss, name="loss")
    _ = identity(color_categories(tf.squeeze(label, [3])), "rgb_label")
    _ = identity(tf.expand_dims(masked_loss, -1), name='loss_image')
    _ = tf.identity(loss_reg, name="loss_reg")
    return total_loss

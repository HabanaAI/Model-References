from stereo.data.map_func.map_func_utils import sample_id, origin_l2_focal_l2, crop_pad_images_and_fix_origin


def map_func(features, pred_mode=False):

    features = sample_id(features, pred_mode=pred_mode)

    # assuming input is (height, width) = (310, 720) crop and pad to (320, 704) to produce multiples of 2**6
    image_names = ['I_cntr', 'I_srnd_0', 'I_srnd_1', 'I_srnd_2']
    features = crop_pad_images_and_fix_origin(features, (0, 0), (8, 8), (5, 5), (0, 0), image_names,
                                              pred_mode=pred_mode)

    # this avoids addition and division with a const at architecture itself
    features = origin_l2_focal_l2(features, pred_mode=pred_mode, unet_levels=6)

    return features



from stereo.data.map_func.map_func_utils import xy, blur_kernels, crop_pad_images_and_fix_origin


def map_func(features, consts, blur_kernels_names, pred_mode=False):

    # assuming input is (height, width) = (310, 720) crop and pad to (320, 704) to produce multiples of 2**6
    image_names = ['deeplab', 'im_lidar_inv', 'im_lidar_short_inv', 'im_mask', 'photo_loss_im_mask']
    features = crop_pad_images_and_fix_origin(features, (0, 0), (8, 8), (5, 5), (0, 0), image_names,
                                              pred_mode=pred_mode)

    # add x, y images and blur kernels for photo loss
    features = xy(features, im_name='I_cntr', pred_mode=pred_mode)
    features = blur_kernels(features, consts, blur_kernels_names)

    return features

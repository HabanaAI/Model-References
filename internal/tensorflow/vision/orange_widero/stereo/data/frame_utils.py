import cv2
import numpy as np
from copy import copy
from scipy.signal import convolve2d
from stereo.common.gt_packages_wrapper import NonExistingGiError

def get_img(clip, grab_index, camera_name, tone_map='gtm', level=-2, frame_id=None):
    if tone_map in ['ltm', 'gtm']:
        frame = clip.get_frame(grab_index=grab_index, camera_name=camera_name,
                               tone_map=tone_map, frame_id=frame_id)
        meim = copy(frame[0]['pyr'][level])
    elif tone_map in ['red', 'clear']:
        assert (level == -2)
        frame = clip.get_frame(grab_index=grab_index, camera_name=camera_name,
                               tone_map='gtm', frame_id=frame_id)
        meim = copy(frame[0]['raw_image'])
        meim.im = rcc_interp(meim.im, red_or_clear=tone_map)
        meim.im = meim.im / 128.0 if tone_map == 'red' else meim.im / 256.0

    meim.im = np.minimum(np.maximum(meim.im, 0.0), 255)
    return meim


def rcc_interp(raw_im, red_or_clear='red'):
    assert (red_or_clear in ['red', 'clear'])
    if red_or_clear == 'clear':
        K = np.ones((3, 3)) / 8.0
        clear_im = raw_im.copy()
        intrp_im = raw_im.astype('float32')
        intrp_im[::2, ::2] = 0.0
        intrp_im = convolve2d(intrp_im, K, mode='same')
        clear_im[::2, ::2] = intrp_im[::2, ::2]
        return clear_im.astype(raw_im.dtype)
    else:
        K = np.array([[0.25, 0.5, 0.25],
                      [0.50, 1.0, 0.50],
                      [0.25, 0.5, 0.25]])
        red_im = np.zeros_like(raw_im, dtype='float32')
        red_im[::2, ::2] = raw_im[::2, ::2]
        red_im = convolve2d(red_im, K, mode='same')
        return red_im.astype(raw_im.dtype)


def crop_scale_meimage_to_arr(meimage, K, scale, crop_rect, dtype=None, interp_method=cv2.INTER_AREA):
    meimage_cropped = meimage.cut(crop_rect)
    if dtype is not None:
        meimage_cropped.im = meimage_cropped.im.astype(dtype)
    origin = np.array([meimage_cropped.origin().x, meimage_cropped.origin().y])
    focal = np.diag(K[:2, :2])
    if scale < 1.0:
        im_sz_orig = np.array(meimage_cropped.im.shape)
        im_sz = np.round(im_sz_orig * scale).astype(int)
        im_scaled = cv2.resize(meimage_cropped.im, (im_sz[1], im_sz[0]), interpolation=interp_method)
        scale_xy = np.flip(im_sz.astype('float32') / im_sz_orig, 0)
        focal_scaled = focal * scale_xy
        assert(focal_scaled[0] == focal_scaled[1])
        focal_scaled = focal_scaled[0]
        origin_scaled = origin * scale_xy
        return im_scaled, focal_scaled, origin_scaled
    elif scale == 1.0:
        return meimage_cropped.im, focal, origin
    else:
        assert False


def get_distorted_geometry(clip, camera_name, level):
    frame = None
    for gfi in range(20):  # Sometimes gfi 0 is missing. Attempt getting a later frame
        try:
            frame = clip.get_frame(gfi, camera_name=camera_name)
            break
        except NonExistingGiError:
            continue
    assert frame
    img = frame[0]['pyr'][level]
    origin = (img.origin().x, img.origin().y)
    im_sz = img.im.shape
    focal = clip.focal_length(camera_name, level=level)
    return im_sz, focal, origin

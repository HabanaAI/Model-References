from stereo.data.frame_utils import get_img
from stereo.data.clip_utils import get_camK, get_transformation_matrix
from devkit.basic_types import Rect
from devkit.meimage_utils import MeImage

import numpy as np
import cv2

INTERP = {'cubic': cv2.INTER_CUBIC, 'nearest': cv2.INTER_NEAREST, 'linear': cv2.INTER_LINEAR}


def rectify_interpolation(xs, ys, image, dtype=np.float64, interp='cubic', borderValue=np.nan):
    """
    interpolation method used in rectify image, you can pass similar functions to rectify_image if you want
    different dtype or interpolation methods
    :param xs: x coordinates either 1 or 2d
    :param ys: y coordinates either 1 or 2d
    :param image: image to interpolate
    :param dtype: dtype of the returned image either float or uint8
    :param interp: interpolation method nearet, cubic, linear
    :param borderValue: what value to fill on pixels outside of the image
    :return: image colors sampled at given coordinates
    """
    return cv2.remap(image.astype(dtype, copy=False), xs.astype(np.float32, copy=False),
                     ys.astype(np.float32, copy=False), INTERP[interp], borderValue=borderValue)


class ViewImageInput(object):

    def __init__(self, clip, camera_from, camera_to, level, undistort, crop_rect, car_body_mask=None, RT_modification=None):

        self.clip = clip
        self.camera_from = camera_from
        self.camera_to = camera_to
        self.level = level
        self.undistort = undistort
        self.crop_rect = crop_rect
        self.register_rectification(RT_modification)
        self.car_body_mask = car_body_mask
        self.car_body_mask_img = None

    def get_car_body_mask_meimage(self):
        assert self.car_body_mask_img is not None
        return self.car_body_mask_img

    def register_car_body_mask(self, example_img):
        if self.car_body_mask is None:
            car_body_mask_level = np.zeros_like(example_img.im)
        else:
            car_body_mask_level = cv2.resize(self.car_body_mask, (example_img.im.shape[1], example_img.im.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
        car_body_mask_img = MeImage(car_body_mask_level, example_img.origin(), example_img.image_rect())
        self.car_body_mask_img = self.rectify_image(car_body_mask_img, interp='nearest')

    def rectify_image(self, img, interp='cubic'):
        im_r = rectify_interpolation(self.X + img.origin().x, self.Y + img.origin().y, img.im, interp=interp)
        im_r = im_r.reshape([int(self.rectify_image_rect.height()), int(self.rectify_image_rect.width())])
        abs_rect = Rect(0, im_r.shape[1], 0, im_r.shape[0])
        abs_rect -= self.rectify_origin
        im_r = MeImage(im_r, self.rectify_origin, abs_rect).cut_image_rect()
        return im_r

    def register_rectification(self, RT_modification=None):
        K = get_camK(self.clip, self.camera_from, level=self.level)
        l, r, b, t = self.crop_rect[0], self.crop_rect[1], self.crop_rect[2], self.crop_rect[3]
        self.rectify_image_rect = Rect(l, r + 1, b, t + 1).floor().as_type(np.int)
        self.rectify_origin = -self.rectify_image_rect.bottom_left()
        x, y = np.meshgrid(np.arange(self.rectify_image_rect.left, self.rectify_image_rect.right),
                           np.arange(self.rectify_image_rect.bottom, self.rectify_image_rect.top))
        x = x.astype('float32')
        y = y.astype('float32')
        self.RT_orig_to_rectified = get_transformation_matrix(self.clip, self.camera_from, self.camera_to,
                                                              RT_modification=RT_modification)
        self.RT_orig_to_rectified[:3, 3] = 0
        if self.undistort and self.camera_to:
            self.K_from = K.copy()
            self.K_to = get_camK(self.clip, self.camera_to, level=self.level)
            self.K_from[:2, 2] = 0
            self.K_to[:2, 2] = 0

            H = self.K_to.dot(self.RT_orig_to_rectified[:3, :3].dot(np.linalg.inv(self.K_from)))
            xy_ = np.stack([x.flatten(), y.flatten(), np.ones([x.size, ], dtype='float32')], axis=1)
            xy_t = np.linalg.inv(H.astype('float32')).dot(xy_.T).T
            xy_t[:, 2] = np.maximum(xy_t[:, 2], np.float32(1e-8))
            xy_t = xy_t / xy_t[:, [2]]
            x, y = xy_t[:, 0].reshape(x.shape), xy_t[:, 1].reshape(x.shape)
        self.X, self.Y = self.clip.unrectify(x, y, cam=self.camera_from, level=self.level)

    def get_frame(self, gi, tone_map='gtm', frame_id=None):
        """
        Return MeImage at gi, possibly after undistortion and rectification. Also returns matching K matrix
        of resulting image.
        """
        if frame_id:
            gi = None
        img = get_img(self.clip, gi, camera_name=self.camera_from, tone_map=tone_map, level=self.level, frame_id=frame_id)
        if self.car_body_mask_img is None:
            self.register_car_body_mask(img)
        K = get_camK(self.clip, self.camera_from, level=self.level)

        if self.undistort and not self.camera_to:
            assert False

        elif self.undistort and self.camera_to:
            K_to = self.K_to.copy()
            img_rectified = self.rectify_image(img)
            K_to[:2, [2]] = np.array([[img_rectified.origin().x, img_rectified.origin().y]]).T
            return img_rectified, K_to, self.RT_orig_to_rectified

        elif (not self.undistort) and (self.camera_to == self.camera_from):
            K[:2, [2]] = np.array([[img.origin().x, img.origin().y]]).T
            return img, K, self.RT_orig_to_rectified

        else:
            raise Exception('Distorted views cannot be rectified')

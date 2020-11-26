from bisect import bisect_left
import numpy as np
from skimage.draw import polygon
from collections import OrderedDict

from stereo.data.frame_utils import crop_scale_meimage_to_arr
from stereo.data.clip_utils import get_camK

from stereo.common.gt_packages_wrapper import itrk2df

from devkit.basic_types import MeImage

from external_import import external_import
with external_import('/mobileye/shared/Tools/qa_algo_tools/test_tools/latest'):
    from base.itrk_parser import ItrkParser
with external_import('/mobileye/shared/Tools/qa_algo_tools/vdTestTools'):
    from vdTestTools.common.vd_3d_box import VD3D_box


class InstanceMaskGenerator(object):
    def __init__(self, clip, cam, level, scale, crop_rect_at_level, itrk_path, vd=True):
        """
        :param clip: a FoF clip
        :param cam: camera name for which masks should be generated
        :param scale: resize level image by scale factor after crop
        :param crop_rect_at_level: rect around origin to crop at given level
        :param itrk_path: a detection mest generated itrk for this camera
        """
        # I create this once per clip and then I use it for any warps I have to do and to help in creating masks.
        #
        # It's better to prep this in the beginning for every camera but in that case you should use the multi_cip
        # object. I have added those changes and commented them out.
        #
        # For demonstration purposes, here I do it only on one clip (LEFT)

        self.clip = clip
        self.level = level
        self.vd = vd

        if vd:
            itrk_parser = ItrkParser(itrk_path, {'VD3D': ['IF']})
            self.cam_port = int(itrk_parser.camera_to_camport[cam])  # TODO: just for cam_port
            self.df = itrk2df(itrk_path, 'VD3D', 'IF')
        else:
            itrk_parser = ItrkParser(itrk_path, {'Ped': ['Clq']})
            self.cam_port = int(itrk_parser.camera_to_camport[cam])  # TODO: just for cam_port
            self.df = itrk2df(itrk_path, 'Ped', 'Clq')
        self.gi_to_gfi = OrderedDict()
        self.gi_to_gfi.update(zip(self.df['grab_index'], self.df['gfi']))
        self.known_gis = self.gi_to_gfi.keys()

        self.mesh_dict = {}
        temp_im = clip.get_frame(grab_index=clip.get_grab_by_gfi(camera_name=cam, gfi=0), camera_name=cam)
        temp_im = temp_im[0]['pyr'][level]
        temp_im_r = clip.rectify_image(image=temp_im, cam=cam, level=level, valid_area=False)
        self.mesh_dict['cam'] = cam
        self.mesh_dict['K'] = get_camK(self.clip, self.mesh_dict['cam'], self.level)
        self.mesh_dict['image_rect'] = temp_im.image_rect()
        self.mesh_dict['image_rect_r'] = temp_im_r.image_rect()
        self.mesh_dict['origin'] = temp_im.origin()
        self.mesh_dict['origin_r'] = temp_im_r.origin()
        self.mesh_dict['X'], self.mesh_dict['Y'] = np.meshgrid(np.arange(temp_im.im.shape[1]),
                                                               np.arange(temp_im.im.shape[0]))
        self.mesh_dict['X_r'], self.mesh_dict['Y_r'] = np.meshgrid(np.arange(temp_im_r.im.shape[1]),
                                                                   np.arange(temp_im_r.im.shape[0]))
        self.mesh_dict['X'] -= self.mesh_dict['origin'].x
        self.mesh_dict['Y'] -= self.mesh_dict['origin'].y
        self.mesh_dict['X_r'] -= self.mesh_dict['origin_r'].x
        self.mesh_dict['Y_r'] -= self.mesh_dict['origin_r'].y
        self.mesh_dict['ps'] = np.matrix(data=[self.mesh_dict['X_r'].flatten(), self.mesh_dict['Y_r'].flatten(),
                                               np.ones(self.mesh_dict['X_r'].flatten().shape)])
        self.mesh_dict['curr_shape'] = temp_im.im.shape
        self.mesh_dict['curr_r_shape'] = temp_im_r.im.shape

        # taken from stereo.data.frame_utils.crop_scale_meimage_to_arr()
        self.level = level
        self.scale = scale  # maintains crop_scale_meimage_to_arr()'s logic
        self.crop_rect_at_level = crop_rect_at_level

    def get_mask_image(self, gi, rectified):

        masks, ids = self.get_Masks(gi, rectified)
        masks_compressed, ids = bit_compress(masks, ids)
        return masks_compressed, ids

    def get_Masks(self, grab_index, rectify=False):
        # NOTE that the itrk_parser must correspond to the cam, so frontCornerLeft should receive the LEFT clip's
        # itrk_parser while rear should get the REAR clip's itrk_parser
        # RETURNS: list of quad masks as uint8 numpy arrays in order from largest to smallest, corresponding public ids

        if grab_index < self.known_gis[0] or grab_index > self.known_gis[-1]:
            clqs = []
        else:
            gi_ind = bisect_left(self.known_gis, grab_index)
            if grab_index == self.known_gis[gi_ind]:
                gfi = self.gi_to_gfi[grab_index]
            else:
                gi_prev = self.known_gis[gi_ind-1]
                gi_next = self.known_gis[gi_ind]
                gfi = self.gi_to_gfi[gi_prev] if abs(gi_prev - grab_index) < abs(gi_next - grab_index) else \
                    self.gi_to_gfi[gi_next]
            clqs = self.df.loc[(self.df['gfi'] == gfi) &
                                     (self.df['camPort'] == int(self.cam_port))].to_dict('records')

        level_factor = np.power(2., -self.level)
        masks = []
        public_ids = []
        for clq in clqs:
            mask = np.zeros(self.mesh_dict['curr_shape'])
            box_obj = VD3D_box(clq)
            # Take only the approved VDs
            if np.bool(np.int32(clq['isApproved'])) and clq['camPort'] == self.cam_port:
                if not (box_obj.rect or box_obj.quadrate):
                    continue

                if box_obj.quadrate:
                    coords = np.zeros((4, 2))
                    coords[0, :] = [box_obj.quadrate.lb_point.x, box_obj.quadrate.lb_point.y]
                    coords[1, :] = [box_obj.quadrate.lt_point.x, box_obj.quadrate.lt_point.y]
                    coords[2, :] = [box_obj.quadrate.rt_point.x, box_obj.quadrate.rt_point.y]
                    coords[3, :] = [box_obj.quadrate.rb_point.x, box_obj.quadrate.rb_point.y]
                    r, c = polygon(coords[:, 1] * level_factor + self.mesh_dict['origin'].y,
                                   coords[:, 0] * level_factor + self.mesh_dict['origin'].x)
                    check = np.logical_and(np.logical_and(r > 0, r < self.mesh_dict['curr_shape'][0] - 1),
                                           np.logical_and(c > 0, c < self.mesh_dict['curr_shape'][1] - 1))
                    mask[r[check], c[check]] = 1.
                if box_obj.rect:
                    l = box_obj.rect.left * level_factor
                    r = box_obj.rect.right * level_factor
                    b = box_obj.rect.bottom * level_factor
                    t = box_obj.rect.top * level_factor
                    mask[np.logical_and(np.logical_and(self.mesh_dict['X'] > l,
                                                          self.mesh_dict['X'] <= r),
                                           np.logical_and(self.mesh_dict['Y'] > b, self.mesh_dict['Y'] <= t))] = 1

            mask = MeImage(im=mask, origin=self.mesh_dict['origin'], image_rect=self.mesh_dict['image_rect'])

            if rectify:
                mask = self.clip.rectify_image(image=mask, cam=self.mesh_dict['cam'], level=self.level,
                                               valid_area=False)
                mask.im[np.isnan(mask.im)] = 0
                mask.im[mask.im > 0.1] = 1
                mask.im[mask.im <= 0.1] = 0

            mask = mask.as_type(np.uint8)

            mask, _, _ = crop_scale_meimage_to_arr(mask, self.mesh_dict['K'], self.scale, self.crop_rect_at_level, dtype='uint8')

            mask[np.isnan(mask)] = 0
            mask[mask > 0.1] = 1
            mask[mask <= 0.1] = 0

            if np.sum(mask) > 0:
                masks.append(mask.astype(np.uint8))
                if self.vd:
                    public_ids.append(int(clq['public_id']))
                else:
                    public_ids.append(int(clq['publicID']))

        if not masks:

            shape_im = MeImage(im=np.zeros(self.mesh_dict['curr_shape'], dtype=np.uint8),
                               origin=self.mesh_dict['origin'], image_rect=self.mesh_dict['image_rect'])
            if rectify:
                shape_im = self.clip.rectify_image(image=shape_im, cam=self.mesh_dict['cam'], level=self.level,
                                                   valid_area=False)

            shape_im, _, _ = crop_scale_meimage_to_arr(shape_im, self.mesh_dict['K'], self.scale,
                                                       self.crop_rect_at_level, dtype='uint8')

            masks = [shape_im] * 8
            public_ids = [int(0)] * 8

        # Sort list by np.sum on each element so the biggest boxes are first
        masks, ids = (list(t) for t in
                            zip(*sorted(zip(masks, public_ids), key=lambda tup: np.sum(tup[0]), reverse=True)))

        return masks[:8], ids[:8]


def bit_compress(mask_lst, id_lst):
    # INPUT: a list of uint8 numpy arrays and a list of scalars
    # OUTPUT: a compressed uint8 bit array and list of scalars
    # NOTE this will alter the input lists if they have less than 8 items and because of how python works
    # they will stay altered

    if len(mask_lst) > 8:
        raise ValueError('Mask lists should have no more than 8 items for each bit array')
    if len(mask_lst) == 0:
        raise ValueError('Mask lists should not be empty')

    while len(mask_lst) < 8:
        mask_lst.append(np.zeros_like(mask_lst[0], dtype=np.uint8))
        id_lst.append(0)

    compressed_array = np.array(mask_lst)
    compressed_array = np.squeeze(np.packbits(compressed_array.astype(np.uint8), axis=0))
    id_lst = np.array(id_lst)
    return compressed_array, id_lst


def unpack_bit_array(compressed_array):
    # This is for testing purposes, it doesn't need to be in the dump. It reverses the bit compress function.
    out_bits = np.unpackbits(np.expand_dims(compressed_array, axis=0), axis=0)
    return out_bits

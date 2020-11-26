from bisect import bisect_left
from collections import OrderedDict
import numpy as np
from stereo.common.gt_packages_wrapper import itrk2df

from external_import import external_import
with external_import('/mobileye/shared/Tools/qa_algo_tools/vdTestTools'):
    from vdTestTools.common.vd_3d_box import VD3D_box
with external_import('/mobileye/shared/mepy_packages'):
    from devkit.basic_types import Rect
    from test_tools.base.itrk_parser import ItrkParser
from stereo.data.frame_utils import get_distorted_geometry

def rectify_rect(rectify, rect):
    x = np.array([rect.left, rect.right, rect.right, rect.left])
    y = np.array([rect.top, rect.top, rect.bottom, rect.bottom])
    r_x, r_y = rectify(x, y)
    r_lrbt = np.array([np.min(r_x), np.max(r_x), np.min(r_y), np.max(r_y)])
    return Rect().from_array(r_lrbt)


def enlarge_rect(rect, ratio):
    rect[:2] = (rect[:2] - np.mean(rect[:2])) * ratio + np.mean(rect[:2])
    rect[2:] = (rect[2:] - np.mean(rect[2:])) * ratio + np.mean(rect[2:])
    return rect


def rect_includes(rect, pts):
    if isinstance(rect, (list, np.ndarray)):
        rect = Rect().from_array(rect)
    inc = (rect.left <= pts[:, 0]) * (rect.right-1 >= pts[:, 0]) * \
        (rect.bottom <= pts[:, 1]) * (rect.top - 1 >= pts[:, 1])
    return inc

def cropped_rect_in_image_rel(rect_lrbt, im_sz, origin):
    return np.minimum(np.maximum(int(np.round(rect_lrbt[0] + origin[0])), 1), im_sz[1]-1) - origin[0], \
           np.minimum(np.maximum(int(np.round(rect_lrbt[1] + origin[0])), 1), im_sz[1]-1) - origin[0], \
           np.minimum(np.maximum(int(np.round(rect_lrbt[2] + origin[1])), 1), im_sz[0]-1) - origin[1] , \
           np.minimum(np.maximum(int(np.round(rect_lrbt[3] + origin[1])), 1), im_sz[0]-1) - origin[1]

def cropped_rect_in_image(rect_lrbt, im_sz, origin):
    return np.minimum(np.maximum(int(np.round(rect_lrbt[0] + origin[0])), 0), im_sz[1]), \
           np.minimum(np.maximum(int(np.round(rect_lrbt[1] + origin[0])), 0), im_sz[1]), \
           np.minimum(np.maximum(int(np.round(rect_lrbt[2] + origin[1])), 0), im_sz[0]), \
           np.minimum(np.maximum(int(np.round(rect_lrbt[3] + origin[1])), 0), im_sz[0])


class VDPDMasks(object):

    def __init__(self, clip, camera_name, itrk_path, enlarge_ratio=1.1):
        """
        Init VDPDMasks
        :param clip: a FoF clip
        :param camera_name: camera name for which masks should be generated
        :param itrk_path: a detection mest generated itrk for this camera
        :param enlarge_ratio: factor by which to enlarge detection rects
        """
        self.clip = clip
        self.im_sz_l0, _, self.origin_l0 = get_distorted_geometry(self.clip, camera_name, 0)
        self.itrk_path = itrk_path
        self.camera_name = camera_name
        self.enlarge_ratio = enlarge_ratio

        # keep cam_port and load vd and pd dfs from itrk
        itrk_parser = ItrkParser(itrk_path, {'VD3D': ['MF'], 'Ped': ['Clq']}, agenda='all')  # TODO: just for cam_port
        self.cam_port = itrk_parser.camera_to_camport[camera_name]
        self.vd_mf_df = itrk2df(itrk_path, 'VD3D', 'MF')
        self.vd_mobility_df = itrk2df(itrk_path, 'VD3D', 'Mobility')
        self.ped_clq_df = itrk2df(itrk_path, 'Ped', 'Clq')

        # keep a list of known gis based on itrk df rows
        self.gi_to_gfi = OrderedDict()
        self.known_gis = []


        # find pubids of moving vd and approved peds
        self.moving_vd_pubids = []
        self.approved_ped_pubids = []

        if self.vd_mf_df is not None:
            vd_mf_df_moving = self.vd_mf_df[self.vd_mf_df['mfIsMoving'] > 0]
            self.moving_vd_pubids = np.unique(vd_mf_df_moving['public_id'].values)
            self.gi_to_gfi.update(zip(self.vd_mf_df['grab_index'], self.vd_mf_df['gfi']))

        if self.vd_mobility_df is not None:
            vd_mobility_df_moved = self.vd_mobility_df[self.vd_mobility_df['never_moved'] == 0]
            vd_mobility_df_seen_moved = self.vd_mobility_df[self.vd_mobility_df['never_seen_monving'] == 0]
            self.moving_vd_pubids = np.unique(np.concatenate([self.moving_vd_pubids,
                                                              vd_mobility_df_moved['public_id'].values,
                                                              vd_mobility_df_seen_moved['public_id'].values]))
        if self.ped_clq_df is not None:
            ped_clq_df = self.ped_clq_df[self.ped_clq_df['isApproved'] == 1]
            self.approved_ped_pubids = np.unique(ped_clq_df['publicID'].values)
            self.gi_to_gfi.update(zip(self.ped_clq_df['grab_index'], self.ped_clq_df['gfi']))

        if self.vd_mf_df is not None or self.ped_clq_df is not None:
            self.known_gis = sorted(self.gi_to_gfi.keys())

    def get_mask_image(self, gi, im_sz, focal, origin, name, rectified):
        """
        return mask image
        :param gi: grab index at which to compute rects
        :param im_sz: shape of mask image
        :param focal: focal lengths [fx, fy] of mask image
        :param origin: origin [ox, oy] of mask image
        :param name: detection name. One of ['VD', 'PD', 'VD_moving', 'PD_approved', 'VD+PD', 'VD_moving+PD', ...]
        :param rectified: should rects be computed in undistorted image coordinates or not
        :return: a uint8 image with 1 at detection pixels and 0 otherwise
        """
        level = -2
        clip_focal_at_level = self.clip.focal_length(self.camera_name, level=level)
        scale = focal / clip_focal_at_level
        img_mask = np.zeros(im_sz, dtype='uint8')
        rects = self.get_rects(gi, name, level, rectified)
        for rect in rects:
            scaled_rect = rect * scale
            l, r, b, t = cropped_rect_in_image(scaled_rect, im_sz, origin)
            img_mask[b:t, l:r] = 1
        return img_mask

    def get_rects(self, gi, name, level, rectified):
        """
        return rects at any gi
        :param gi: grab index at which to compute rects
        :param name: detection name
        :param level: rects computed at this level
        :param rectified: should rects be computed in undistorted image coordinates or not
        :return: a list of rects, each encoded as an (l, r, b, t) tuple
        """
        if gi < self.known_gis[0] or gi > self.known_gis[-1]:
            return []
        gi_ind = bisect_left(self.known_gis, gi)
        if gi == self.known_gis[gi_ind]:
            rects_vd, rects_ped = \
                self.get_rects_T0(gi, name, level, rectified)
            return rects_vd.values() + rects_ped.values()
        else:
            gi_prev = self.known_gis[gi_ind-1]
            gi_next = self.known_gis[gi_ind]
            rects_vd_prev, rects_ped_prev = \
                self.get_rects_T0(gi_prev, name, level, rectified)
            rects_vd_next, rects_ped_next = \
                self.get_rects_T0(gi_next, name, level, rectified)
            ts = self.clip.ts_map.ts_by_gi(gi)
            ts_prev = self.clip.ts_map.ts_by_gi(gi_prev)
            ts_next = self.clip.ts_map.ts_by_gi(gi_next)
            alpha = (ts - ts_prev) / (ts_next - ts_prev)
            matched_vd_ids = set(rects_vd_prev).intersection(set(rects_vd_next))
            matched_ped_ids = set(rects_ped_prev).intersection(set(rects_ped_next))
            rects_vd = [rects_vd_prev[_id]*(1-alpha) + rects_vd_next[_id]*alpha for _id in matched_vd_ids]
            rects_ped = [rects_ped_prev[_id]*(1-alpha) + rects_ped_next[_id]*alpha for _id in matched_ped_ids]
            return rects_vd + rects_ped

    def get_rects_T0(self, gi, name, level, rectified):
        """
        return rects at T0 gi's
        :param gi: grab index at which to compute rects
        :param name: detection name. One of ['VD', 'PD', 'VD_moving', 'PD_approved', 'VD+PD', 'VD_moving+PD', ...]
        :param level: rects computed at this level
        :param rectified: should rects be computed in undistorted image coordinates or not
        :return: a list of rects, each encoded as an (l, r, b, t) tuple
        """
        gfi = self.gi_to_gfi[gi]
        vd_clqs = self.vd_mf_df.loc[(self.vd_mf_df['gfi'] == gfi) &
                                    (self.vd_mf_df['camPort'] == int(self.cam_port))].to_dict('records')
        ped_clqs = self.ped_clq_df.loc[(self.ped_clq_df['gfi'] == gfi) &
                                       (self.ped_clq_df['camPort'] == int(self.cam_port))].to_dict('records')

        mask_rects_vd = {}
        mask_rects_ped = {}

        if 'VD' in name:
            for clq in vd_clqs:
                if (int(clq['public_id']) in self.moving_vd_pubids) or 'VD_moving' not in name:
                    box_obj = VD3D_box(clq)
                    rect = box_obj.get_blocking_rect()
                    if rectified:
                        l, r, b, t = cropped_rect_in_image_rel([rect.left, rect.right, rect.bottom, rect.top],
                                                           self.im_sz_l0, self.origin_l0)
                        rect = rectify_rect(lambda x, y: self.clip.rectify(x, y, cam=self.camera_name, level=0),
                                            Rect().from_array([l, r ,b, t]))
                    if rect is None:
                        continue
                    rect_lrbt = np.array([rect.left, rect.right, rect.bottom, rect.top]) * (2**(-level))
                    rect_lrbt = enlarge_rect(rect_lrbt, self.enlarge_ratio)
                    if not np.any(np.isnan(rect_lrbt)):
                        mask_rects_vd[int(clq['public_id'])] = rect_lrbt

        if 'PD' in name:
            for clq in ped_clqs:
                if (int(clq['publicID']) in self.approved_ped_pubids) or 'PD_approved' not in name:
                    rect = [clq['cam_left'], clq['cam_right'], clq['cam_bottom'], clq['cam_top']]
                    rect = Rect().from_array(rect)
                    if rectified:
                        l, r, b, t = cropped_rect_in_image_rel([rect.left, rect.right, rect.bottom, rect.top],
                                                           self.im_sz_l0, self.origin_l0)
                        rect = rectify_rect(lambda x, y: self.clip.rectify(x, y, cam=self.camera_name, level=0),
                                            Rect().from_array([l, r, b, t]))
                        if rect is None:
                            continue
                    rect_lrbt = np.array([rect.left, rect.right, rect.bottom, rect.top]) * (2 ** (-level))
                    rect_lrbt = enlarge_rect(rect_lrbt, self.enlarge_ratio)
                    if not np.any(np.isnan(rect_lrbt)):
                        mask_rects_ped[int(clq['publicID'])] = rect_lrbt

        return mask_rects_vd, mask_rects_ped

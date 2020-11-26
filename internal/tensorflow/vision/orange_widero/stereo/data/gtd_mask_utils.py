import numpy as np
from matplotlib.path import Path

from stereo.common.gt_packages_wrapper import VdStateManager, VclPlugin, PedPlugin, GtdDB, apply_rt

from dcorrect.distortionCorrection import DistortionCorrection
from stereo.data.clip_utils import clip_path

from scipy.spatial import ConvexHull


def line2image(p1, p2, dc, focal):
    projected1 = (p1[:-1] / p1[-1]) * focal
    me_pt1 = dc.distort(projected1[0], projected1[1], sfm2me=True)
    projected2 = (p2[:-1] / p2[-1]) * focal
    me_pt2 = dc.distort(projected2[0], projected2[1], sfm2me=True)
    if np.isnan(p1).any() or np.isnan(p1).any() or (p1[2] < 0.1 and p2[2] < 0.1) or (
            np.isnan(me_pt1).any() and np.isnan(me_pt2).any()):
        return np.ones((3,)) * np.nan, np.ones((3,)) * np.nan
    swap = False
    if p1[2] >= 0.1 and not np.isnan(me_pt1).any():
        pp1 = np.array(p1)
        pp2 = np.array(p2)
    else:
        swap = True
        pp1 = np.array(p2)
        pp2 = np.array(p1)
    t = 1.0

    for i in np.arange(30):
        ppp2 = pp1 + (pp2 - pp1) * t
        if ppp2[2] >= 0.1:
            projected = (ppp2[:-1] / ppp2[-1]) * focal
            me_pt = dc.distort(projected[0], projected[1], sfm2me=True)
            if not np.isnan(me_pt[0]):
                break
            else:
                t = 0.9 * t
        else:
            t = 0.8 * t
    if swap:
        return ppp2, pp1
    else:
        return pp1, ppp2


def cam2me(cam_points, dc, focal):
    assert isinstance(cam_points, np.ndarray)
    assert len(cam_points.shape) == 2
    assert cam_points.shape[0] == 3
    projected = (cam_points[:-1, :] / cam_points[-1, :]) * focal
    me_pts = dc.distort(projected[0, :], projected[1, :], sfm2me=True)
    final_x, final_y = me_pts
    final_x[cam_points[-1, :] < 0] = np.nan
    final_y[cam_points[-1, :] < 0] = np.nan
    return final_x, final_y


def box_H(state):
    minH = 1.5
    maxH = 3.5
    minL = 4.5
    maxL = 8.
    L = np.maximum(np.minimum(state[-1], maxL), minL)
    H = ((L - minL) / (maxL - minL)) * (maxH - minH) + minH
    return H


def proj_state(state, clip, cam, dc, enlarge_ratio=1.0):
    state_ = np.array(state)
    H = box_H(state)
    t_in_target = VdStateManager.get_vehicle_box(state_, roof=H) * enlarge_ratio
    target2body = VdStateManager.state2RT(state)
    t_in_body = VdStateManager.move_veh(t_in_target, target2body)
    target = VdStateManager.move_veh(t_in_body, clip.get_c2v(cam, inv=True))
    [rear_left_bottom, rear_left_top, rear_right_bottom,
     rear_right_top, front_left_bottom, front_left_top,
     front_right_bottom, front_right_top] = [np.squeeze(item) for item in np.split(target.T, 8, axis=0)]
    rear_left_bottom, front_left_bottom = line2image(rear_left_bottom, front_left_bottom, dc, clip.get_fl(cam))
    rear_right_bottom, front_right_bottom = line2image(rear_right_bottom, front_right_bottom, dc, clip.get_fl(cam))
    rear_left_top, front_left_top = line2image(rear_left_top, front_left_top, dc, clip.get_fl(cam))
    rear_right_top, front_right_top = line2image(rear_right_top, front_right_top, dc, clip.get_fl(cam))
    points = np.stack([rear_left_bottom, rear_left_top, rear_right_bottom,
                       rear_right_top, front_left_bottom, front_left_top,
                       front_right_bottom, front_right_top]).T
    me_points_x, me_points_y = cam2me(points, dc, clip.get_fl(cam))
    return np.stack([me_points_x, me_points_y], axis=1)


def proj_ped(ped, clip, cam, dc):
    position_veh = np.array([ped['x'], ped['y'], ped['z']])
    H = ped['H']
    W_half = ped['W'] / 2.
    L_half = ped['W'] / 2.
    boxPoints = []
    for x in [-W_half, W_half]:
        for y in [0, H]:
            for z in [-L_half, L_half]:
                boxPoints.append(np.array([x, y, z]) + position_veh)
    points = apply_rt(np.stack(boxPoints, axis=1), clip.get_c2v(cam, inv=True))
    me_points_x, me_points_y = cam2me(points, dc, clip.get_fl(cam))
    return np.stack([me_points_x, me_points_y], axis=1)

def points_to_mask(points, meshgrid, im_sz, origin):
    hull = ConvexHull(points)
    poly_verts = Path(np.stack([points[hull.vertices, 0] + origin[0],
                                points[hull.vertices, 1] + origin[1]]).T)
    grid = poly_verts.contains_points(meshgrid)
    return grid.reshape((im_sz[0], im_sz[1]))

class GTD_VDPDMasks(object):

    def __init__(self, clip, clip_name, enlarge_ratio=1.1, speed_thresh_for_moving=2.):
        """
        Init VDPDMasks
        :param clip: a FoF clip
        :param camera_name: camera name for which masks should be generated
        :param itrk_path: a detection mest generated itrk for this camera
        :param enlarge_ratio: factor by which to enlarge detection rects
        """
        self.clip = clip
        self.enlarge_ratio = enlarge_ratio
        self.dc = {}
        self.available_cams = ['main', 'rear', 'frontCornerLeft', 'frontCornerRight', 'rearCornerLeft', 'rearCornerRight']
        for cam in self.available_cams:
            self.dc[cam] = DistortionCorrection(clip_path(clip_name), camera=cam, max_rad=2000, max_rad_inv=30000,
                                           mest_bitexact=False)
        self.vd_data = GtdDB(VclPlugin).get_clip_data(clip_name)
        self.peds_data = GtdDB(PedPlugin).get_clip_data(clip_name)
        assert self.vd_data is not None
        # find pubids of moving vd and approved peds
        self.moving_vd_ids = None if self.vd_data is None \
            else np.unique(self.vd_data.id.values[np.abs(self.vd_data.V.values) > speed_thresh_for_moving])

        self.first_frameId_with_object = sorted(self.vd_data.frame_id.values)[0]

    def get_vd_boxes(self, frameId, cam, name):
        boxes = []
        if self.vd_data is not None:
            states_lines = self.vd_data[self.vd_data.frame_id == frameId]

            for _, line in states_lines.iterrows():
                if int(line['id']) in self.moving_vd_ids or 'VD_moving' not in name:
                    state = VdStateManager.dict2state(line)
                    points = proj_state(state, self.clip, cam, self.dc[cam], enlarge_ratio=self.enlarge_ratio)
                    if np.isnan(points).any():
                        continue
                    boxes.append(points)
        return boxes

    def get_peds_boxes(self, frameId, cam):
        boxes = []
        if self.peds_data is not None:
            peds_lines = self.peds_data[self.peds_data.frame_id == frameId]
            for _, line in peds_lines.iterrows():
                points = proj_ped(line, self.clip, cam, self.dc[cam])
                if np.isnan(points).any():
                    continue
                boxes.append(points)
        return boxes

    def get_mask_image(self, gi, cam, im_sz, focal, origin, name, rectified):
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
        assert cam in self.available_cams
        frameId = self.clip.ts_map.frame_id_by_gi(gi, 'main')
        if frameId < self.first_frameId_with_object:
            print ('gi %\d is before first frameId with GTD object (%\d)' % (gi, self.first_frameId_with_object))
            assert(False)
        x, y = np.meshgrid(np.arange(im_sz[1]), np.arange(im_sz[0]))
        x, y = x.flatten(), y.flatten()
        meshgrid = np.vstack((x, y)).T
        img_mask = np.zeros(im_sz, dtype='uint8')
        clip_focal_at_level = self.clip.focal_length(cam, level=level)
        scale = focal / clip_focal_at_level
        boxes = []
        if 'VD' in name:
            boxes.extend(self.get_vd_boxes(frameId, cam, name))
        if 'PD' in name:
            boxes.extend(self.get_peds_boxes(frameId, cam))
        for points in boxes:
            if rectified:
                points = np.stack(
                    list(self.clip.rectify(points[:, 0], points[:, 1], cam=cam, level=level)),
                    axis=1)
            points = points * scale
            img_mask = np.logical_or(img_mask, points_to_mask(points, meshgrid, im_sz, origin))
        return img_mask


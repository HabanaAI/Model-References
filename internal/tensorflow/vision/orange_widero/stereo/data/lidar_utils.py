from copy import deepcopy
import gc

import numpy as np
from scipy import spatial
from scipy.interpolate import griddata
from scipy.signal import convolve2d

from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from random import randint, choice

from stereo.common.gt_packages_wrapper import apply_rt, dehom, Semantic
from stereo.common.gt_packages_wrapper import get_l2main

from stereo.data.sky_seg_utils import add_missing_lidar
from stereo.data.frame_utils import get_distorted_geometry
from stereo.data.clip_utils import clip_path
from stereo.data.lidar_img_utils import cast_lidar_on_img

try:
    from tqdm import tqdm
except:
    def tqdm(l):
        return l

VIS_THRESH = 0.9437  # The mean of mean visibilities of ~1500 frames

def visibility_estimation(points2d, depth, k=27, mask='thresh', thresh=VIS_THRESH, upper_bound=40):
    """
    Based on:
    Biasutti, Pierre, et al. "Visibility estimation in point clouds with variable density" 2019.
    The difference from the original paper is not using d_max.
    """
    if len(points2d) == 0:
        return [], []

    tree = spatial.cKDTree(points2d)
    # _, knn = tree.query(tree.data, k=k)
    _, knn = tree.query(tree.data, k=k, distance_upper_bound=upper_bound)

    d = np.zeros(points2d.shape[0])

    for j in range(points2d.shape[0]):
        # neighbors = [k for k in knn[j] if k != j]
        neighbors = knn[j]  # neighborhood includes the point itself
        neighbors = neighbors[neighbors < points2d.shape[0]]
        neighbors_dist = np.sort(depth[neighbors])
        d_min = neighbors_dist[0]
        # d_max = neighbors_dist[-1]
        d[j] = (depth[j] - d_min)/d_min

    visibility = np.exp(-(d**2))
    if mask == 'mean':
        visibility_mask = visibility > np.mean(visibility)
    elif mask == 'median':
        visibility_mask = visibility > np.median(visibility)
    elif mask == 'thresh':
        visibility_mask = visibility > thresh
    else:
        raise ValueError("Mask must be 'mean', 'median' or 'thresh'")

    # print "in vis_est, retained %f of points" % (float(np.sum(visibility_mask))/len(visibility_mask))
    return visibility, visibility_mask


def visibility_estimation_imgs(lidar_im, occlusion_params, cluster_im=None):
    """
    Based on:
    Biasutti, Pierre, et al. "Visibility estimation in point clouds with variable density" 2019.
    The difference from the original paper is not using d_max.
    """
    xx, yy = np.meshgrid(np.arange(lidar_im.shape[1]), np.arange(lidar_im.shape[0]))

    points2d = np.c_[yy.flatten(), xx.flatten()]
    depth = lidar_im.flatten()
    valid = depth > 0
    points2d = points2d[valid]
    depth = depth[valid]
    if cluster_im is not None:
        clusters = cluster_im.flatten()[valid]

    _, visibility_mask = visibility_estimation(points2d, depth, **occlusion_params['vis_est_kwargs'])

    depth_im = np.zeros_like(lidar_im)
    depth_im[points2d[visibility_mask, 0], points2d[visibility_mask, 1]] = depth[visibility_mask]

    # print "in vis_est, retained %f of points" % (float(np.sum(visibility_mask)) / len(visibility_mask))

    if cluster_im is not None:
        cluster_im = np.zeros_like(lidar_im)
        cluster_im[points2d[visibility_mask, 0], points2d[visibility_mask, 1]] = clusters[visibility_mask]
        return depth_im, cluster_im

    return depth_im

def z_buffer(points2d, depth, im_sz, origin, scale=1, clusters=None):
    img_lidar = np.zeros(im_sz, dtype='float32')
    p_pixels = np.round(points2d * scale + origin).astype(int)
    # sort by depth
    order = np.argsort(np.copy(depth))[::-1]
    new_depth = depth[order]
    p_pixels = p_pixels[order]
    ind = (p_pixels[:, 0] > 0) & (p_pixels[:, 0] < im_sz[1]) & (
            p_pixels[:, 1] > 0) & (p_pixels[:, 1] < im_sz[0])
    # lower depth values will override higher values
    img_lidar[p_pixels[ind, 1], p_pixels[ind, 0]] = new_depth[ind]

    temp_depth = np.zeros_like(depth)
    temp_depth[order] = new_depth*ind.astype(np.int32)
    mask = temp_depth != 0

    if clusters is not None:
        img_clusters = np.zeros((img_lidar.shape[0], img_lidar.shape[1]))
        clusters = clusters[order, :]
        img_clusters[p_pixels[ind, 1], p_pixels[ind, 0]] = clusters[ind, 1]

        return img_lidar, img_clusters

    return img_lidar, mask

def prevent_colinearity(x1, y1, mask):
    # If all the points in the cluster are on the same x or y line, the 3D interpolation
    # across the cluster throws an error (Qhull)
    if len(np.unique(x1)) == 1:
        rand_idx = randint(0, len(x1)-1)
        if x1[rand_idx] == 0:
            sample_from = [1]
        elif x1[rand_idx] == mask.shape[1] - 1:
            sample_from = [-1]
        else:
            sample_from = [-1,1]
        x1[rand_idx] = x1[rand_idx] + choice(sample_from)

    if len(np.unique(y1)) == 1:
        rand_idx = randint(0, len(y1)-1)
        if y1[rand_idx] == 0:
            sample_from = [1]
        elif y1[rand_idx] == mask.shape[1] - 1:
            sample_from = [-1]
        else:
            sample_from = [-1,1]
        y1[rand_idx] = y1[rand_idx] + choice(sample_from)

    return x1,y1

def interp_lidar(im_lidar, mask_kernel_sz=3):
    if np.all(im_lidar == 0.0):
        im_lidar_interp = np.ones_like(im_lidar) * 10000
        im_lidar_interp_mask = np.zeros_like(im_lidar)
    else:
        x, y = np.meshgrid(range(im_lidar.shape[1]), range(im_lidar.shape[0]))
        lidar_y, lidar_x = np.where(im_lidar > 0)
        lidar_Z = im_lidar[im_lidar > 0]
        im_lidar_interp = griddata((lidar_y, lidar_x), lidar_Z**-1, (y, x), method='linear')**-1
        im_lidar_interp_mask = convolve2d((im_lidar > 0)*1.0, np.ones((mask_kernel_sz, mask_kernel_sz)), 'same') > 0
    return im_lidar_interp, im_lidar_interp_mask

def points_in_box(points, box):
    assert(box.shape[0] == points.shape[1])
    in_box = np.ones((len(points,)), dtype=bool)
    for i in range(box.shape[0]):
        in_box = in_box & ((points[:, i] > box[i, 0]) & (points[:, i] < box[i, 1]))
    return in_box

def simpleProject2image(clip, wpoints, level=-2, cam='main'):
    fl = clip.get_fl(cam=cam, level=level)
    points = dehom(wpoints.T).T * fl
    xs, ys = clip.unrectify(points[:, 0], points[:, 1], cam=cam, level=level)
    return xs, ys

def filter_lidar(gi, clip, masks, level=-2, host_3d_mask=None, sky_from_file=True):

    masks_cams = masks.keys()


    hmask = host_3d_mask
    l2main = clip.lidar_data.get_l2c(cam='main')

    lidar = clip.lidar_data.get_lidar_data(gi, cam='main')
    # TODO add missing lidar points here

    if lidar is None:
        return None, None
    gi_ts = clip.ts_map.ts_by_gi(gi, cam='main') * 1e-6
    wpoints = lidar[['X', 'Y', 'Z']].values
    ptss = lidar['delta'] * 1e-6 + gi_ts
    lidar['point_ts'] = ptss
    lidar['grab_ts'] = gi_ts * 1e-6
    lidar['gi'] = gi
    orig_lidar = lidar.apply(deepcopy)
    wpoints_main = apply_rt(wpoints, l2main)
    not_in_box = np.ones(wpoints.shape[0], dtype=bool)
    if hmask is not None:
        for i in xrange(3):
            not_in_box = not_in_box & (wpoints_main[:, i] > hmask[i, 0]) & (wpoints_main[:, i] < hmask[i, 1])
        not_in_box = ~not_in_box

    wpoints = wpoints[not_in_box]
    lidar = lidar[not_in_box]
    valid_ids = np.arange(len(lidar))
    in_cam_mask_ids = []
    in_cam_ids = []
    for camera in masks_cams:
        valid_ids_cam = np.array(valid_ids)
        l2camera = clip.lidar_data.get_l2c(cam=camera)
        wpoints_camera = apply_rt(wpoints, l2camera)
        zind = wpoints_camera[:, 2] > 0
        valid_ids_cam = valid_ids_cam[zind]
        lidar_camera = lidar[zind].apply(deepcopy)
        wpoints_camera = wpoints_camera[zind]
        xs, ys = simpleProject2image(clip, wpoints_camera, cam=camera, level=level)
        valid = ~np.isnan(xs)
        if lidar_camera.shape[0] > 0 :
            xs1, ys1 = clip.rel2abs_array(xs[valid], ys[valid], cam=camera, level=level)
            xs1 = np.round(xs1).astype(int)
            ys1 = np.round(ys1).astype(int)
            mask = masks[camera]
            valid1 = (xs1 > 0) & (xs1 < mask.shape[1]) & (ys1 > 0) & (ys1 < mask.shape[0])
            valid[valid] = valid1
            valid_ids_cam = valid_ids_cam[valid]
            in_cam_mask_ids.append(valid_ids_cam[mask[ys1[valid1], xs1[valid1]] == 0])
            in_cam_ids.append(valid_ids_cam)

    if len(in_cam_mask_ids) == 0:
        return None, None

    not_in_any_mask = np.ones((len(lidar),), dtype=np.bool)
    not_in_any_mask[np.unique(np.concatenate(in_cam_mask_ids))] = False
    in_any_cam = np.zeros((len(lidar),), dtype=np.bool)
    in_any_cam[np.unique(np.concatenate(in_cam_ids))] = True
    filterd_lidar = lidar[np.logical_and(not_in_any_mask, in_any_cam)]

    moving_lidar = lidar[np.logical_not(np.logical_and(not_in_any_mask, in_any_cam))]
    moving_clusters = np.unique(moving_lidar[moving_lidar.semantic == 0].cluster.values).tolist()
    for cluster in moving_clusters:
        moving_percentile = (1.*len(moving_lidar[moving_lidar.cluster == cluster]))/len(lidar[lidar.cluster == cluster])
        if moving_percentile < 0.5:
            moving_clusters.remove(cluster)
    if len(moving_clusters) > 0:
        moving_points_ids = []
        for moving_cluster in moving_clusters:
            moving_points_ids.append(np.where((filterd_lidar.cluster == moving_cluster).values)[0])
        moving_points_ids = np.unique(np.concatenate(moving_points_ids))
        static_ids = np.ones(len(filterd_lidar), dtype=np.bool)
        static_ids[moving_points_ids] = False

        filterd_lidar = filterd_lidar[static_ids]

    return orig_lidar, filterd_lidar

def get_cluster_mask(points, origin, height=1080, width=1280):
    pnts = np.c_[points[:, 0] + origin[0], points[:, 1] + origin[1]]
    hull = ConvexHull(pnts)
    arr = np.c_[pnts[hull.vertices, 0], pnts[hull.vertices, 1]]
    verts_lst = np.round(arr).astype(np.int32).flatten().tolist()
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(verts_lst, outline=1, fill=1)
    mask = np.array(img)
    return mask


def clean_lidar2(points, fwpoints, clusters, target_im_sz, target_origin, scale, plane_points=None):
    INF_NUM = 1000
    NON_CLUST_NUM = -2

    r_height = target_im_sz[0]
    r_width = target_im_sz[1]
    x = np.arange(0, r_width)
    y = np.arange(0, r_height)
    cluster_ids = list(set(clusters[:, 1]))
    cluster_im = np.ones((r_height, r_width)) * NON_CLUST_NUM
    lidar_im = np.ones((r_height, r_width)) * INF_NUM
    full_interp_lidar_im = np.ones((r_height, r_width)) * INF_NUM
    for i, cluster in enumerate(cluster_ids):
        if cluster == -1:
            continue
        cluster_mask = clusters[:, 1] == cluster
        if np.sum(cluster_mask) < 20:
            continue
            # if clusters[cluster_mask, 0][0] != Semantic.NON_GROUND.value:
            # continue
        cluster_zs = fwpoints[cluster_mask, 2]
        cluster_pnts = points[cluster_mask]
        if plane_points is not None:
            cluster_plane_pnts = plane_points[cluster_mask]
            cluster_plane_pnts = cluster_plane_pnts[np.sum(cluster_plane_pnts, axis=1) != 0]
            mask = get_cluster_mask(np.r_[cluster_pnts, cluster_plane_pnts], target_origin,
                                    height=r_height, width=r_width)
        else:
            mask = get_cluster_mask(cluster_pnts, target_origin, height=r_height, width=r_width)
        if np.sum(mask) < 1000:
            continue
        temp_cluster_im = mask * cluster.astype(np.int32)
        temp_lidar_im, _ = z_buffer(cluster_pnts, cluster_zs, target_im_sz, target_origin, scale)
        temp_lidar_im[temp_lidar_im == 0] = np.nan
        if np.sum(~np.isnan(temp_lidar_im)) < 4:
            continue
        temp_arr = np.ma.masked_invalid(temp_lidar_im)
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~temp_arr.mask]
        y1 = yy[~temp_arr.mask]
        newarr = temp_arr[~temp_arr.mask]
        x1, y1 = prevent_colinearity(x1, y1, mask)
        interp_lidar_im = griddata((x1, y1), newarr.ravel(), (xx, yy), method='cubic')
        interp_lidar_im[~mask] = INF_NUM
        replace_mask = interp_lidar_im < full_interp_lidar_im
        full_interp_lidar_im[replace_mask] = interp_lidar_im[replace_mask]
        cluster_im[np.logical_and(replace_mask, ~np.isnan(temp_lidar_im))] = \
            temp_cluster_im[np.logical_and(replace_mask, ~np.isnan(temp_lidar_im))]
        lidar_im[replace_mask] = temp_lidar_im[replace_mask]
    lidar_im[np.logical_or(lidar_im == INF_NUM, np.isnan(lidar_im))] = 0.
    cluster_im = cluster_im.astype(np.int32)

    return lidar_im, cluster_im


'''

from shapely import geometry
from shapely.ops import cascaded_union, polygonize
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay
import numpy as np
import math


def alpha_shape(points, alpha):
    tri = Delaunay(points)
    triangles = points[tri.vertices]
    a = ((triangles[:, 0, 0] - triangles[:, 1, 0]) ** 2 + (triangles[:, 0, 1] - triangles[:, 1, 1]) ** 2) ** 0.5
    b = ((triangles[:, 1, 0] - triangles[:, 2, 0]) ** 2 + (triangles[:, 1, 1] - triangles[:, 2, 1]) ** 2) ** 0.5
    c = ((triangles[:, 2, 0] - triangles[:, 0, 0]) ** 2 + (triangles[:, 2, 1] - triangles[:, 0, 1]) ** 2) ** 0.5
    s = (a + b + c) / 2.0
    areas = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:, (0, 1)]
    edge2 = filtered[:, (1, 2)]
    edge3 = filtered[:, (2, 0)]
    edge_points = np.unique(np.concatenate((edge1, edge2, edge3)), axis=0).tolist()
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points


def get_cluster_mask(points, origin, height=1080, width=1280, alpha=.005, buffer_px=5):
    pnts = np.c_[points[:, 0] + origin[0], points[:, 1] + origin[1]]
    concave_hull, edge_points = alpha_shape(pnts, alpha=alpha)
    if concave_hull.type != 'Polygon':
        print "MultiPolygon found"
        print len(points)
        print len(concave_hull.geoms)
        while concave_hull.type != 'Polygon':
            alpha -= .001
            if alpha == 0.:
                concave_hull = concave_hull[0]
                break
            concave_hull, edge_points = alpha_shape(pnts, alpha=alpha)
    concave_hull = concave_hull.buffer(buffer_px)
    ext_coords = concave_hull.exterior.coords.xy
    ext_coords = np.c_[np.array(ext_coords[0]), np.array(ext_coords[1])]
    verts_lst = np.round(ext_coords).astype(np.int32).flatten().tolist()
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(verts_lst, outline=1, fill=1)
    mask = np.array(img)
    return mask

'''

class LidarProcessor(object):

    def __init__(self, clip, clip_name, gtem, vdpd_masks, camera_name, lidar_3d_box=np.array([[-1, 1], [-2, 2], [-3.5, 1.7]]), lidar_reader=None):

        from dcorrect.distortionCorrection import DistortionCorrection
        self.clip = clip
        self.dc = DistortionCorrection(clip_path(clip.clip_name), camera=camera_name, max_rad=2000, max_rad_inv=30000,
                                       mest_bitexact=False)
        self.gtem = gtem
        self.vdpd_masks = vdpd_masks
        self.camera_name = camera_name

        self.lidar_3d_box = lidar_3d_box
        try:
            self.lidar_3d_box[2, 0] = -(self.clip.conf_files('main')['camera']['main']['distBackBump'] + 0.1)
            self.lidar_3d_box[2, 1] = self.clip.conf_files('main')['camera']['main']['bumperDist'] + 0.1
        except:
            pass

        self.level_orig = -2
        self.im_sz_orig, self.focal_orig, self.origin_orig = get_distorted_geometry(clip, camera_name, self.level_orig)
        self.im_sz_orig_cams, self.focal_orig_cams, self.origin_orig_cams = {}, {}, {}
        self.lidar_reader = lidar_reader
        self.l2main = get_l2main(clip_name)
        self.l2c = self.clip.get_transformation_matrix('main', self.camera_name).dot(self.l2main)
        self.xx, self.yy = None, None

    def get_lidar_range(self, gi, lidar_range):
        # fix lidar_range so that gis are within gtem gis
        gis = lidar_range + gi
        if self.gtem is not None:
            first_gi = self.gtem['gis'][1]  # self.clip.ts_map.all_gis['main'][0]
            last_gi = self.gtem['gis'][-1]  # self.clip.ts_map.all_gis['main'][-1]
        else:
            first_gi = self.clip.ts_map.all_gis['main'][0]
            last_gi = self.clip.ts_map.all_gis['main'][-1]
        gis = gis[gis >= first_gi]
        gis = np.unique(gis[gis <= last_gi])
        return gis - gi

    def get_raw_lidar(self, gis, get_missing_lidar=False):
        """
        Get Lidar points for list of `gi`s

        :param gis: list of ints
        :param get_missing_lidar: boolean, True if want to add Lidar points with no reflected intensity.
        :return ptss:
        :return P_main: 3D lidar points in `main` camera CS
        :rtype: (n_points, 3) ndarray
        """

        P_main = np.zeros((0, 3))
        ptss = np.zeros((0,), dtype=np.float64)
        l2main = self.l2main
        for i, gi_ in enumerate(gis):
            lidar = self.clip.lidar_data.get_lidar_data(gi_, cam=self.camera_name)
            if get_missing_lidar:
                lidar, _ = add_missing_lidar(lidar, r_missing=10000)
            if lidar is None:
                continue
            P_main_ = apply_rt(lidar[['X', 'Y', 'Z']].values, l2main)
            deltas = lidar['delta']
            gi_ts = self.clip.gi_to_ts(gi_, cam=self.camera_name) * 1e-6
            ptss_ = deltas * 1e-6 + gi_ts

            in_box = points_in_box(P_main_, self.lidar_3d_box)
            P_main_ = P_main_[~in_box]
            ptss_ = ptss_[~in_box]
            P_main = np.r_[P_main, P_main_]
            ptss = np.r_[ptss, ptss_]
            if i % 10 == 0:
                gc.collect()  # These may be
        return P_main, ptss

    def get_lidar_image(self, gi, lidar_range, im_sz, focal, origin, lidar, rectified):
        """
        Return a depth image of the Lidar points, projected to the camera defined for the parent`LidarProcessor` object.

        :param gi: int
        :param lidar_range: array of the extreme gi's values (ints) around the current gi [min_gi, max_gi]
        :param im_sz: shape of image; sequence of ints
        :param focal: float; focal length in meter
        :param origin: (x,y) ; sequence of ints
        :param lidar: dictionary of parameters/options.
        :param rectified:
        :param get_missing_lidar: boolean. True if want to include unreflected Lidar points. (filtered in ViewGenerator by segmentation mask)
        :return:
        :rtype:
        """
        # temporary: if 'get_missing_lidar' flag doesn't exist:
        if 'get_missing_lidar' not in lidar.keys():
            lidar['get_missing_lidar'] = 0
        mask_name, unrolled, filtered_points, occlusion_params, filter_params, run, get_missing_lidar = \
                                        lidar['mask_name'], lidar['unrolled'], lidar['filtered_points'], \
                                        lidar['occlusion_params'], lidar['filter_params'], lidar['run'], \
                                        lidar['get_missing_lidar']


        lidar_range = self.get_lidar_range(gi, lidar_range)

        if self.lidar_reader is not None:
            if filtered_points:
                wpoints_main, clusters, ptss, plane_wpnts_main_objects = self.lidar_reader.get_cluster_lidar(self.clip, gi,
                                                                                                             lidar_range,
                                                                                                             cam=self.camera_name,
                                                                                                             plane=True)
            else:
                wpoints, ptss = self.lidar_reader.get_lidar(self.clip, gi, lidar_range, cam=self.camera_name,
                                                            filtered=False)
                wpoints_main = apply_rt(wpoints, self.l2main)
                clusters = np.zeros((wpoints_main.shape[0], 2), dtype=np.int32)
                plane_wpnts_main_objects = np.zeros((np.sum(clusters[:, 0] == Semantic.NON_GROUND.value), 3))

        else:
            wpoints_main, ptss = self.get_raw_lidar(gi+lidar_range, get_missing_lidar=get_missing_lidar)
            clusters = np.zeros((wpoints_main.shape[0], 2), dtype=np.int32)
            plane_wpnts_main_objects = np.zeros((np.sum(clusters[:,0] == Semantic.NON_GROUND.value),3))

        # Fill in plane_pnts_main with blanks so that it aligns with the other data
        # wpoints_main, clusters, ptss, plane_pnts_main should all be the same length
        plane_wpnts_main = np.zeros_like(wpoints_main)
        plane_wpnts_main[clusters[:,0] == Semantic.NON_GROUND.value] = plane_wpnts_main_objects

        if len(ptss) == 0:
            raise ValueError("No lidar points retrieved")

        main2c = self.clip.get_transformation_matrix('main', self.camera_name)

        # Throw out all plane points from points above main camera's Y origin
        plane_wpnts_main[wpoints_main[:,1] > 0.,:] = 0.

        wpoints_cam = apply_rt(wpoints_main, main2c)
        plane_pnts_cam = apply_rt(plane_wpnts_main, main2c)

        if unrolled:
            if self.gtem is None:
                # Can't unroll anything without EM
                return None, None
            points, valid, fwpoints, _ = cast_lidar_on_img(self.clip, wpoints_cam, self.gtem['params'], gi,
                                                           camera_name=self.camera_name, rectified=rectified,
                                                           level=self.level_orig, ptss=ptss, rounded=False,
                                                           l2c=False, dc=self.dc)
            clusters = clusters[valid]
            plane_pnts_cam = plane_pnts_cam[valid]
            ptss = ptss[valid]

            blank_plane_mask = np.sum(plane_pnts_cam, axis=1) == 0
            points_plane, plane_valid, _, _ = cast_lidar_on_img(self.clip, plane_pnts_cam, self.gtem['params'], gi,
                                                                camera_name=self.camera_name, rectified=rectified,
                                                                level=self.level_orig, ptss=ptss, rounded=False,
                                                                l2c=False)
            points_plane_full = np.zeros((plane_valid.shape[0], 2), dtype=np.float32)
            points_plane_full[np.logical_or(blank_plane_mask, ~plane_valid), :] = 0.
            points_plane = points_plane_full

        else:
            points = self.focal_orig * (wpoints_cam[:, :-1] / wpoints_cam[:, [-1]])
            if not rectified:
                points[:, 0], points[:, 1] = self.clip.unrectify(points[:, 0], points[:, 1], cam=self.camera_name,
                                                                level=self.level_orig)
                valid = ~np.isnan(points[:, 0])
            else:
                valid = ~np.isnan(self.clip.unrectify(points[:, 0], points[:, 1], cam=self.camera_name,
                                                       level=self.level_orig)[0])

            valid = np.logical_and(valid, wpoints_cam[:, 2] > 0.)
            points, fwpoints = points[valid], wpoints_cam[valid]

            clusters = clusters[valid]
            points_plane = plane_pnts_cam[valid]

        if "safety" in run:
            reduction = self.tough_frame_check(points, fwpoints, clusters, im_sz, origin, focal/self.focal_orig,
                                                occlusion_params, points_plane=points_plane)

        vis_mask = np.ones_like(clusters[:,0], dtype=np.bool)
        img_lidar, img_clusters = z_buffer(points, fwpoints[:,2], im_sz, origin, focal / self.focal_orig,
                                           clusters=clusters)

        if "vis_est" in run:
            # remove occluded lidar points using either the 'main' or 'corners' heuristic
            if occlusion_params['heuristic_name'] == 'main':
                _, vis_mask = visibility_estimation(points, fwpoints[:, 2], **occlusion_params['vis_est_kwargs'])
            elif occlusion_params['heuristic_name'] == 'corners':
                img_lidar, temp_mask = z_buffer(points, fwpoints[:, 2], im_sz, origin, focal / self.focal_orig)
                temp_mask = np.logical_and(temp_mask, fwpoints[:,2] > 0)
                _, vis_mask = visibility_estimation(points[temp_mask], fwpoints[temp_mask, 2], **occlusion_params['vis_est_kwargs'])
                temp_mask[temp_mask == True] = vis_mask
                vis_mask = temp_mask

            # Remove points not in vis_mask
            points = points[vis_mask]
            fwpoints = fwpoints[vis_mask]
            clusters = clusters[vis_mask]
            points_plane = points_plane[vis_mask]

            img_lidar, img_clusters = z_buffer(points, fwpoints[:,2], im_sz, origin, focal / self.focal_orig,
                                               clusters=clusters)

        if "cluster" in run:
            img_lidar, img_clusters = clean_lidar2(points, fwpoints, clusters, im_sz, origin, focal/self.focal_orig,
                                                   plane_points=points_plane)

        if "safety" in run:
            aggressive_threshold1 = .1
            aggressive_threshold2 = .2
            if self.camera_name not in ['rearCornerRight', 'frontCornerRight']:
                aggressive_threshold1 += .15
                aggressive_threshold2 += .15
            if reduction > aggressive_threshold1:
                aggressive_occlusion_params = {}
                aggressive_occlusion_params['vis_est_kwargs'] = {}
                aggressive_occlusion_params['vis_est_kwargs']['k'] = 500
                aggressive_occlusion_params['vis_est_kwargs']['upper_bound'] = 200
                aggressive_occlusion_params['vis_est_kwargs']['thresh'] = occlusion_params['vis_est_kwargs']['thresh']
                img_lidar, img_clusters = visibility_estimation_imgs(img_lidar, aggressive_occlusion_params, img_clusters)
            if reduction > aggressive_threshold2:
                img_lidar, img_clusters = visibility_estimation_imgs(img_lidar, aggressive_occlusion_params, img_clusters)

        if self.vdpd_masks is not None:
            mask = self.vdpd_masks.\
                get_mask_image(gi, self.camera_name, im_sz, focal, origin, mask_name, rectified=rectified)
            img_lidar = img_lidar * (1 - mask)
            img_clusters = img_clusters * (1 - mask)

        if "filter" in run:
            img_lidar = self.filter_lidar_image(img_lidar, **filter_params)

        return img_lidar.astype(np.float32), img_clusters.astype(np.int32)


    def filter_lidar_image(self, im, factor=2, rad=50, thresh=0.7):

        im_lidar = 1 * im
        im_lidar[im_lidar == 0.] = -1
        steps = np.linspace(-1, 100, 102)
        im_lidar_dig = np.digitize(im_lidar, steps)
        if self.xx is None:
            self.xx, self.yy = np.meshgrid(np.arange(im_lidar_dig.shape[1]), np.arange(im_lidar_dig.shape[0]))
        im_lidar_dig_tensor = np.zeros((im_lidar_dig.shape[0], im_lidar_dig.shape[1], len(steps) - 2), dtype=np.int32)
        for i in np.arange(2, len(steps)):
            im_lidar_dig_tensor[:,:,i-2] = im_lidar_dig == i
        # im_lidar_dig_tensor[np.arange(im_lidar_dig.shape[0]), np.arange(im_lidar_dig.shape[1]), np.maximum(np.minimum(im_lidar_dig - 2, im_lidar_dig_tensor.shape[2] - 1), 0)] = 1

        im_lidar_dig_integral = np.cumsum(np.cumsum(im_lidar_dig_tensor, 0), 1)
        x1 = np.maximum(np.minimum(self.xx + rad, im_lidar_dig.shape[1] - 1), 0)
        x2 = np.maximum(np.minimum(self.xx - rad, im_lidar_dig.shape[1] - 1), 0)
        y1 = np.maximum(np.minimum(self.yy + rad, im_lidar_dig.shape[0] - 1), 0)
        y2 = np.maximum(np.minimum(self.yy - rad, im_lidar_dig.shape[0] - 1), 0)
        windows = (im_lidar_dig_integral[y1, x1, :] +
                   im_lidar_dig_integral[y2, x2, :] -
                   im_lidar_dig_integral[y1, x2, :] -
                   im_lidar_dig_integral[y2, x1, :])
        windows_cumsum = np.cumsum(windows, 2)
        y, x = np.where(im_lidar_dig - 2 > 0)
        ids = np.where(
            windows_cumsum[y, x, ((im_lidar_dig[y, x] - 3) / factor).astype(np.int32)] > thresh * windows_cumsum[y, x, -1])[0]
        im_lidar[y[ids], x[ids]] = 0
        im_lidar[im_lidar == -1] = 0
        return im_lidar

    def tough_frame_check(self, points, fwpoints, clusters, im_sz, origin, scale, occlusion_params, points_plane=None):
        # Check if this frame is going to produce vis est problem
        img_lidar, img_clusters = clean_lidar2(points, fwpoints, clusters, im_sz, origin, scale,
                                               plane_points=points_plane)

        num_lidar_pnts = np.sum(img_lidar != 0)

        img_lidar, img_clusters = visibility_estimation_imgs(img_lidar, occlusion_params=occlusion_params,
                                                             cluster_im=img_clusters)

        filtered_num_lidar_pnts = np.sum(img_lidar != 0)
        reduction = (1. * num_lidar_pnts - filtered_num_lidar_pnts) / num_lidar_pnts

        return reduction
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.ndimage import binary_erosion
from stereo.common.gt_packages_wrapper import FoFutils
from stereo.data.label_utils import seg_mod_init, segment_image_from_clip, vis_seg




def rad2deg(a):
    if type(a) is tuple:
        theta = tuple(rad2deg(b) for b in a)
    else:
        theta = a * 180. / np.pi
    return theta


def deg2rad(theta):
    if type(theta) is tuple:
        a = tuple(deg2rad(t) for t in theta)
    else:
        a = theta * np.pi / 180.
    return a


def cart2sph(X, Y, Z):
    """ Cartesian to spherical coordinates. Angles in radians."""
    rho = np.sqrt(X ** 2 + Y ** 2)
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    theta = np.arctan2(rho, Z)
    phi = np.arctan2(Y, X)
    return r, theta, phi


def sph2cart(r, theta, phi):
    """ Spherical to cartesian coordinates. Angles in radians."""
    Z = r * np.cos(theta)
    rho = r * np.sin(theta)
    X = rho * np.cos(phi)
    Y = rho * np.sin(phi)
    return X, Y, Z


def lidar_from_clip(clip, gi):
    """
    Get lidar points (in Lidar CS) and delta's from clip
    """
    clip.init_lidar()
    lidar_pd = clip.lidar_data.get_lidar_data(gi) # Lidar Pandas object
    lidar_wpoints = lidar_pd[['X', 'Y', 'Z']].to_numpy().T
    deltas = lidar_pd['delta'].to_numpy()
    return lidar_wpoints, deltas


def nearby_bins(angle_vec):
    """
    Determining bins of theta. Same indices `ind_group` for nearby values of `angle_vec`
    """
    vec_round = np.round(rad2deg(angle_vec) * 10) / 10
    # vec_en_round = np.round(rad2deg(angle_vec) * 10)
    unique_vals = np.unique(vec_round)
    ind_group = np.zeros(len(angle_vec), dtype=int)
    for i, val in enumerate(unique_vals):
        ind_group[vec_round == val] = i + 1
    return ind_group


def group_to_dict(arrays_tuple, group_inds):
    """
    Arranging in dictionaries according to group indices `group_inds`.
    """
    dicts_tuple = ()
    for j in range(len(arrays_tuple)):
        dict_temp = {}
        for i in np.unique(group_inds).astype(int):
            dict_temp[i] = arrays_tuple[j][group_inds == i]
        dicts_tuple += (dict_temp,)

    return dicts_tuple


def sep_theta(theta_vec, phi_vec, delta_vec):
    """
    Separating data into groups of distinct values of `theta`.
    Output in dictionaries, units of degrees, with the assignment `theta_gr` of each array element to its group
    """
    theta_deg = rad2deg(theta_vec)
    phi_deg = rad2deg(phi_vec)
    theta_gr = nearby_bins(theta_deg)
    theta_deg_dict, phi_deg_dict, delta_dict = group_to_dict((theta_deg, phi_deg, delta_vec),
                                                     theta_gr)  # arrange in dicts according to group
    return theta_deg_dict, phi_deg_dict, delta_dict, theta_gr


def theta_medians(theta_dict):
    """ Median of `theta` for group of `theta` values / each key of dict"""
    n_theta = len(theta_dict.keys())
    theta_meds = np.zeros(n_theta)
    for i in range(n_theta):
        theta_meds[i] = np.median(theta_dict[i + 1])
    return theta_meds


def scan_jump(dict, sort=False):
    """
    Finding mean of medians of each key (/"row") of values in `dict`
    Used for find scan jumps (angle/time)
    'sort' should help when original data is not sorted
    Consider implementing a method more robust to sparse data.
    """
    jumps_meds = np.zeros((len(dict.keys()),))
    for c, i in enumerate(dict.keys()):
        if sort:
            jumps_meds[c] = np.median(np.diff(np.sort(dict[i])))
        else:
            jumps_meds[c] = np.median(np.diff(dict[i]))
    jump = jumps_meds.mean()
    return jump


def edges_scan(dict, scan_jump):
    """
    Find edge points on each key ("row") to extrapolate to.
    For safety margin, extending up to one `scan_jump` from edge values.
    """
    scan_min = np.concatenate(dict.values()).min()
    scan_max = np.concatenate(dict.values()).max()
    if scan_jump > 0:
        scan_start = scan_min + scan_jump
        scan_end = scan_max - scan_jump
    elif scan_jump < 0:
        scan_start = scan_max + scan_jump
        scan_end = scan_min - scan_jump
    return scan_start, scan_end


def arrange_on_grid(vec, jump, start, end):
    """
    Arranging grid of scan times `delta` (per "row" of `theta`)
    Output: `inds` of existing (lidar) points on the grid, `start` value (updated if vec[0] comes before),
            number of grid points `n_grid` (useful for determining end of grid)
    """
    vec_norm = (vec - start) / jump
    norm_diff = np.diff(np.r_[0, vec_norm])
    ind_diff = np.round(norm_diff)
    inds = np.cumsum(ind_diff).astype(int)
    if inds[0] < 0:
        start -= (-inds[
            0]) * jump  # written less efficient for readability: if turns out that `vec` has a point before `start`, decrease `start` to match this starting point
        inds -= inds[0]
    ind_end = np.round((end - start) / jump).astype(int)
    ind_end = np.max([ind_end, inds[-1]]).astype(int)
    n_grid = ind_end + 1
    return inds, start, n_grid



def missing_phis(phi_dict, delta_dict, theta_meds):
    """ Find "missing" `phi`s (were in scan but no returned signal)"""
    phi_missing = np.array([])
    theta_missing = np.array([])
    delta_missing = np.array([])
    n_theta = len(phi_dict.keys())
    phi_jump = scan_jump(phi_dict)
    delta_jump = scan_jump(delta_dict)
    delta_start, delta_end = edges_scan(delta_dict, delta_jump)  # edge values up to which extrapolate missing points
    for i in range(n_theta):
        inds_temp, delta_start_temp, n_grid_temp = arrange_on_grid(delta_dict[i + 1], delta_jump, delta_start,
                                                                   delta_end)
        full_temp = np.arange(n_grid_temp)  # all the indices, including the ones to be extrapolated
        ind_miss_temp = full_temp[~np.isin(full_temp, inds_temp)]
        phi0_temp = phi_dict[i + 1][0] - phi_jump * inds_temp[0]
        miss_phi_temp = phi0_temp + phi_jump * ind_miss_temp
        miss_delta_temp = delta_start_temp + delta_jump * ind_miss_temp
        phi_missing = np.append(phi_missing, miss_phi_temp)
        delta_missing = np.append(delta_missing, miss_delta_temp)
        theta_missing = np.append(theta_missing, theta_meds[i] * np.ones(len(miss_phi_temp)))
    return phi_missing, theta_missing, delta_missing


def find_missing_lidar(wpoints, deltas, r_missing=10000):
    """
    Given lidar points from one grab_index, find the missing points.
    Need to set a (far) distance `r_missing` to missing points.
    All angles in radians here.

    :param wpoints: (3, n_points) 3D Lidar world points in Lidar CS
    :params deltas: (n_points,)
    :return wpoints_missing: (3, n_missing) "missing" Lidar 3D world points, i.e.,
            points where no Lidar was reflected corresponding to `r_missing`, in Lidar CS
    :return delta_missing: (n_missing,)
    :return angles_missing:
    """
    X_vec, Y_vec, Z_vec = wpoints
    delta_vec = deltas.squeeze()
    r_vec, theta_vec, phi_vec = cart2sph(X_vec, Z_vec, Y_vec)
    theta_dict, phi_dict, delta_dict, theta_gr = sep_theta(theta_vec, phi_vec, delta_vec) # dictionaries (outputs of `sep_theta`) in degrees
    theta_meds = theta_medians(theta_dict)
    phi_missing, theta_missing, delta_missing = missing_phis(phi_dict, delta_dict, theta_meds)
    angles_missing = {'phi': phi_missing, 'theta': theta_missing}
    X_missing, Z_missing, Y_missing = sph2cart(r_missing, deg2rad(theta_missing), deg2rad(phi_missing))
    wpoints_missing = np.stack((X_missing, Y_missing, Z_missing))
    return wpoints_missing, delta_missing, angles_missing


def add_missing_lidar(lidar_df, r_missing=10000):
    """
    Add to Lidar dataframe its missing points.
    :param lidar_df: pandas dataframe, from `get_lidar_data`
    :param r_missing: distance from lidar to be assigned to missing points (very far).
    :return:
    """
    wpoints = (lidar_df[['X', 'Y', 'Z']].values).T
    deltas = lidar_df['delta'].values
    rays = lidar_df['ray'].values
    wpoints_missing, delta_add, angles_missing = \
        find_missing_lidar(wpoints, deltas, r_missing=10000)
    delta_add = delta_add.astype(int)
    X_add, Y_add, Z_add = wpoints_missing
    thetas_unique = np.sort( np.unique(angles_missing['theta']))[::-1] # `theta` values in descending order
    theta2ray_dict = {thetas_unique[i]: i for i in range(len(thetas_unique))} # translate between `theta` and ray
    ray_add = np.vectorize(theta2ray_dict.__getitem__)(angles_missing['theta'])
    cluster_add = (-1) * np.ones(X_add.shape)
    semantic_add = (-1) * np.ones(X_add.shape)
    intens_add = np.zeros(X_add.shape) # put zero intensity for missing points
    list_add = zip(X_add, Y_add, Z_add, ray_add, cluster_add, semantic_add, delta_add, intens_add)
    df_add = pd.DataFrame(list_add, columns=lidar_df.columns)
    lidar_df_with_missing = lidar_df.append(df_add, ignore_index=True)
    return lidar_df_with_missing, df_add


def world2im(wpoints, f, origin_pix, only_in_front=True):
    """
    Converting world points from same CS to image points
    :param wpoints:
    :param f: focal length in pixels
    :param origin_pix: (x,y)
    :param only_in_front: boolean; keep only points with positive Z (default True)
    :return image_points: 2D pixels projected to image (2, n_points)
    """
    x_im0 = wpoints[0, :] * f / wpoints[2, :]
    y_im0 = wpoints[1, :] * f / wpoints[2, :]
    xim = x_im0 + origin_pix[0]
    yim = y_im0 + origin_pix[1]
    image_points0 = np.stack([xim, yim])
    cond_pos = (wpoints[2,:] > 0)
    image_points = image_points0[:, cond_pos]
    return image_points, cond_pos


def im2world(cam_points, Z_cam, f, origin_pix):
    """
    Converting points on camera image to 3D world points in same camera CS.
    :param cam_points: (2, n_points)
    :type cam_points: ndarray
    :param Z_cam: (n_points,)
    :type Z_cam: ndarray
    :param f: focal length
    :param origin_pix (x,y) tuple
    :return: `wpoints_cart` (3, n_points), cartesian coordinates
    :rtype: ndarray
    """
    if isinstance(Z_cam,int) or isinstance(Z_cam, float):
        Z_cam = Z_cam* np.ones(cam_points.shape[1])
    x_im, y_im = cam_points
    x_im1 = x_im - origin_pix[0]
    y_im1 = y_im - origin_pix[1]
    X = x_im1 / f * Z_cam
    Y = y_im1 / f * Z_cam
    wpoints_cart = np.stack([X, Y, Z_cam])
    return wpoints_cart

def pinimage(im_points, nx, ny):
    """
    Filter points that are inside image. `nx` and `ny` are the horizontal and vertical number of pixels.
    """
    condx = (im_points[0, :] < nx - 1) * (im_points[0, :] >= 0)
    condy = (im_points[1, :] < ny - 1) * (im_points[1, :] >= 0)
    condboth = condx * condy
    return im_points[:, condboth], condboth


def lidar_transformation_params(clip, gi, cam_name, pyr_level):
    """
    Parameters needed for transformation betwen lidar and cam (RT, focal, origin)
    """
    if isinstance(clip, FoFutils):
        clip2 = clip.get_clip(cam=cam_name, grab_index=gi)
    else:
        clip2 = clip
    clip2.init_lidar()
    Metemp = clip2.get_frame(grab_index=gi, camera_name=cam_name)[0]['pyr'][pyr_level]
    Me_rect = clip2.rectify_image(Metemp, cam=cam_name, level=pyr_level)
    nv_rect, nh_rect = Me_rect.im.shape
    origin_rect = Me_rect.origin()
    focal = clip2.focal_length(camera_name=cam_name, level=pyr_level)
    RT_lid_to_cam = clip2.get_lidar2cam(cam=cam_name)
    return RT_lid_to_cam, focal, origin_rect, nv_rect, nh_rect

def lidar_to_image(wpoints_cart, clip, gi, cam_name, pyr_level):
    """
    Converting world points in Lidar CS to rectified image of camera
    """
    RT_lid_to_cam, focal, origin_rect, nv_rect, nh_rect = lidar_transformation_params(clip, gi, cam_name, pyr_level)
    X_vec, Y_vec, Z_vec = wpoints_cart
    lid_points_inh = np.stack([X_vec, Y_vec, Z_vec, np.ones(X_vec.shape)])
    lidar_to_main_points4 = RT_lid_to_cam.dot(lid_points_inh)
    lid_im_rect, cond_positiveZ = world2im(lidar_to_main_points4[:3, :], focal, origin_rect)
    Z_im = lidar_to_main_points4[2, cond_positiveZ]
    lid_cam, cond_in_image = pinimage(lid_im_rect, nh_rect, nv_rect)
    Z_cam = Z_im[cond_in_image]
    cond_pos_image = cond_positiveZ.copy()
    cond_pos_image[cond_pos_image] = cond_in_image
    return lid_cam, Z_cam, cond_pos_image



def filt_sky_seg(cam_points, seg_sky, erosion=False):
    """
    Filtering sky points according to segmentation
    """
    if erosion:
        # erosion_iters = np.ceil(5 *(2 ** (-1 - pyr_level ) )).astype(int) # 5 iterations for level -1 is arbitrary, seems to complete poles
        seg_sky = binary_erosion(seg_sky, iterations=5)
    inds = np.round(cam_points).astype(int)
    seg_points = seg_sky[inds[1, :], inds[0, :]]
    cond_sky = seg_points
    cam_points_filt = cam_points[:, cond_sky]
    return cam_points_filt, cond_sky

def filter_sky_lidar_image(labels, lidar_image, Zthresh=1000,
                                    erosion=True):
    """
    Remove from `lidar_image` missing points, defined by Z > Zthresh, which are not segmented as sky.
    Currently can segment only with 'deeplab'.
    :param labels: dictionary of segmentation masks. Must have 'deeplab'
    :param lidar_image: Depth image of Lidar points. Pixels without Lidar data have Z=0
    :param Zthresh: Depth threshold for checking whether sky [meters]
    :param erosion: boolean; if True, use erosion to eliminate small regions of wrong sky segmentation
    :return lidar_image_filtered: Depth image of Lidar points, after filtering out "wrong" sky points,
            i.e. missing Lidar points that are not segmented as sky.
    """
    assert ('deeplab' in labels.keys()), "cannot filter sky without 'deeplab'"
    sky_mask = (labels['deeplab']['segmentation_mask'] == 10) # sky label for deeplab is 10. Currently segment only according to Deeplab. If additional nets will be added, should implement their usage.
    if erosion:
        sky_mask = binary_erosion(sky_mask, iterations=5)

    # Define points which are Z>Zthresh and sky_mask=0 as =0.
    # Meaning previous value is preserved only if Z<Zthresh or sky_mask=1:
    sky_condition = np.logical_or(lidar_image < Zthresh, sky_mask)
    lidar_image_filtered = lidar_image * sky_condition
    return lidar_image_filtered



def sky_world_and_image(model, clip, gi, cam_name, pyr_level, wpoints, deltas,
                        r_sky=10000, erosion=False):
    """
    Get sky image (camera) points, sky 3D world points, times of acquisition delta,
    and the sky segmentation mask.
    Function tailored for use in LidarProcessor:get_raw_lidar.
    :param model:
    :param clip:
    :param gi:
    :param cam_name:
    :param pyr_level:
    :param wpoints: 3D Lidar world-points in Lidar CS (3,n)
    :param deltas:
    :param r_sky:
    :param erosion:
    :return sky_cam:
    :return sky_wpoints: 3D sky world-points in Lidar CS
    :return deltas_sky:
    :return sky_seg_mask:
    """
    # plot_XZ(wpoints[0,:], wpoints[2,:]) # to visualize direction of lidar (debug)
    _, sky_seg_mask, _ = segment_image_from_clip(model, clip, gi, cam_name, pyr_level)
    wpoints_missing, deltas_missing, _ = find_missing_lidar(wpoints, deltas, r_sky) # missing in Lidar CS
    missing_cam, _, cond_pos_image = lidar_to_image(wpoints_missing, clip, gi, cam_name, pyr_level)
    sky_cam, cond_sky = filt_sky_seg(missing_cam, sky_seg_mask, erosion)
    wpoints_missing_for_cam = wpoints_missing[:,cond_pos_image] # wpoints still in Lidar CS
    deltas_missing_cam = deltas_missing[cond_pos_image]
    sky_wpoints = wpoints_missing_for_cam[:, cond_sky] # sky_wpoints in Lidar CS
    deltas_sky = deltas_missing_cam[cond_sky]
    return sky_cam, sky_wpoints, deltas_sky, sky_seg_mask


def find_sky_points(model, clip, gi, cam_name, pyr_level, wpoints, deltas, r_sky=10000,
                    erosion=False, visualize=False):
    """
    Find missing points from lidar world points data of one grab_index (wpoints, deltas; Lidar CS),
    generate a sky segmentation mask, and filter out sky "camera" points in image.
    """
    im, sky_seg_mask, seg_map_im = segment_image_from_clip(model, clip, gi, cam_name, pyr_level)
    wpoints_missing, _, _ = find_missing_lidar(wpoints, deltas, r_sky) # wpoints_missing: (3, n_missing)
    lidar_cam, Z_cam, _ = lidar_to_image(wpoints, clip, gi, cam_name, pyr_level)
    missing_cam, _, _ = lidar_to_image(wpoints_missing, clip, gi, cam_name, pyr_level)
    sky_cam, _ = filt_sky_seg(missing_cam, sky_seg_mask, erosion)
    if visualize:
        vis_seg(im, seg_map_im)
        plot_XZ(wpoints_missing[0, :], wpoints_missing[2, :])
        show_im_lidar(im, lidar_cam, Z_cam, missing_cam, sky_cam)
    return sky_cam, missing_cam, lidar_cam, Z_cam, im, sky_seg_mask


def clip_to_image_lidar_sky(clip, gi, cam_name, pyr_level, erosion=False, r_sky=10000,
                            visualize=False):
    """
    FoF clip to image, lidar camera 2D points + depths (Z), "missing" Lidar points
    with no reflected intensity, sky 2D points (on camera image), sky segmentation mask.
    Wrap all functionality in one function.
    :param clip:
    :param gi:
    :param cam_name:
    :param pyr_level:
    :param erosion:
    :param r_sky:
    :return:
    """
    model = seg_mod_init('deeplab')
    lidar_wpoints, deltas = lidar_from_clip(clip, gi)
    lidar_sky_cam, lidar_missing_cam, lidar_image_cam, Z_cam_rect, im, sky_seg_mask = \
        find_sky_points(model, clip, gi, cam_name, pyr_level, lidar_wpoints, deltas,
                        r_sky=r_sky, erosion=erosion, visualize=visualize)

    return im, lidar_image_cam, Z_cam_rect, lidar_missing_cam, lidar_sky_cam, sky_seg_mask

def visualize_angles_grid(angles_lidar, angles_missing=None, angles_sky=None, units='degrees'):
    """
    Plot theta-phi graph. Inputs are tuples of numpy vectors (theta,phi).
    If units='radians', convert to degrees.
    Lidar points (magenta) with 'sky' missing points (yellow) and unlabeled missing points (gray)
    """
    plt.style.use('dark_background')
    if units == 'radians':
        angles_lidar = rad2deg(angles_lidar)
        if angles_missing is not None:
            angles_missing = rad2deg(angles_missing)
        if angles_sky is not None:
            angles_sky = rad2deg(angles_sky)

    plt.figure(figsize=(5, 3))
    plt.scatter(angles_lidar[1], angles_lidar[0], c='m', s=0.5)
    if angles_missing is not None:
        plt.scatter(angles_missing[1], angles_missing[0], c='gray', s=0.5)
    if angles_sky is not None:
        plt.scatter(angles_sky[1], angles_sky[0], c='y', s=0.5)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\theta$')
    plt.tight_layout()
    plt.show()


def plot_XZ(xwpoints, zwpoints):
    """
    Show lidar/missing world points on an X-Z plot.
    """
    # %matplotlib notebook
    fig = plt.figure(figsize=(4, 2))
    plt.plot(xwpoints, zwpoints, '.r', label='', markersize=0.5)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def show_im_lidar(im, lidar_image_cam, Z_cam_rect, lidar_missing_cam=None, lidar_sky_cam=None):
    """
    Show image with lidar points; optionally show missing and/or sky points.
    :param lidar_image_cam: (2, n_lidar)
    """
    plt.figure(figsize=(11, 5))
    plt.imshow(im, origin='lower', cmap='gray')
    if isinstance(Z_cam_rect,int) or isinstance(Z_cam_rect, float):
        Z_cam_rect = Z_cam_rect* np.ones(lidar_image_cam.shape[1])
    sc = plt.scatter(lidar_image_cam[0, :], lidar_image_cam[1, :], c=Z_cam_rect, s=0.5)
    if lidar_missing_cam is not None:
        scm = plt.scatter(lidar_missing_cam[0, :], lidar_missing_cam[1, :], c='r', s=0.5)
    if lidar_sky_cam is not None:
        scms = plt.scatter(lidar_sky_cam[0, :], lidar_sky_cam[1, :], c=[1,0.3,0.3], s=0.5)
    plt.colorbar(sc, shrink=0.6)
    plt.show()



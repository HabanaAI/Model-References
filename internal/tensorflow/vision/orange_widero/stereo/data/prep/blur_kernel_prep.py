import numpy as np
from scipy.ndimage import map_coordinates as interp2
from scipy.interpolate import griddata
from scipy.signal import convolve2d

def interp_lidar(im_lidar, mask_kernel_sz=5):
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

def warp_source2target_by_Z(im_source, origin_source, focal_source,
         RT_target2source, Z_target, target_shape=None,
         origin_target=None, focal_target=None):
    if not origin_target:
        origin_target = origin_source
    if not focal_target:
        focal_target = focal_source
    if not target_shape:
        target_shape = im_source.shape
    assert target_shape[0] == Z_target.shape[0] and target_shape[1] == Z_target.shape[1]
    x,y = np.meshgrid(np.arange(target_shape[1])-origin_target[0], np.arange(target_shape[0])-origin_target[1] )
    X = Z_target*x/focal_target
    Y = Z_target*y/focal_target
    P = np.c_[X.flatten(), Y.flatten(), Z_target.flatten(), np.ones_like(X.flatten())].T
    P = np.matmul(RT_target2source, P)[:3,:]
    xx = focal_source*P[0,:]/P[2,:] + origin_source[0]
    yy = focal_source*P[1,:]/P[2,:] + origin_source[1]
    return interp2(im_source, [yy.ravel(), xx.ravel()], order=3).reshape(target_shape)

def prep_frame(dataSetIndex, frame, view_names, fail_missing_lidar=True, inferenceOnly=False, views=None):
    cntr_cams = ['main', 'frontCornerLeft', 'frontCornerRight', 'rearCornerLeft', 'rearCornerRight', 'rear']

    if views is None:
        views = dataSetIndex.read_views(frame=frame, view_names=view_names)
    ims_dict = {}
    mask_dict = {}
    cntr_views = {}
    srnd_views = {}
    for cam in cntr_cams:
        cntr_views[cam] = cam+'_to_'+cam
        srnd_views[cam] = []
    for cam in cntr_cams:
        for view in view_names:
            if view.endswith('_to_'+cam) and view not in cntr_views.values():
                srnd_views[cam].append(view)

    for cam in cntr_cams:
        cntr_view = cntr_views[cam]
        ims_dict[cntr_view] = np.expand_dims(views[cntr_view]['image']/ 255., 0).astype(np.float32)
        if not inferenceOnly:
            mask = 1 - np.logical_or(views[cntr_view]['masks']['PD_approved'],
                                     views[cntr_view]['masks']['VD_moving'])
            Z, im_lidar_interp_mask = interp_lidar(views[cntr_view]['lidars']['medium'])
            mask = mask*im_lidar_interp_mask
            mask_dict[cntr_view+'_mask'] = np.expand_dims(1-mask, 0).astype(np.float32)
            for srnd in srnd_views[cam]:
                RT_cntr_to_srnd = np.matmul(np.linalg.inv(views[srnd]['RT_view_to_main']),
                                            views[cntr_view]['RT_view_to_main'])
                srnd_w = warp_source2target_by_Z(views[srnd]['image']/ 255.,
                              views[srnd]['origin'],
                              views[srnd]['focal'],
                              RT_cntr_to_srnd, Z) * mask
                ims_dict[srnd] = np.expand_dims(srnd_w, 0).astype(np.float32)
    example = {}
    example.update(ims_dict)
    example.update(mask_dict)
    return [example]



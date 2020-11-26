import numpy as np

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import OrderedDict
from scipy.ndimage import map_coordinates as interp2
from PIL import Image, ImageDraw


def get_cmap(vmin=0, vmax=1, cmap='gist_rainbow'):
    modified_cmap = cm.get_cmap(cmap)
    modified_cmap.set_under('k')
    modified_cmap.set_bad('k')
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    m = cm.ScalarMappable(norm=norm, cmap=modified_cmap)
    return m


def depth_visualization(im_lidar, cmap='jet', inverse=True, alpha=False, clim=None):
    im_depth = im_lidar.copy()
    im_depth[im_depth <= 0] = np.nan
    if inverse:
        im_depth = im_depth ** -1
    m = get_cmap(vmin=np.nanmin(im_depth) if clim is None or clim[0] is None else clim[0],
                 vmax=np.nanmax(im_depth) if clim is None or clim[1] is None else clim[1],
                 cmap=cmap)
    if alpha:
        return m.to_rgba(im_depth)
    else:
        return m.to_rgba(im_depth)[..., :3]
    

def blend_depth(im_cam, im_lidar, alpha=0.5, cmap='jet', inverse=True, clim=None):
    im_cam_rgb = im_cam[..., np.newaxis]*(1, 1, 1)
    im_depth = depth_visualization(im_lidar, cmap=cmap, inverse=inverse, alpha=False, clim=clim)
    depth_ind = im_lidar > 0
    im_cam_rgb[depth_ind] = (alpha * im_depth[depth_ind]) + ((1 - alpha) * im_cam_rgb[depth_ind])
    return im_cam_rgb


def blend_mask(im_cam, im_mask, alpha=0.2, color=(1, 0, 0)):
    im_cam_rgb = im_cam[..., np.newaxis]*(1, 1, 1)
    im_mask_rgb = im_mask[..., np.newaxis]*color
    mask_ind = im_mask > 0
    im_cam_rgb[mask_ind] = (alpha * im_mask_rgb[mask_ind]) + ((1 - alpha) * im_cam_rgb[mask_ind])
    return im_cam_rgb


def blend_imgs(im_cam, im_lidar=None, im_mask=None, alpha_lidar=0.9, alpha_mask=0.2,
                        cmap='jet', inverse=True, color_mask=(1, 0, 0)):
    im_cam_rgb = im_cam[..., np.newaxis]*(1, 1, 1)
    if im_lidar is not None:
        im_depth = depth_visualization(im_lidar, cmap=cmap, inverse=inverse, alpha=False)
        depth_ind = im_lidar > 0
        im_cam_rgb[depth_ind] = (alpha_lidar * im_depth[depth_ind]) + ((1 - alpha_lidar) * im_cam_rgb[depth_ind])
    if im_mask is not None:
        im_mask_rgb = im_mask[..., np.newaxis]*color_mask
        mask_ind = im_mask > 0
        im_cam_rgb[mask_ind] = (alpha_mask * im_mask_rgb[mask_ind]) + ((1 - alpha_mask) * im_cam_rgb[mask_ind])
    return im_cam_rgb


def points2d_to_3d(x, y, z, origin, focal, im_shape):
    # y = im_shape[0] - y  # invert y axis
    X = (x - origin[0]) * z / focal[0]
    Y = (y - origin[1]) * z / focal[1]
    # np.c_[X, z, -Y]
    return np.c_[X, Y, z]


def points2d_from_image(im_depth, depth_mask=None):
    valid = im_depth > 0
    if depth_mask is not None:
        valid = valid & (depth_mask > 0)
    y, x = np.where(valid)
    z = im_depth[y, x]
    return x, y, z


def points3d_from_image(im_depth, im_grayscale, origin, focal, depth_mask=None):
    focal = (focal, focal) if np.array(focal).size == 1 else focal
    x, y, z = points2d_from_image(im_depth, depth_mask)
    pcd = points2d_to_3d(x, y, z, origin, focal, im_depth.shape)
    grayscale = im_grayscale[y, x] if im_grayscale is not None else None
    return pcd, grayscale


def view_pptk3d(pcd, grayscale, host_box=None, point_size=0.02, fix_coords_type=0,
                outlier_removal=False, outlier_removal_points=3, outlier_removal_radius=0.3,
                viewer=None, lookat=(0, 0, 0), r=15, phi=3 * np.pi / 2, theta=0, save_to_tmp=False):
    import pptk
    from stereo.common.visualization.pptk_wrapper import ViewWrapper

    if host_box is not None:
        x, y, z = np.meshgrid(np.arange(host_box[0, 0], host_box[0, 1], 0.1),
                              np.arange(host_box[1, 0], host_box[1, 1], 0.1),
                              np.arange(host_box[2, 0], host_box[2, 1], 0.1))
        box = np.c_[x.flatten(), y.flatten(), z.flatten()]
        box_color = np.zeros_like(x.flatten())
        pcd = np.concatenate([pcd, box])
        grayscale = np.concatenate([grayscale, box_color])
    if fix_coords_type==0:
        pcd = np.c_[pcd[:, 0], pcd[:, 2], -pcd[:, 1]]
    else:
        pcd = np.c_[pcd[:, 0], pcd[:, 2], pcd[:, 1]]

    if outlier_removal:
        import open3d as o3d

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
        cl, ind = o3d.geometry.radius_outlier_removal(pcd_o3d, nb_points=outlier_removal_points,
                                                      radius=outlier_removal_radius)
        pcd = pcd[ind]
        grayscale = grayscale[ind]
    if save_to_tmp:
        import os
        temp_pcd_path = os.path.join(os.environ['HOME'], 'temp', 'pcd.npz')
        data = {'pcd': pcd, 'grayscale': grayscale}
        np.savez(temp_pcd_path, **data)
        return None

    if not viewer:
        viewer = pptk.viewer(pcd, grayscale)
    else:
        viewer.clear()
        viewer.load(pcd, grayscale)

    viewer.color_map([[0, 0, 0], [1, 1, 1]], scale=[0.0, 1.0])
    viewer.set(point_size=point_size, show_grid=True, lookat=lookat, r=r, phi=phi, theta=theta)
    viewer.set(selected=np.array([], dtype='uint32'))
    viewer.set(show_axis=False)
    viewer.set(floor_level=-1.5)
    return ViewWrapper(viewer, grayscale)


def view_pptk(im_depth, origin, focal, im_cam=None, depth_mask=None, im_mask=None, cmap='gist_rainbow',
              return_pcd=False, im_depth2=None, outlier_removal=True, point_size=0.02):
    import pptk

    x, y, z = points2d_from_image(im_depth, depth_mask)
    pcd = points2d_to_3d(x, y, z, origin, focal, im_depth.shape)

    if outlier_removal:
        import open3d as o3d

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
        cl, ind = o3d.geometry.radius_outlier_removal(pcd_o3d, nb_points=3, radius=0.30)
    else:
        ind = np.arange(pcd.shape[0])

    attr = OrderedDict()
    if im_cam is not None:
        # make 3 channels:
        grayscale = np.c_[(im_cam[y, x],) * 3]
        # add alpha:
        grayscale = np.c_[grayscale, np.ones(grayscale.shape[0])]

        if outlier_removal:
            grayscale_clean = np.ones_like(grayscale) * (0, 0, 0, 0)
            grayscale_clean[ind] = grayscale[ind]
            attr['grayscale clean'] = grayscale_clean
            grayscale_clean = np.ones_like(grayscale) * (1, 0, 0, 1)
            grayscale_clean[ind] = grayscale[ind]
            attr['grayscale clean+outliers'] = grayscale_clean

        attr['grayscale'] = grayscale

    for c, name in zip((z ** -1, x, y, z), ('inv z', 'x', 'y', 'z')):
        attr[name] = get_cmap(vmin=np.nanmin(c[ind]), vmax=np.nanmax(c[ind]), cmap=cmap).to_rgba(c)
        if outlier_removal:
            inv_ind = np.ones(pcd.shape[0], np.bool)
            inv_ind[ind] = 0
            attr[name][inv_ind] = (0, 0, 0, 0)

    if im_depth2 is not None:
        x2, y2, z2 = points2d_from_image(im_depth2, depth_mask=None)
        pcd2 = points2d_to_3d(x2, y2, z2, origin, focal, im_depth.shape)

        new_attr = OrderedDict()
        for a in attr.keys():
            if a == 'grayscale':
                # add a grayscale attribute with and without im_depth2
                new_attr[a] = np.r_[attr[a], np.ones((pcd2.shape[0], 4)) * (1, 0, 0, 0)]
                new_attr[a + "+lidar"] = np.r_[attr[a], np.ones((pcd2.shape[0], 4)) * (1, 0, 0, 0.3)]

                error = np.abs(im_depth - im_depth2)[y2, x2]
                error = get_cmap(vmin=0, vmax=50, cmap='jet').to_rgba(error, alpha=0.3)
                new_attr[a + "+lidar error"] = np.r_[attr[a], error]
            else:
                new_attr[a] = np.r_[attr[a], np.ones((pcd2.shape[0], 4)) * (0, 0, 0, 0)]
        attr = new_attr

        pcd = np.r_[pcd, pcd2]

    print("color attributes:")
    for i, a in enumerate(attr.keys()):
        print("\t" + str(i + 1) + ": " + a)

    pcd_ = np.c_[pcd[:, 0], pcd[:, 2], -pcd[:, 1]]
    v = pptk.viewer(pcd_, *attr.values())
    v.set(point_size=point_size, show_grid=True, lookat=(0, 0, 0), r=15, phi=3 * np.pi / 2, theta=0)
    if return_pcd:
        return v, pcd
    return v


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
    xx[P[2,:] <= 0] = 1e10
    yy[P[2,:] <= 0] = 1e10
    return interp2(im_source, [yy.ravel(), xx.ravel()], order=3).reshape(target_shape)


def text_to_image(text, im_size, text_color='red'):
    img = Image.new('RGB', im_size)
    d = ImageDraw.Draw(img)
    d.text((5, 5), text, fill=text_color)
    return img


def concat_images(images, rows, cols):
    original_shape = height, width = images[0].shape
    new_shape = np.array(original_shape) * [rows, cols]
    out_img = np.zeros(new_shape)
    for i in range(rows):
        for j in range(cols):
            out_img[height*i:height*(i+1), width*j:width*(j+1)] = images[i * cols + j]
    return out_img

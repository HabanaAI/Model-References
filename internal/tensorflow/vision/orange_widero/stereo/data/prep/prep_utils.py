import numpy as np
import os,sys
from file2path import file2path

def filter_short(im, view, factor=2, rad=50, thresh=0.2):
    im_lidar = 1 * im
    im_lidar[im_lidar == 0.] = -1
    steps = np.linspace(-1, 100, 102)
    im_lidar_dig = np.digitize(im_lidar, steps)
    xx, yy = np.meshgrid(np.arange(im_lidar_dig.shape[1]), np.arange(im_lidar_dig.shape[0]))
    im_lidar_dig_tensor = np.zeros((im_lidar_dig.shape[0], im_lidar_dig.shape[1], len(steps) - 2), dtype=np.int32)
    for i in np.arange(2, len(steps)):
        im_lidar_dig_tensor[:, :, i - 2] = im_lidar_dig == i
    # im_lidar_dig_tensor[np.arange(im_lidar_dig.shape[0]), np.arange(im_lidar_dig.shape[1]), np.maximum(np.minimum(im_lidar_dig - 2, im_lidar_dig_tensor.shape[2] - 1), 0)] = 1

    im_lidar_dig_integral = np.cumsum(np.cumsum(im_lidar_dig_tensor, 0), 1)
    x1 = np.maximum(np.minimum(xx + rad, im_lidar_dig.shape[1] - 1), 0)
    x2 = np.maximum(np.minimum(xx - rad, im_lidar_dig.shape[1] - 1), 0)
    y1 = np.maximum(np.minimum(yy + rad, im_lidar_dig.shape[0] - 1), 0)
    y2 = np.maximum(np.minimum(yy - rad, im_lidar_dig.shape[0] - 1), 0)
    windows = (im_lidar_dig_integral[y1, x1, :] +
               im_lidar_dig_integral[y2, x2, :] -
               im_lidar_dig_integral[y1, x2, :] -
               im_lidar_dig_integral[y2, x1, :])
    windows_cumsum = np.cumsum(windows, 2)
    y, x = np.where(im_lidar_dig - 2 > 0)
    ids = np.where(
        windows_cumsum[y, x, ((im_lidar_dig[y, x] - 3) / factor).astype(np.int32)] > thresh * windows_cumsum[y, x, -1])[
        0]
    im_lidar[y[ids], x[ids]] = 0
    im_lidar[im_lidar == -1] = 0
    instance_masks = view['instance_masks']['vd_masks_scaled_compressed']
    instance_masks = np.unpackbits(instance_masks).reshape((instance_masks.shape[0], instance_masks.shape[1], 8))
    instance_masks = [np.squeeze(mask) for mask in np.split(instance_masks, 8, 2)]
    rm_x, rm_y = [], []
    for mask in instance_masks:
        if not np.all(np.logical_not(mask)):
            im_lidar_mask = im_lidar * mask
            im_lidar_mask[im_lidar_mask == 0] = np.nan
            y, x = np.where(im_lidar_mask > 1.5 * np.percentile(im_lidar_mask[np.isfinite(im_lidar_mask)], 30))
            rm_x.extend(x)
            rm_y.extend(y)
    im_lidar[rm_y, rm_x] = 0
    return im_lidar


def inv_lidar(lidar):
    return ((1.0 - np.equal(lidar, 0.0)) * ((lidar + 1e-9) ** -1)).astype(np.float32)


def prep_view_v23(views, cntr, srnd, speed_thresh=1., extra_images=False, inferenceOnly=False):
    data = {}
    data['I_cntr'] = views[cntr]['image'][:, :, np.newaxis] / 255.
    T_cntr_srnd = []

    for i,view in enumerate(srnd):
        data['I_srnd_%d' % i] = views[view]['image'][:, :, np.newaxis] / 255.
        RT_srnd_to_cntr = np.matmul(np.linalg.inv(views[cntr]['RT_view_to_main']),
                                    views[view]['RT_view_to_main'])
        T_cntr_srnd.append(np.expand_dims(-RT_srnd_to_cntr[:3, 3], 0))

    data['focal'] = np.array([views[cntr]['focal'], views[cntr]['focal']]).astype(np.float32)
    data['origin'] = np.array(views[cntr]['origin']).astype(np.float32)
    data['T_cntr_srnd'] = np.concatenate(T_cntr_srnd, axis=0).astype(np.float32)
    data['gi'] = views[cntr]['grab_index']
    data['clip_name'] = np.array(views[cntr]['clip_name'])

    if inferenceOnly:
        return data

    data['photo_loss_im_mask'] = np.logical_or(views[cntr]['masks']['car_body'][:, :, np.newaxis],
                                               np.equal(data['I_cntr'], 0.)).astype(np.float32)

    if extra_images:
        data['I_cntr_red'] = views[cntr]['images_extra']['red'][:, :, np.newaxis] / 255.
        data['I_cntr_ltm'] = views[cntr]['images_extra']['ltm'][:, :, np.newaxis] / 255.
        for i,view in enumerate(srnd):
            for tm in ['ltm', 'red']:
                data['I_srnd_%s_%d' % (tm, i)] = views[view]['images_extra']['red'][:, :, np.newaxis] / 255.

    data['im_lidar_inv'] = inv_lidar(views[cntr]['lidars']['medium'][:, :, np.newaxis])
    data['im_lidar_short_inv'] = inv_lidar(views[cntr]['lidars']['short'][:, :, np.newaxis])
    data['im_mask'] = np.logical_or(views[cntr]['masks']['PD_approved'][:, :, np.newaxis],
                                views[cntr]['masks']['VD_moving'][:, :, np.newaxis]).astype(np.float32)
    data['photo_loss_im_mask'] = np.logical_or(views[cntr]['masks']['car_body'][:, :, np.newaxis],
                                            np.equal(data['I_cntr'], 0.)).astype(np.float32)
    data['gi'] = views[cntr]['grab_index'].astype(np.int32)
    data['clip_name'] = views[cntr]['clip_name']
    data['cntr_cam_name'] =  np.array(cntr.split('_to_')[0])
    data['speed'] = np.abs(views[cntr]['speed'].astype(np.float32))

    if data['speed'] < speed_thresh:
        return None
    else:
        return data


def prep_view_v25(views, cntr, srnd, speed_thresh=1., extra_images=False, inferenceOnly=False):
    data = {}
    data['I_cntr'] = views[cntr]['image'][:, :, np.newaxis] / 255.
    T_cntr_srnd = []
    for i,view in enumerate(srnd):
        data['I_srnd_%d' % i] = views[view]['image'][:, :, np.newaxis] / 255.
        RT_srnd_to_cntr = np.matmul(np.linalg.inv(views[cntr]['RT_view_to_main']),
                                    views[view]['RT_view_to_main'])
        T_cntr_srnd.append(np.expand_dims(-RT_srnd_to_cntr[:3, 3], 0))
    data['focal'] = np.array([views[cntr]['focal'], views[cntr]['focal']]).astype(np.float32)
    data['origin'] = np.array(views[cntr]['origin']).astype(np.float32)
    data['T_cntr_srnd'] = np.concatenate(T_cntr_srnd, axis=0).astype(np.float32)
    if inferenceOnly:
        return data
    if extra_images:
        data['I_cntr_red'] = views[cntr]['images_extra']['red'][:, :, np.newaxis] / 255.
        data['I_cntr_ltm'] = views[cntr]['images_extra']['ltm'][:, :, np.newaxis] / 255.
        for i,view in enumerate(srnd):
            for tm in ['ltm', 'red']:
                data['I_srnd_%s_%d' % (tm, i)] = views[view]['images_extra']['red'][:, :, np.newaxis] / 255.
    data['im_lidar_inv'] = inv_lidar(views[cntr]['lidars']['medium'][:, :, np.newaxis])
    data['im_lidar_clusters'] = views[cntr]['lidar_clusters']['medium'].astype(np.int32)

    data['im_mask'] = np.logical_or(views[cntr]['masks']['PD_approved'][:, :, np.newaxis],
                                views[cntr]['masks']['VD_moving'][:, :, np.newaxis]).astype(np.float32)
    data['gi'] = views[cntr]['grab_index'].astype(np.int32)
    data['clip_name'] = views[cntr]['clip_name']
    data['cntr_cam_name'] =  np.array(cntr.split('_to_')[0])
    data['speed'] = np.abs(views[cntr]['speed'].astype(np.float32))

    if data['speed'] < speed_thresh:
        return None
    else:
        return data


def prep_view_n(views, cntr, srnd, fail_missing_lidar=True, inferenceOnly=False, useShort=False, useVD_notMoving=False):
    im_cntr = views[cntr]['image'] / 255.
    im_srnd = []
    T_cntr_srnd = []
    for view in srnd:
        if view is 'fake':
            im_srnd.append(np.zeros_like(im_cntr))
            RT_srnd_to_cntr = np.eye(4, dtype=np.float32)
            T_cntr_srnd.append(np.expand_dims(-RT_srnd_to_cntr[:3, 3], 0))
        else:
            im_srnd.append(views[view]['image'] / 255.)
            RT_srnd_to_cntr = np.matmul(np.linalg.inv(views[cntr]['RT_view_to_main']),
                                        views[view]['RT_view_to_main'])
            T_cntr_srnd.append(np.expand_dims(-RT_srnd_to_cntr[:3, 3], 0))
    focal = np.array([views[cntr]['focal'], views[cntr]['focal']]).astype(np.float32)
    origin = np.array(views[cntr]['origin']).astype(np.float32)

    ims = [im_cntr]
    ims.extend(im_srnd)
    inp = np.concatenate([im[np.newaxis, :, :] for im in ims], axis=0).astype(np.float32)
    T_cntr_srnd = np.concatenate(T_cntr_srnd, axis=0).astype( np.float32)
    if inferenceOnly:
        return {'inp': inp, 'T_cntr_srnd': T_cntr_srnd, 'origin': origin, 'focal': focal}
    try:
        im_lidar = views[cntr]['lidars']['medium'].astype(np.float32)
        im_lidar_short = views[cntr]['lidars']['short'].astype(np.float32)
        if useShort:
            im_lidar_short = filter_short(im_lidar_short, views[cntr])
        assert not np.any(np.isnan(im_lidar))
        assert not np.any(np.isnan(im_lidar_short))

    except:
        if fail_missing_lidar:
            return None
        else:
            im_lidar = np.zeros_like(im_cntr).astype(np.float32)
            im_lidar_short = np.zeros_like(im_cntr).astype(np.float32)
    if fail_missing_lidar and np.min(im_lidar) == 0. and np.max(im_lidar) == 0.:
        return None
    if fail_missing_lidar and useShort and np.min(im_lidar_short) == 0. and np.max(im_lidar_short) == 0.:
        return None
    if useVD_notMoving:
        im_mask = np.logical_or(views[cntr]['masks']['PD_approved'],
                                views[cntr]['masks']['VD']).astype(np.float32)
    else:
        im_mask = np.logical_or(views[cntr]['masks']['PD_approved'],
                                views[cntr]['masks']['VD_moving']).astype(np.float32)
    gi = views[cntr]['grab_index'].astype(np.int32)
    clip_name = views[cntr]['clip_name']

    view = {'inp': inp, 'im_lidar': im_lidar, 'im_lidar_short': im_lidar_short, 'im_mask': im_mask,
            'T_cntr_srnd': T_cntr_srnd, 'origin': origin, 'focal': focal,
            'gi': gi, 'clip_name': clip_name, 'cntr_cam_name': np.array(cntr.split('_to_')[0])}
    if 'speed' in views[cntr].keys():
        view['speed'] = views[cntr]['speed'].astype(np.float32)
    return view


def prep_view(views, cntr, right, left, fail_missing_lidar=True, inferenceOnly=False):
    im_cntr = views[cntr]['image'] / 255.
    im_left = views[left]['image'] / 255.
    im_right = views[right]['image'] / 255.

    RT_left_to_cntr = np.matmul(
        np.linalg.inv(views[cntr]['RT_view_to_main']),
        views[left]['RT_view_to_main'])
    T_cntr_srnd_left = -RT_left_to_cntr[:3, 3]
    RT_right_to_cntr = np.matmul(
        np.linalg.inv(views[cntr]['RT_view_to_main']),
        views[right]['RT_view_to_main'])
    T_cntr_srnd_right = -RT_right_to_cntr[:3, 3]

    focal = np.array([views[cntr]['focal'], views[cntr]['focal']]).astype(np.float32)
    origin = np.array(views[cntr]['origin']).astype(np.float32)

    inp = np.concatenate([im_cntr[np.newaxis, :, :], im_left[np.newaxis, :, :], im_right[np.newaxis, :, :]], axis=0).astype(np.float32)
    T_cntr_srnd = np.concatenate([T_cntr_srnd_left[np.newaxis, :], T_cntr_srnd_right[np.newaxis, :]], axis=0).astype(np.float32)

    if inferenceOnly:
        return {'inp': inp, 'T_cntr_srnd': T_cntr_srnd, 'origin': origin, 'focal': focal}

    try:
        im_lidar = views[cntr]['lidars']['medium'].astype(np.float32)
        im_lidar_short = views[cntr]['lidars']['short'].astype(np.float32)
        assert not np.any(np.isnan(im_lidar))
        assert not np.any(np.isnan(im_lidar_short))

    except:
        if fail_missing_lidar:
            return None
        else:
            im_lidar = np.zeros_like(im_cntr).astype(np.float32)
            im_lidar_short = np.zeros_like(im_cntr).astype(np.float32)
    if fail_missing_lidar and np.min(im_lidar) == 0. and np.max(im_lidar) == 0.:
        return None
    im_mask = np.logical_or(views[cntr]['masks']['PD_approved'],
                            views[cntr]['masks']['VD_moving']).astype(np.float32)
    gi = views[cntr]['grab_index'].astype(np.int32)
    clip_name = views[cntr]['clip_name']

    view = {'inp': inp, 'im_lidar': im_lidar, 'im_lidar_short': im_lidar_short, 'im_mask': im_mask,
            'T_cntr_srnd': T_cntr_srnd, 'origin': origin, 'focal': focal,
            'gi': gi, 'clip_name': clip_name, 'cntr_cam_name': np.array(cntr.split('_to_')[0])}
    if 'speed' in views[cntr].keys():
        view['speed'] = views[cntr]['speed'].astype(np.float32)
    
    return view


def prep_view_instance(views, cntr, right, left, fail_missing_lidar=True, inferenceOnly=False, curr_ids=None,
                       curr_RT=None, prefix=""):
    im_cntr = views[cntr]['image'] / 255.
    im_left = views[left]['image'] / 255.
    im_right = views[right]['image'] / 255.

    RT_left_to_cntr = np.matmul(
        np.linalg.inv(views[cntr]['RT_view_to_main']),
        views[left]['RT_view_to_main'])
    T_cntr_srnd_left = -RT_left_to_cntr[:3, 3]
    RT_right_to_cntr = np.matmul(
        np.linalg.inv(views[cntr]['RT_view_to_main']),
        views[right]['RT_view_to_main'])
    T_cntr_srnd_right = -RT_right_to_cntr[:3, 3]

    focal = np.array([views[cntr]['focal'], views[cntr]['focal']]).astype(np.float32)
    origin = np.array(views[cntr]['origin']).astype(np.float32)

    inp = np.concatenate([im_cntr[np.newaxis, :, :], im_left[np.newaxis, :, :], im_right[np.newaxis, :, :]], axis=0).astype(np.float32)
    T_cntr_srnd = np.concatenate([T_cntr_srnd_left[np.newaxis, :], T_cntr_srnd_right[np.newaxis, :]], axis=0).astype(np.float32)

    if inferenceOnly:
        return {'inp': inp, 'T_cntr_srnd': T_cntr_srnd, 'origin': origin, 'focal': focal}

    im_lidar = views[cntr]['lidars']['medium'].astype(np.float32)
    if fail_missing_lidar and np.min(im_lidar) == 0. and np.max(im_lidar) == 0.:
        return None
    im_mask = np.logical_or(views[cntr]['masks']['PD_approved'],
                            views[cntr]['masks']['VD_moving']).astype(np.float32)
    pd_mask = views[cntr]['masks']['PD_approved'].astype(np.float32)
    # gi = views[cntr]['grab_index']
    # clip_name = views[cntr]['clip_name']

    instance_mask = views[cntr]['instance_masks']['vd_masks_scaled_compressed']
    instance_ids = views[cntr]['instance_masks']['vd_ids']

    # What is saved as view_to_clip is actually clip_to_view
    curr2frame = views[cntr]['RT_view_to_clip'].astype(np.float32)
    ##########################################################
    # REMEMBER TO CHANGE THIS WHEN BUG IN MAKE_VIEW IS FIXED #
    ##########################################################

    # curr2frame = np.linalg.inv(views[cntr]['RT_view_to_clip']).astype(np.float32)

    if curr_ids is not None and curr_RT is not None:
        curr_ids = curr_ids.astype(np.int32)
        unpacked_instances = unpack_bit_array(instance_mask)
        new_unpacked_instances = np.zeros_like(unpacked_instances)
        counter = 0
        for i,id in enumerate(curr_ids):
            if id in instance_ids and id != 0:
                new_unpacked_instances[counter,:,:] = unpacked_instances[instance_ids.tolist().index(id),:,:]
                counter += 1
        instance_mask, instance_ids = bit_compress(new_unpacked_instances, curr_ids)

        # What is saved as view_to_clip is actually clip_to_view
        curr2frame = np.matmul(np.linalg.inv(views[cntr]['RT_view_to_clip']), curr_RT)
        # curr2frame = np.matmul(views[cntr]['RT_view_to_clip'], curr_RT)

    instance_ids = instance_ids.astype(np.float32)
    curr2frame = curr2frame.astype(np.float32)
    instance_mask = instance_mask.astype(np.float32)

    # gi = gi.astype(np.int32)

    return {prefix+'inp': inp, prefix+'im_lidar': im_lidar, prefix+'im_mask': im_mask, prefix+'pd_mask': pd_mask,
            prefix+'instance_mask': instance_mask, prefix+'instance_ids': instance_ids,
            prefix+'T_cntr_srnd': T_cntr_srnd, prefix+'origin': origin, prefix+'focal': focal,
            prefix+'curr2frame': curr2frame}


def prep_view_instance_with_extras(views, cntr, right, left, fail_missing_lidar=False, inferenceOnly=False, curr_ids=None,
                       curr_RT=None, prefix=""):

    im_cntr = views[cntr]['image'] / 255.
    im_left = views[left]['image'] / 255.
    im_right = views[right]['image'] / 255.

    RT_left_to_cntr = np.matmul(
        np.linalg.inv(views[cntr]['RT_view_to_main']),
        views[left]['RT_view_to_main'])
    T_cntr_srnd_left = -RT_left_to_cntr[:3, 3]
    RT_right_to_cntr = np.matmul(
        np.linalg.inv(views[cntr]['RT_view_to_main']),
        views[right]['RT_view_to_main'])
    T_cntr_srnd_right = -RT_right_to_cntr[:3, 3]

    focal = np.array([views[cntr]['focal'], views[cntr]['focal']]).astype(np.float32)
    origin = np.array(views[cntr]['origin']).astype(np.float32)

    inp = np.concatenate([im_cntr[np.newaxis, :, :], im_left[np.newaxis, :, :], im_right[np.newaxis, :, :]], axis=0).astype(np.float32)
    T_cntr_srnd = np.concatenate([T_cntr_srnd_left[np.newaxis, :], T_cntr_srnd_right[np.newaxis, :]], axis=0).astype(np.float32)

    if inferenceOnly:
        curr2frame = views[cntr]['RT_view_to_clip'].astype(np.float32)
        if curr_RT is not None:
            curr2frame = np.matmul(np.linalg.inv(views[cntr]['RT_view_to_clip']), curr_RT)

        return {prefix+'inp': inp, prefix+'T_cntr_srnd': T_cntr_srnd, prefix+'origin': origin, prefix+'focal': focal,
                prefix+'curr2frame': curr2frame}

    # im_lidar = views[cntr]['lidars']['medium'].astype(np.float32)
    # if fail_missing_lidar and np.min(im_lidar) == 0. and np.max(im_lidar) == 0.:
    #     return None
    im_mask = np.logical_or(views[cntr]['masks']['PD_approved'],
                            views[cntr]['masks']['VD_moving']).astype(np.float32)
    pd_mask = views[cntr]['masks']['PD_approved'].astype(np.float32)
    # gi = views[cntr]['grab_index']
    # clip_name = views[cntr]['clip_name']

    instance_mask = views[cntr]['instance_masks']['vd_masks_scaled_compressed']
    instance_ids = views[cntr]['instance_masks']['vd_ids']

    # What is saved as view_to_clip is actually clip_to_view
    curr2frame = views[cntr]['RT_view_to_clip'].astype(np.float32)

    # Add segmentation
    seg_im = views[cntr]['labels']['seg'].astype(np.float32)

    vcl = views[cntr]['labels']['vcl'].astype(np.float32)

    if curr_ids is not None and curr_RT is not None:
        curr_ids = curr_ids.astype(np.int32)
        unpacked_instances = unpack_bit_array(instance_mask)
        new_unpacked_instances = np.zeros_like(unpacked_instances)
        new_vcl = np.ones_like(vcl, dtype=np.float32)*-1
        counter = 0
        for i,id in enumerate(curr_ids):
            if id in instance_ids and id != 0:
                new_unpacked_instances[counter,:,:] = unpacked_instances[instance_ids.tolist().index(id),:,:]
                new_vcl[counter,:] = vcl[instance_ids.tolist().index(id),:]
                counter += 1
        instance_mask, instance_ids = bit_compress(new_unpacked_instances, curr_ids)

        vcl = new_vcl

        curr2frame = np.matmul(np.linalg.inv(views[cntr]['RT_view_to_clip']), curr_RT)
        # What is saved as view_to_clip is actually clip_to_view
        # curr2frame = np.matmul(views[cntr]['RT_view_to_clip'], curr_RT)

    instance_ids = instance_ids.astype(np.float32)
    curr2frame = curr2frame.astype(np.float32)
    instance_mask = instance_mask.astype(np.float32)

    return {prefix+'inp': inp, prefix+'im_mask': im_mask, prefix+'pd_mask': pd_mask,
            prefix+'instance_mask': instance_mask, prefix+'instance_ids': instance_ids,
            prefix+'T_cntr_srnd': T_cntr_srnd, prefix+'origin': origin, prefix+'focal': focal,
            prefix+'curr2frame': curr2frame, prefix+'seg_im': seg_im, prefix+'vcl': vcl}


def prep_input_structure_view(views, cntr, steps=np.linspace(0.0, 3.0 ** -1, 45), lidar_completion=False, global_step=False, add_training=False):
    im_sz = views[cntr]['image'].shape
    origin = views[cntr]['origin']
    left, right = -origin[0], im_sz[1] - origin[0]
    bottom, top = -origin[1], im_sz[0] - origin[1]
    x, y = np.meshgrid(np.arange(left, right), np.arange(bottom, top))
    x = (x[np.newaxis, :, :, np.newaxis]).astype('float32')
    y = (y[np.newaxis, :, :, np.newaxis]).astype('float32')
    arch_consts = {'x': x, 'y': y, 'steps': steps}
    loss_consts = {'x': x, 'y': y}
    arch_inputs = ['inp', 'T_cntr_srnd', 'focal', 'origin']
    if lidar_completion:
        arch_inputs.append('im_lidar')
        arch_inputs.append('training')
    elif add_training:
        arch_inputs.append('training')
    loss_inputs = ['inp', 'T_cntr_srnd', 'focal', 'origin', 'im_lidar', 'im_mask']
    if global_step:
        loss_inputs.append('global_step')
    return arch_inputs, loss_inputs, arch_consts, loss_consts


def prep_instance_input_structure_view(views, cntr, steps=np.linspace(0.0, 3.0 ** -1, 45)):
    im_sz = views[cntr]['image'].shape
    origin = views[cntr]['origin']
    left, right = -origin[0], im_sz[1] - origin[0]
    bottom, top = -origin[1], im_sz[0] - origin[1]
    x, y = np.meshgrid(np.arange(left, right), np.arange(bottom, top))
    x = (x[np.newaxis, :, :, np.newaxis]).astype('float32')
    y = (y[np.newaxis, :, :, np.newaxis]).astype('float32')
    arch_consts = {'x': x, 'y': y, 'steps': steps}
    loss_consts = {'x': x, 'y': y}
    arch_inputs = ['curr_inp', 'prev_inp', 'next_inp', 'prev_curr2frame', 'next_curr2frame',
         'curr_T_cntr_srnd', 'prev_T_cntr_srnd', 'next_T_cntr_srnd', 'curr_focal',
         'curr_origin']
    loss_inputs = ['curr_inp', 'prev_inp', 'next_inp', 'prev_curr2frame', 'next_curr2frame',
                   'curr_T_cntr_srnd', 'prev_T_cntr_srnd', 'next_T_cntr_srnd', 'curr_focal',
                   'curr_origin', 'curr_im_mask', 'prev_im_mask', 'next_im_mask', 'curr_instance_mask',
                   'prev_instance_mask', 'next_instance_mask', 'curr_instance_ids', 'prev_instance_ids',
                   'next_instance_ids']
    return arch_inputs, loss_inputs, arch_consts, loss_consts


def prep_instance_input_structure_view(views, cntr, steps=np.linspace(0.0, 3.0 ** -1, 45)):
    im_sz = views[cntr]['image'].shape
    origin = views[cntr]['origin']
    left, right = -origin[0], im_sz[1] - origin[0]
    bottom, top = -origin[1], im_sz[0] - origin[1]
    x, y = np.meshgrid(np.arange(left, right), np.arange(bottom, top))
    x = (x[np.newaxis, :, :, np.newaxis]).astype('float32')
    y = (y[np.newaxis, :, :, np.newaxis]).astype('float32')
    arch_consts = {'x': x, 'y': y, 'steps': steps}
    loss_consts = {'x': x, 'y': y}
    arch_inputs = ['curr_inp', 'prev_inp', 'next_inp', 'prev_curr2frame', 'next_curr2frame',
         'curr_T_cntr_srnd', 'prev_T_cntr_srnd', 'next_T_cntr_srnd', 'curr_focal',
         'curr_origin']
    loss_inputs = ['curr_inp', 'prev_inp', 'next_inp', 'prev_curr2frame', 'next_curr2frame',
                   'curr_T_cntr_srnd', 'prev_T_cntr_srnd', 'next_T_cntr_srnd', 'curr_focal',
                   'curr_origin', 'curr_im_mask', 'prev_im_mask', 'next_im_mask', 'curr_instance_mask',
                   'prev_instance_mask', 'next_instance_mask', 'curr_instance_ids', 'prev_instance_ids',
                   'next_instance_ids']
    return arch_inputs, loss_inputs, arch_consts, loss_consts


def prep_instance_input_structure_view_small(views, cntr, steps=np.linspace(0.0, 3.0 ** -1, 45), inferenceOnly=False):
    if inferenceOnly:
        views = views['curr']
    im_sz = views[cntr]['image'].shape
    origin = views[cntr]['origin']
    left, right = -origin[0], im_sz[1] - origin[0]
    bottom, top = -origin[1], im_sz[0] - origin[1]
    x, y = np.meshgrid(np.arange(left, right), np.arange(bottom, top))
    x = (x[np.newaxis, :, :, np.newaxis]).astype('float32')
    y = (y[np.newaxis, :, :, np.newaxis]).astype('float32')
    arch_consts = {'x': x, 'y': y, 'steps': steps}
    loss_consts = {'x': x, 'y': y}
    if inferenceOnly:
        arch_inputs = ['curr_inp', 'prev_inp', 'prev_curr2frame',
                       'curr_T_cntr_srnd', 'prev_T_cntr_srnd', 'curr_focal',
                       'curr_origin']
    else:
        arch_inputs = ['curr_inp', 'prev_inp', 'next_inp', 'prev_curr2frame', 'next_curr2frame',
             'curr_T_cntr_srnd', 'prev_T_cntr_srnd', 'next_T_cntr_srnd', 'curr_focal',
             'curr_origin']
    loss_inputs = ['curr_inp', 'prev_inp', 'next_inp', 'prev_curr2frame', 'next_curr2frame',
                   'curr_T_cntr_srnd', 'prev_T_cntr_srnd', 'next_T_cntr_srnd', 'curr_focal',
                   'curr_origin', 'curr_im_mask', 'prev_im_mask', 'next_im_mask', 'curr_instance_mask',
                   'prev_instance_mask', 'next_instance_mask', 'curr_instance_ids', 'prev_instance_ids',
                   'next_instance_ids', 'curr_seg_im', 'prev_seg_im', 'next_seg_im', 'curr_vcl',
                   'prev_vcl', 'next_vcl']
    return arch_inputs, loss_inputs, arch_consts, loss_consts


def bit_compress(mask_lst, id_lst):
    # INPUT: a list of uint8 numpy arrays and a list of scalars
    # OUTPUT: a compressed uint8 bit array and list of scalars

    # NOTE this will alter the input lists if they have less than 8 items and because of how python works
    # they will stay altered

    while len(mask_lst) < 8:
        mask_lst.append(np.zeros_like(mask_lst[0], dtype=np.uint8))
        id_lst.append(0)

    compressed_array = np.array(mask_lst)
    compressed_array = np.squeeze(np.packbits(compressed_array, axis=0))

    return compressed_array, id_lst


def unpack_bit_array(compressed_array):

    out_bits = np.unpackbits(np.expand_dims(compressed_array, axis=0), axis=0)

    return out_bits


def get_center_view(views):
    return [view for view in views if len(set(view.split("_to_"))) == 1][0]


def get_surround_views(views):
    cntr_view = get_center_view(views)
    return [view for view in views if view != cntr_view]


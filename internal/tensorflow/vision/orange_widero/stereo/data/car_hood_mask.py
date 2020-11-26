# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

import numpy as np
import os
import cv2
import pickle
from devkit.clip import MeClip
from stereo.data.clip_utils import clip_name_to_sess_path, clip_name_to_sess_name
from random import sample
from tqdm import tqdm
from scipy.ndimage import label


# +
def get_session_section_path_from_clip(clip_name, section):
    session_path = clip_name_to_sess_path(clip_name)
    return os.path.join(session_path, 'Clips_' + section)

def sample_session_section(section, session_section_path, cams, min_num_of_imgs=250, pyr=-1):
    session_clips = [item for item in os.listdir(session_section_path) if section in item]
    num_clips = len(session_clips)
    num_of_sampled_frames = int(np.ceil(min_num_of_imgs/num_clips)) + 1
    cam_imgs = {cam:[] for cam in cams}
    for sess_clip_name in tqdm(session_clips):
        try:
            clip = MeClip(sess_clip_name)
            sampled_frames = sample(np.arange(clip.last_gfi()), num_of_sampled_frames)
            for cam in cams:
                for frame_num in sampled_frames:
                    frame = clip.get_frame(frame=frame_num, camera_name=cam, tone_map='ltm')
                    cam_imgs[cam].append(frame[0]['pyr'][pyr].im)
        except:
            pass
    return cam_imgs

def get_session_imgs(clip_name, sections_cameras, output_path, save_imgs=True, ext=''):
    dump_dir = os.path.join(output_path, clip_name_to_sess_name(clip_name))
    dump_path = os.path.join(dump_dir, 'imgs' + ext)
    all_imgs = {}
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            all_imgs = pickle.load(f)
    else:
        for section in sections_cameras:
            session_section_path = get_session_section_path_from_clip(clip_name, section)
            all_imgs.update(sample_session_section(section, session_section_path, sections_cameras[section]))
        if save_imgs:
            if not os.path.exists(dump_dir):
                os.mkdir(dump_dir)
            with open(dump_path, 'wb') as output:
                pickle.dump(all_imgs, output, pickle.HIGHEST_PROTOCOL)
    return all_imgs


# +
def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def crop_images(imgs):
    black_count = np.count_nonzero(imgs[0], axis=1)
    pad = first_nonzero(black_count, axis=0), first_nonzero(black_count[::-1], axis=0)
    crop = np.array(imgs[:,pad[0]:-pad[1]])
    return (crop, pad)

def pad_black_margins(img, pad):
    w, h = img.shape
    padded = np.zeros((sum(pad)+w,h))
    padded[:pad[0]] = np.zeros((pad[0],h))
    padded[pad[0]:-pad[1]] = img
    padded[-pad[1]:] = np.zeros((pad[1],h))
    return padded


def remove_noises(img, thresh=100, remove_right_noise=True):
    if remove_right_noise:
        img[:,-2:] = 0
    dilation = cv2.dilate(img.astype('uint8'), np.ones((3,3)), iterations=5)
    dilation = remove_noises_by_connected_components(dilation, thresh=thresh)
    dilation = cv2.dilate(dilation.astype('uint8'), np.ones((3,3)), iterations=5)
    erosion = cv2.erode(dilation, np.ones((3,3)), iterations=6)
    return erosion
    
def is_intersecting_edge(component):
    return sum(component[0]) > 0 or sum(component[-1]) > 0 or sum(component[:,0]) > 0 or sum(component[:,-1]) > 0
    
def remove_noises_by_connected_components(img, thresh):
    labeled, n = label(img, structure=np.ones((3,3)))
    labeled[labeled==0] = -1
    components_sizes = np.array([np.sum(labeled == i) for i in np.arange(n+1)])
    if thresh:
        selected_components = np.where(components_sizes > thresh)[0]
        if len(selected_components) == 0:
            selected_components = [components_sizes.argmax()]
    else:
        raise Exception('thresh or number of must be given')
    selected_components = [comp for comp in selected_components if is_intersecting_edge(labeled == comp)]
    return np.isin(labeled, selected_components)


# -

def get_mask_edges(cam_imgs, grad_thresh=2, orientation_thresh=0.7):
    cam_imgs = np.array(cam_imgs).astype('float32')
    cam_imgs, pad = crop_images(cam_imgs)
    
    dy, dx = np.array(np.gradient(cam_imgs, axis=[1,2]))

    magnitude_thresh = np.sqrt(dx**2+dy**2) > grad_thresh
    orientation = np.arctan2(dy, dx)
    
    dy, dx = 0,0
    
    np.add.at(orientation, orientation < 0 , np.pi)
    orientation *= 2
    orientation = np.exp(orientation*1j)
    orientation_mean = np.abs(np.mean(orientation * magnitude_thresh, axis=0))
    
    mask_edges = orientation_mean > orientation_thresh
    mask_edges = remove_noises(mask_edges)
       
    mask_edges = pad_black_margins(mask_edges, pad)
    return mask_edges


def get_mask(mask_edges, cam):
    left = 'Left' in cam
    front = 'front' in cam
    if left:
        mask_edges = mask_edges.copy()[:,::-1]
    if front:
        mask_edges = np.rot90(mask_edges.copy(),3)
    mask = np.ones_like(mask_edges)
    first_non_zeros = first_nonzero(mask_edges, 1)
    for i in np.arange(mask.shape[0]):
        if first_non_zeros[i] != -1:
            mask[i, first_non_zeros[i]:] = 0
    if front:
        mask = np.rot90(mask)
    if left:
        mask = mask[:,::-1]
    return mask


def get_cameras_masks(all_cams_imgs, thresh=0.7):
    masks = {}
    for cam in all_cams_imgs:
        mask_edges = get_mask_edges(all_cams_imgs[cam], orientation_thresh=thresh)
        masks[cam] = get_mask(mask_edges, cam)
    return masks


def save_cameras_masks(masks, output_path, clip_name, as_obj=False):
    base_dir = os.path.join(output_path, clip_name_to_sess_name(clip_name))
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    if as_obj:
        masks_path = os.path.join(base_dir, 'masks.npz')
        with open(masks_path, 'wb') as output:
            np.savez(output, **masks)
        return
    
    masks_dir = os.path.join(base_dir, 'masks')
    os.mkdir(masks_dir)
    for cam in masks:
        mask_path = os.path.join(masks_dir, cam + '.png')
        cv2.imwrite(mask_path, 255 * masks[cam])

import numpy as np
import os
import h5py

# NSS_Voting
hl_mapping = {'bike': 8, 'suspected_sidewalk': -9, 'out_of_image': -2, 'not_road': 0, 'pedestrian': 9,
              'generalObject': 10, 'suspected_generalObject': -13, 'suspected_road': -4, 'road_under_car': 4,
              'suspected_road_under_car': -7, 'suspected_road_edge': -5, 'suspected_elevated': -6,
              'suspected_not_road': -3, 'suspected_pedestrian': -12, 'car': 7, 'suspected_problem': -2, 'elevated': 3,
              'sem_drivable': 5, 'ignore': -1, 'road_edge': 2, 'suspected_car': -10, 'suspected_sem_drivable': -8,
              'sidewalk': 6, 'suspected_bike': -11, 'road': 1}

re_mapping = {'canal': 8, 'concreteBarrier': 1, 'wall': 6, 'guardrail': 0, 'snow': 4, 'suspected_problem': -2,
              'ignore': -1, 'curb': 2, 'gate': 5, 'greeneryGravel': 3, 'ditch': 7}

# RnR
RnR_mapping = {'unknown': 0, 'not_road': 1, 'road': 2, 'road_edge': 3, 'elevated': 4, 'car': 5, 'bike': 6, 'ped': 7,
               'genObj': 8}

def retrieve_segmentation(clip_name, gi, camera_name, nss_dir='/mobileye/algo_STEREO3/stereo/data/NSS/',
                          rnr_dir='/mobileye/GT_FPA/yehonatans/temp/rnr_out/rnrObj/', ignore_car_body=True, NSS=True):
    if NSS:
        clip_dir = os.path.join(nss_dir, camera_name, 'hl_preds', clip_name)
        fp = os.path.join(clip_dir, str(gi) + '.hdf')
    else:
        clip_dir = os.path.join(rnr_dir, camera_name, 'segmentation', clip_name)
        fp = os.path.join(clip_dir, str(gi)+'.hdf')
    f = h5py.File(fp, 'r')
    hl = np.array(f[str(gi)]).astype(np.int8)

    ignore_fp = os.path.join(nss_dir, camera_name, 'ignore_masks', clip_name + '.hdf')
    f = h5py.File(ignore_fp, 'r')
    ignore_mask = np.array(f[clip_name])

    if ignore_car_body and 'corner' in camera_name.lower():
        car_body_dir = '/mobileye/algo_STEREO3/stereo/data/data_eng/car_body_masks/'
        if 'alfred' in clip_name.lower():
            car = 'Alfred'
        elif 'bella' in clip_name.lower():
            car = 'Bella'
        elif 'oct' in clip_name.lower():
            car = 'OCT'
        elif 'krq' in clip_name.lower():
            car = 'KRQ'
        else:
            raise ValueError('Car Body Mask not found')
        body_masks = np.load(os.path.join(car_body_dir, car + '.npz'))
        body_mask = ~body_masks[camera_name].astype(np.bool)
        ignore_mask = np.logical_or(ignore_mask, body_mask)

        if not NSS:
            ignore_fp = os.path.join(nss_dir, camera_name, 'ignore_masks', clip_name + '.hdf')
            f = h5py.File(ignore_fp, 'r')
            ignore_mask = np.logical_or(ignore_mask, np.array(f[clip_name]))

    return hl, ignore_mask


def get_seg_mask(hl, road=False, transition=False, sidewalk=False, vd=False, peds=False, NSS=True):

    mask = np.zeros_like(hl)
    if NSS:
        # NSS_Voting
        if road:
            mask = np.logical_or.reduce((mask, hl==1, hl==4))
        if transition:
            mask = np.logical_or.reduce((mask, hl==2, hl==3))
        if sidewalk:
            mask = np.logical_or.reduce((mask, hl==5, hl==6, hl==3))
        if vd:
            mask = np.logical_or(mask, hl==7)
        if peds:
            mask = np.logical_or.reduce((mask, hl==9, hl==8))
    else:
        # RnR
        if road:
            mask = np.logical_or(mask, hl==2)
        if transition:
            mask = np.logical_or(mask, hl==3)
        if sidewalk:
            mask = np.logical_or.reduce((mask, hl==3, hl==4))
        if vd:
            mask = np.logical_or(mask, hl==5)
        if peds:
            mask = np.logical_or.reduce((mask, hl==6, hl==7))

    return mask
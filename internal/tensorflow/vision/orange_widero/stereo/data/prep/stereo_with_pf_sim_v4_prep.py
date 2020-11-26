import numpy as np
from stereo.data.prep.prep_utils import get_center_view, get_surround_views


def prep_frame(dataSetIndex, frame, view_names, fail_missing_lidar=True, inferenceOnly=False, views=None):
    if views is None:
        views = dataSetIndex.read_views(frame=frame, view_names=view_names)
    cntr = get_center_view(views)
    srnd = get_surround_views(views)
    im_cntr = views[cntr]['image'] / 255.
    im_srnd = []
    T_cntr_srnd = []
    for view in srnd:
        im_srnd.append(views[view]['image'] / 255.)
        RT_srnd_to_cntr = np.matmul(np.linalg.inv(views[cntr]['RT_view_to_main']),
                                    views[view]['RT_view_to_main'])
        T_cntr_srnd.append(np.expand_dims(-RT_srnd_to_cntr[:3, 3], 0))
    focal = np.array([views[cntr]['focal'], views[cntr]['focal']]).astype(np.float32)
    origin = np.array(views[cntr]['origin']).astype(np.float32)

    ims = [im_cntr]
    ims.extend(im_srnd)
    inp = np.concatenate([im[np.newaxis, :, :] for im in ims], axis=0).astype(np.float32)
    T_cntr_srnd = np.concatenate(T_cntr_srnd, axis=0).astype(np.float32)
    if inferenceOnly:
        return [{'inp': inp, 'T_cntr_srnd': T_cntr_srnd, 'origin': origin, 'focal': focal}]

    gi = views[cntr]['grab_index'].astype(np.int32)
    clip_name = np.array(views[cntr]['clip_name'].encode('ascii','ignore'))

    view = {'inp': inp, 'sim_depth': views[cntr]['sim_depth'].astype(np.float32),
            'T_cntr_srnd': T_cntr_srnd, 'origin': origin, 'focal': focal,
            'gi': gi, 'clip_name': clip_name, 'cntr_cam_name': np.array(cntr.split('_to_')[0])}
    if 'speed' in views[cntr].keys():
        view['speed'] = views[cntr]['speed'].astype(np.float32)

    return [view]



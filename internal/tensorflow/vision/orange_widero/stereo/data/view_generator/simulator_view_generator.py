import os
import numpy as np


def is_center_view(view_name):
    if '_to_' not in view_name:
        return False
    from_cam, to_cam = view_name.split('_to_')
    return from_cam == to_cam


class SimulatorViewGenerator(object):
    def __init__(self, clip_path, view_names=[]):
        self.clip_path = clip_path
        self.clip_name = os.path.basename(clip_path)
        self.view_names = view_names

    def get_gi_views(self, gi, mask=False):
        views = {}
        masks = {view: 1 for view in self.view_names}
        if mask:
            masks_path = "/mobileye/algo_STEREO3/guygo/masks/"
            masks = {view: np.load(masks_path + "mask_%s.npy" % view) for view in self.view_names}
        for view in self.view_names:
            gi_npz_path = os.path.join(self.clip_path, view, "%s_%s_%07d.npz" %
                                       (self.clip_name, view, gi))
            gi_data = np.load(gi_npz_path)
            views[view] = {
                'clip_name': self.clip_name,
                'grab_index': gi,
                'image': gi_data['image'] * masks[view],
                'focal': gi_data['focal'],
                'origin': gi_data['origin'],
                'RT_view_to_main': gi_data['RT_view_to_main'],
            }
            if is_center_view(view):
                views[view].update({
                    'semantic_segmentation': gi_data['semantic_segmentation'],
                    'sim_depth': gi_data['sim_depth']
                })
        return views

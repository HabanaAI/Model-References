from os.path import join, exists
import numpy as np
from glob import glob

from stereo.data.clip_utils import clip_name_to_sess_name, camera_name_to_section ,clip_name_to_sections
from file2path import file2path

class PreDumpIndexV2(object):

    def __init__(self, predump_dir):
        self.predump_dir = predump_dir
        self.availble_clips = [itrk.split('/')[-1].split('.itrk')[0]
                               for itrk in glob(join(predump_dir, 'GTEM_itrks', '*itrk'))]

    def get_car_body_mask(self, front_clip_name, camera):
        """"return car body mask, in distorted level-0,  currently only works for Alfred, KRQ and OCT clips"""
        if camera not in ['frontCornerLeft', 'frontCornerRight', 'rearCornerLeft', 'rearCornerRight']:
            return None
        car_body_masks_dir = '/mobileye/algo_STEREO3/stereo/data/data_eng/car_body_masks'

        car_names = ['Alfred', 'KRQ', 'OCT']
        for car_name in car_names:
            if car_name in clip_name_to_sess_name(front_clip_name):
                car_body_masks = np.load(join(car_body_masks_dir, car_name+'.npz'))
                return 1 - car_body_masks[camera]
        return None

    def lidar_path(self, front_clip_name):
        assert front_clip_name in self.availble_clips
        return join(self.predump_dir, 'lidar', front_clip_name)

    def gtem_itrk_path(self, front_clip_name):
        assert front_clip_name in self.availble_clips
        itrk_path = join(self.predump_dir, 'GTEM_itrks', front_clip_name + '.itrk')
        if not exists(itrk_path):
            itrk_path = file2path(itrk_path)
        return itrk_path

    def mest_itrk_path(self, front_clip_name, camera_name):

        section = camera_name_to_section[camera_name]
        clips_dict = clip_name_to_sections(front_clip_name)
        itrk_path = join(self.predump_dir, 'itrks', clips_dict[section] + '.itrk.gz')
        if not exists(itrk_path):
            itrk_path = file2path(itrk_path)
        return itrk_path

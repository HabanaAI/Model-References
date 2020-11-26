import pickle
from tqdm import tqdm
from glob import glob
from os import listdir
from os.path import exists, join, isdir, basename, split
import numpy as np

from stereo.data.clip_utils import sess_path_has_velodyne, clip_name_to_sess_path, clip_name_to_sess_name, clip_name_to_clips_dict
from file2path import file2path

class PreDumpIndex(object):

    def __init__(self, predump_dir, rebuild_index=False, save_index=False):
        self.camera_name_to_section = {'main': 'Front', 'rear': 'Rear',
                                       'frontCornerLeft': 'Left', 'frontCornerRight': 'Right',
                                       'rearCornerLeft': 'Left', 'rearCornerRight': 'Right',
                                       'parking_left': 'Left', 'parking_right': 'Right'}
        self.camera_names = self.camera_name_to_section.keys()
        self.sections = list(set(self.camera_name_to_section.values()))
        self.predump_dir = predump_dir

        if not rebuild_index:
            assert (not save_index)
            self.load_index()
        else:
            self.rebuild_index()
            if save_index:
                self.save_index()

    def save_index(self):
        index_path = self.predump_dir + '/predump_index.pickle'
        with open(index_path, 'wb') as f:
            data = {'valid_front_clips': self.valid_front_clips,
                    'valid_front_clips_with_lidar': self.valid_front_clips_with_lidar,
                    'front_clip_sessions': self.front_clip_sessions}

            data['sessions_subdir'] = self.sessions_subdir
            data['sessions'] = self.sessions
            data['front_clip_to_sections'] = self.front_clip_to_sections
            pickle.dump(data, f)

    def load_index(self):
        index_path = self.predump_dir + '/predump_index.pickle'
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
        self.valid_front_clips = data['valid_front_clips']
        self.valid_front_clips_with_lidar = data['valid_front_clips_with_lidar']
        self.front_clip_sessions = data['front_clip_sessions']
        try:
            self.sessions = data['sessions']
        except:
            predump_entries = glob(self.predump_dir + '/*')
            self.sessions = [basename(e) for e in predump_entries if isdir(join(self.predump_dir, e))]

        self.sessions_subdir = data['sessions_subdir']
        self.front_clip_to_sections = data['front_clip_to_sections']

    def rebuild_index(self):
        self.sessions = []
        self.sessions_subdir = {}
        DMPS = [dmp for dmp in listdir(self.predump_dir) if 'DMP' in dmp]
        for DMP in DMPS:
            sessions_dir = [item for item in listdir(join(self.predump_dir, DMP, 'BatchCam2Cam')) if
                            isdir(join(self.predump_dir, DMP, 'BatchCam2Cam', item)) and 'nb_logs' not in item]
            sessions = []
            for sess in sessions_dir:
                try:
                    clipext_content = listdir(join(self.predump_dir, DMP, 'BatchCam2Cam', sess, 'clipext'))
                    for f in clipext_content:
                        if 'meta' in f:
                            clip_name = f.split('.')[0]
                            sess_name = clip_name_to_sess_name(clip_name)
                            sessions.append(sess_name)
                            self.sessions_subdir[sess_name] = (DMP, sess)
                            break
                except:
                    continue
            self.sessions.extend(sessions)

        self.valid_front_clips = []
        self.valid_front_clips_with_lidar = []
        self.front_clip_sessions = {}
        self.front_clip_to_sections = {}
        for s in tqdm(self.sessions):
            if not exists(join(self.predump_dir, self.sessions_subdir[s][0], 'BatchCam2Cam', self.sessions_subdir[s][1], 'solvesurroundcalib/etc/camera.conf')):
                continue
            clipext_content = listdir(join(self.predump_dir, self.sessions_subdir[s][0], 'BatchCam2Cam', self.sessions_subdir[s][1], 'clipext'))
            sess_front_clip_names = [item.split('.')[0] for item in clipext_content if 'Fron' in item and 'meta' in item]
            sess_has_velodyne = False  # check if the first clip in this session leads to a Velodyne dir
            if len(sess_front_clip_names) > 0:
                if sess_path_has_velodyne(clip_name_to_sess_path(sess_front_clip_names[0])):
                    sess_has_velodyne = True
            for c in sess_front_clip_names:
                all_sections = True
                clips_dict = clip_name_to_clips_dict(c)
                full_clips_dict = True
                for section in self.sections:
                    full_clips_dict = full_clips_dict and section in clips_dict.keys()
                if not full_clips_dict:
                    continue
                for section in self.sections:
                    section_exist = exists(file2path(join(self.predump_dir, self.sessions_subdir[s][0], 'Brain/itrk', clips_dict[section]+'.itrk.gz')))
                    all_sections = all_sections and section_exist
                all_cameras = True
                for camera_name in ['main']:  # in self.camera_names
                    section = self.camera_name_to_section[camera_name]
                    gtem_exist = exists(join(self.predump_dir, self.sessions_subdir[s][0],
                                                       'BatchCam2Cam', self.sessions_subdir[s][1],
                                                       'creategtemforcamera', camera_name, clips_dict[section]+'.itrk'))

                    all_cameras = all_cameras and gtem_exist
                if all_sections and all_cameras:
                    self.valid_front_clips.append(c)
                    self.front_clip_sessions[c] = s
                    self.front_clip_to_sections[c] = clips_dict
                    if sess_has_velodyne:
                        self.valid_front_clips_with_lidar.append(c)

    @staticmethod
    def front_clip_path_to_section(clip_path, section):
        base, clip_file = split(clip_path)
        suffix = clip_file[-13:] if clip_file.endswith('.itrk.gz') else clip_file[-10:]
        section_file = glob(base + '/*_' + section + suffix)
        if len(section_file) != 1:
            return None
        return section_file[0]

    def get_car_body_mask(self, front_clip_name, camera):
        """"return car body mask, in distorted level-0,  currently only works for Alfred, KRQ and OCT clips"""
        if camera not in ['frontCornerLeft', 'frontCornerRight', 'rearCornerLeft', 'rearCornerRight']:
            return None
        car_body_masks_dir = '/mobileye/algo_STEREO3/old_stereo/data/data_eng/car_body_masks'
        car_names = ['Alfred', 'KRQ', 'OCT']
        for car_name in car_names:
            if car_name in self.front_clip_sessions[front_clip_name]:
                car_body_masks = np.load(join(car_body_masks_dir, car_name+'.npz'))
                return 1 - car_body_masks[camera]
        return None

    def etc_dir(self, front_clip_name):
        if front_clip_name not in self.valid_front_clips:
            return None
        session = self.front_clip_sessions[front_clip_name]
        return join(self.predump_dir, self.sessions_subdir[session][0], 'BatchCam2Cam', self.sessions_subdir[session][1],
             'solvesurroundcalib/etc')
        return join(self.predump_dir, session, 'etc')

    def lidar_path(self, front_clip_name):

        gtem_main_path = self.gtem_itrk_path(front_clip_name, 'main')

        lidar_base = '/'.join(gtem_main_path.split('/')[:-4])
        return file2path(join(lidar_base, 'lidar', front_clip_name))

    def mest_itrk_path(self, front_clip_name, camera_name):
        if front_clip_name not in self.valid_front_clips:
            return None

        session = self.front_clip_sessions[front_clip_name]
        section = self.camera_name_to_section[camera_name]
        clips_dict = self.front_clip_to_sections[front_clip_name]
        return file2path(join(self.predump_dir, self.sessions_subdir[session][0], 'Brain/itrk', clips_dict[section] + '.itrk.gz'))


    def gtem_itrk_path(self, front_clip_name, camera_name):
        if front_clip_name not in self.valid_front_clips:
            return None
        session = self.front_clip_sessions[front_clip_name]
        section = self.camera_name_to_section[camera_name]
        clips_dict = self.front_clip_to_sections[front_clip_name]
        return join(self.predump_dir, self.sessions_subdir[session][0],
                     'BatchCam2Cam', self.sessions_subdir[session][1],
                     'creategtemforcamera', camera_name, clips_dict[section] + '.itrk')



if __name__ == '__main__':
    predump_dir = '/mobileye/algo_STEREO3/stereo/data/data_eng/'
    predump = PreDumpIndex(predump_dir, rebuild_index=True, save_index=True)
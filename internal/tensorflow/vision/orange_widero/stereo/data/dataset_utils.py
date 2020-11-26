import gc
import os
import numpy as np
import traceback

import pickle
from os import listdir
from os.path import isdir
from multiprocessing.pool import ThreadPool as Pool
import itertools
try:
    from tqdm import tqdm
except:
    def tqdm(l):
        return l

from os.path import exists, join
from random import sample, shuffle
from time import time
import sys
from file2path import file2path


def my_file2path(path):
    if exists(path):
        return path
    return file2path(path)
from stereo.data.clip_utils import clip_name_to_sess_name


class DatasetIndex(object):
    def __init__(self, dataset_dir, index_path=None, rebuild_index=False, save_index=False,
                 test_percentage=0.1, preserve_tt_sessions=None, split_by_clip=False):
        self.dataset_dir = dataset_dir
        self.index_path = join(dataset_dir, 'dataset_index.pickle') if index_path is None else index_path
        self.train_frames_list = []
        self.test_frames_list = []
        self.train_sessions = []
        self.test_sessions = []
        if not rebuild_index:
            self.load_dataset_index()
        else:
            self.rebuild_index(test_percentage, preserve_tt_sessions, split_by_clip)
            shuffle(self.test_frames_list)
            shuffle(self.train_frames_list)
            if save_index:
                print ('rebuild index and save to %s' % self.index_path)
                self.save()
        print ('dataset contains %d train examples and %d test examples' %
               (len(self.train_frames_list), len(self.test_frames_list)))

    def load_dataset_index(self):
        with open(self.index_path, 'rb') as f:
            if sys.version_info.major > 2:
                data = pickle.load(f, encoding='latin1')
            else:
                data = pickle.load(f)
        self.test_frames_list = data['test_frames_list']
        self.train_frames_list = data['train_frames_list']
        return data

    def shuffle(self):
        shuffle(self.test_frames_list)
        shuffle(self.train_frames_list)
        self.save()
        print ('shuffle frames lists and save to %s' % self.index_path)

    def save(self):
        with open(self.index_path, 'wb') as f:
            pickle.dump({'test_frames_list': self.test_frames_list,
                         'train_frames_list': self.train_frames_list}, f)

    def rebuild_index(self, test_percentage, preserve_tt_sessions, split_by_clip):
        raise NotImplementedError

    @staticmethod
    def load_tt_from_index(index_dir):
        if exists(join(index_dir, 'dataset_index.pickle')):
            with open(join(index_dir, 'dataset_index.pickle'), 'rb') as f:
                data = pickle.load(f)
                return data['train_sessions'], data['test_sessions']
        else:
            raise Exception("Dataset not exists")


class TopViewDatasetIndex(DatasetIndex):
    def rebuild_index(self, test_percentage, preserve_tt_sessions, split_by_clip):
        clips = listdir(self.dataset_dir)
        clips = [clip for clip in clips if clip != "nb_logs"]
        if preserve_tt_sessions is not None:
            train_sessions, test_sessions = DatasetIndex.load_tt_from_index(preserve_tt_sessions)
            self.train_sessions += train_sessions
            self.test_sessions += test_sessions
        for clip_name in clips:
            clips_gis = listdir(join(self.dataset_dir, clip_name))
            clips_gis = [int(gi[:-4]) for gi in clips_gis]  # remove .npy extension
            formatted_gis = ["%s@%s" % (clip_name, gi) for gi in clips_gis]
            if split_by_clip:
                if np.random.random() < test_percentage:
                    self.test_frames_list += formatted_gis
                else:
                    self.train_frames_list += formatted_gis
                continue
            clip_session = clip_name_to_sess_name(clip_name)
            if clip_session in self.train_sessions:
                self.train_frames_list += formatted_gis
            elif clip_session in self.test_sessions:
                self.test_frames_list += formatted_gis
            else:
                if np.random.random() < test_percentage:
                    self.test_sessions.append(clip_session)
                    self.test_frames_list += formatted_gis
                else:
                    self.train_sessions.append(clip_session)
                    self.train_frames_list += formatted_gis


class ViewDatasetIndex(DatasetIndex):
    def __init__(self, dataset_dir, index_path=None, rebuild_index=False, save_index=False,
                 test_percentage=0.1, preserve_tt_sessions=None):
        super(ViewDatasetIndex, self).__init__(dataset_dir, index_path, rebuild_index, save_index,
                                               test_percentage, preserve_tt_sessions)
        self.clip_gis = None
        self.clip_gis_speed = None

    def load_dataset_index(self):
        data = DatasetIndex.load_dataset_index(self)
        self.clip_gis = data['clip_gis']
        self.train_sessions = data['train_sessions']
        self.test_sessions = data['test_sessions']
        if 'clip_gis_speed' in data.keys():
            self.clip_gis_speed = data['clip_gis_speed']

    def save(self):
        with open(self.index_path, 'wb') as f:
            pickle.dump({'test_frames_list': self.test_frames_list,
                         'train_frames_list': self.train_frames_list,
                         'test_sessions': self.test_sessions,
                         'train_sessions': self.train_sessions,
                         'clip_gis': self.clip_gis}, f)

    def rebuild_index(self, test_percentage, preserve_tt_sessions, split_by_clip):
        rebuild_start = time()
        print ('ls dataset_dir ')
        t = time()
        f2p_list = [join(self.dataset_dir, item) for item in listdir(self.dataset_dir)
                    if isdir(join(self.dataset_dir, item)) and len(item) == 3]
        elapsed = time() - t
        print ('ls dataset_dir took %d seconds' % np.int64(np.floor(elapsed)))

        print ('extract sessions and gis')
        clip_session = {}
        clip_gis = {}

        def map_func(dirname):
            return [item for item in listdir(dirname) if item.endswith('.npz')]

        views_list = Pool(processes=32).map(map_func, f2p_list)
        views_list = list(itertools.chain.from_iterable(views_list))
        for view in tqdm(views_list):
            if 'parking' in view or 'small' in view:
                clip = '_'.join(view.split('_')[:-5])
            else:
                clip = '_'.join(view.split('_')[:-4])
            gi = np.int64(view.split('_')[-1].split('.')[0])
            if clip not in clip_gis.keys():
                clip_gis[clip] = [gi]
                sess = clip_name_to_sess_name(clip)
                clip_session[clip] = sess
            else:
                if gi not in clip_gis[clip]:
                    clip_gis[clip].append(gi)

        sessions = sorted(list(set(clip_session.values())))
        clips = clip_session.keys()
        if preserve_tt_sessions:
            print ('preserving train-test sessions from %s' % join(preserve_tt_sessions, 'dataset_index.pickle'))
            train_sessions, test_sessions = DatasetIndex.load_tt_from_index(preserve_tt_sessions)
            new_sessions = [sess for sess in sessions if sess not in test_sessions and sess not in train_sessions]
            new_test_sessions = sample(new_sessions, np.int64(np.floor(len(new_sessions)) * test_percentage))
            new_train_sessions = [sess for sess in new_sessions if sess not in new_test_sessions]
            self.test_sessions.extend(test_sessions)
            self.test_sessions.extend(new_test_sessions)
            self.train_sessions.extend(train_sessions)
            self.train_sessions.extend(new_train_sessions)
        else:
            self.test_sessions = sample(sessions, np.int64(np.ceil(len(sessions) * test_percentage)))
            self.train_sessions = [sess for sess in sessions if sess not in self.test_sessions]

        for clip in clips:
            isTrain = clip_session[clip] in self.train_sessions
            if not isTrain:
                assert clip_session[clip] in self.test_sessions
            for gi in clip_gis[clip]:
                if isTrain:
                    self.train_frames_list.append('%s@gi@%07d' % (clip, gi))
                else:
                    self.test_frames_list.append('%s@gi@%07d' % (clip, gi))
        self.clip_gis = clip_gis
        elapsed = time() - rebuild_start
        print ('rebuild took %d seconds' % np.int64(np.floor(elapsed)))

    def read_views(self, view_names, frame, old_format=False):
        if '@gi@' in frame:
            clip_name, gi = frame.split('@gi@')
        elif '@' in frame:
            clip_name, gi = frame.split('@')
        else:
            print('Wrong frame string format')
            assert(False)
        views = read_views(self.dataset_dir, view_names, clip_name, np.int64(gi), old_format=old_format)
        if self.clip_gis_speed:
            speed = self.clip_gis_speed[clip_name][np.int64(gi)]
            for view in views.keys():
                views[view]['speed'] = speed
        return views


def read_view(npz):
    if sys.version_info.major > 2:
        npz_view = np.load(my_file2path(npz), allow_pickle=True, encoding='latin1')
    else:
        npz_view = np.load(my_file2path(npz), allow_pickle=True)
    view = {}
    for key in npz_view.files:
        view[key] = np.expand_dims(npz_view[key],0)[0]
    return view


def read_views(npz_dir, view_names, clip_name, gi, old_format=False):
    if not view_names or (len(view_names) == 1 and view_names[0] == ""):
        if old_format:
            return read_view(os.path.join(npz_dir, "%s_%06d.npz" % (clip_name, gi)))
        else:
            return read_view(os.path.join(npz_dir, "%s_%07d.npz" % (clip_name, gi)))
    views = {}
    for view_name in view_names:
        if old_format:
            views[view_name] = read_view(os.path.join(npz_dir, "%s_%s_%06d.npz" % (clip_name, view_name, gi)))
        else:
            views[view_name] = read_view(os.path.join(npz_dir, "%s_%s_%07d.npz" % (clip_name, view_name, gi)))
    return views


def write_view_dataset(clip_name, part_ind, part_num, view_names, predump, out_dir, verbosity=0, test_run=False,
                       must_have_gtd=True):
    from stereo.data.view_generator.view_generator import ViewGenerator

    if verbosity > 0:
        print('Processing clip %s' % clip_name)
    try:
        vds = ViewGenerator(clip_name, view_names.split(','), predump=predump, must_have_gtd=must_have_gtd)
    except Exception:
        if verbosity > 0:
            traceback.print_exc()
            print('Failed while constructing a ViewDataset\n')
        return 1

    full_gis = vds.gtem_main['data']['gis'][2:]  # Skip first gtem gis so get_unrolled_lidar() doesn't fail
    if test_run:
        full_gis = full_gis[:10]
    part_gis = np.array_split(full_gis, part_num)
    curr_part_gis = part_gis[part_ind]

    for gi in curr_part_gis:
        try:
            gi_views = vds.get_gi_views(gi)
        except Exception:
            if verbosity > 0:
                traceback.print_exc()
                print('Failed reading gi: %d' % gi)
            continue
        for view_name, view in gi_views.items():
            out_fn = "%s_%s_%07d.npz" % (vds.clip_name, view_name, gi)
            try:
                out_path = file2path(out_fn, out_dir, create=True)
            except OSError as err:
                print (err)
            np.savez_compressed(out_path, **view)
        gc.collect()  # Added to be safer. The critical gc.collect() call is in lidar_utils and get_unrolled_lidar
        if verbosity > 0:
            print('Extracted views for gi: %d' % gi)
    if verbosity > 0:
        print('Done\n')
    return 0

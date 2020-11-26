import os
import numpy as np
import json
from stereo.common.gt_packages_wrapper import get_fof_from_clip

from stereo.data.frame_utils import crop_scale_meimage_to_arr
from stereo.data.union_masks import UnionVDPDMasks
from stereo.data.view_image_input import ViewImageInput
from stereo.data.gtem_utils import load_gtem_data, gi_to_gi_RT

from stereo.data.sky_seg_utils import filter_sky_lidar_image
from stereo.data.label_utils import Labeler
from stereo.data.lidar_dumper import LidarReader
from stereo.data.clip_utils import init_fof_lidar_try, get_transformation_matrix
from stereo.data.gtem_utils import get_vehicle_accumulated_dist_gtem
from stereo.data.lidar_utils import LidarProcessor

from devkit.clip import MeClip


class ViewGenerator(object):

    def __init__(self, clip_name, view_names, etc_dir=None, meta_dir=None, predump=None, mode='train',
                                                    mono=False, try_creating_lidar=False, clip=None,
                                                    must_have_gtd=True, create_pixel_labels=False):

        self.clip_name = clip_name
        self.view_names = view_names
        self.etc_dir = etc_dir if etc_dir is not None else None
        self.predump = predump
        self.mode = mode
        assert (mode in ['train', 'eval', 'pred'])
        self.mono = mono

        if mono:
            self.clip = MeClip(clip_name)
            self.sub_clip = self.clip
        else:

            if clip is None:
                self.clip = get_fof_from_clip(clip_name, etc=self.etc_dir, compact=True, meta_dir=meta_dir)
            else:
                self.clip = clip
            self.clip.init_ts_map()
            self.sub_clip = MeClip(clip_name)

        # load the view configurations and initialize the following members
        self.views = None
        self.camera_names_to = None
        self.camera_names_from = None
        self.gtem = None
        self.gtem_main = None
        self.cum_dist_by_gi = None
        self.vdpd_masks = None
        self.has_lidar = None
        self.lidar_reader = None
        self.lidar_procs = None
        self.try_creating_lidar = try_creating_lidar
        self.load_views_conf(view_names, must_have_gtd, create_pixel_labels)

    def load_views_conf(self, view_names, must_have_gtd=True, create_pixel_labels=False):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.views = {}
        self.camera_names_to = set()
        self.camera_names_from = set()
        model_names = set()
        someone_wants_lidar = False
        for view_name in view_names:
            with open(os.path.join(module_dir, '../../data/views', view_name + '.json')) as f:
                conf = json.load(f)
            car_body_mask = None if not self.predump else \
                self.predump.get_car_body_mask(self.clip_name, conf['camera_from'])
            self.views[view_name] = {'conf': conf,
                                     'image_input': ViewImageInput(self.clip, conf['camera_from'],
                                                                   conf['camera_to'], conf['level'],
                                                                   conf['undistort'], conf['crop_rect_at_level'],
                                                                   car_body_mask)}
            self.camera_names_to.add(conf['camera_to'])
            self.camera_names_from.add(conf['camera_from'])
            someone_wants_lidar = someone_wants_lidar or len(conf['lidars']) > 0
            model_names.update(conf['labels']) # collect all models required by some view

        if create_pixel_labels: # initiate an object for segmentation masks
            self.labeler = Labeler(model_names)
        else:
            self.labeler = None

        # gtem and masks are only available in train and eval modes
        if self.mode == 'train' or self.mode == 'eval':
            # separately load gtem for main
            gtem_main_path = self.predump.gtem_itrk_path(self.clip_name)
            gtem_main_data = load_gtem_data(gtem_main_path, self.clip, cam='main')
            self.gtem_main = {'path': gtem_main_path, 'data': gtem_main_data}

            if someone_wants_lidar:
                try:
                    self.lidar_reader = LidarReader(self.clip_name, self.predump)
                except:
                    print("Failed to instantiate LidarReader")
                    assert False

            # use gtem_main to compute accumulated arc length of main camera
            self.cum_dist_by_gi = get_vehicle_accumulated_dist_gtem(self.gtem_main['data'])  # TODO: support mono

            # load per view gtem_data
            self.gtem = {}
            for view_name in view_names:
                gtem_target_cam = self.views[view_name]['conf']['camera_to']
                gtem_path = self.predump.gtem_itrk_path(self.clip_name)
                gtem_data = load_gtem_data(gtem_path, self.clip, cam='main', target_cam=gtem_target_cam)
                self.gtem[view_name] = {'path': gtem_path, 'data': gtem_data}

            vdpd_list = []
            vdpd_list.extend(self.camera_names_to)
            self.vdpd_masks = {}
            for camera_name_to in self.camera_names_to:
                itrk_path = self.predump.mest_itrk_path(self.clip_name, camera_name_to)
                self.vdpd_masks[camera_name_to] = UnionVDPDMasks(self.clip, self.clip_name, camera_name_to, itrk_path,
                                                                 must_have_itrk=False, must_have_gtd=must_have_gtd)

        # construct per view lidar processors
        if self.try_creating_lidar or self.mode == 'train' or self.mode == 'eval':
            self.has_lidar = init_fof_lidar_try(self.clip)
            if self.has_lidar:
                self.lidar_procs = {}
                for view_name in view_names:
                    lidar_target_cam = self.views[view_name]['conf']['camera_to']
                    gtem = None if self.gtem is None else self.gtem[view_name]['data']
                    vdpd_masks = None if self.vdpd_masks is None else self.vdpd_masks[lidar_target_cam]
                    self.lidar_procs[view_name] = \
                        LidarProcessor(clip=self.clip, clip_name=self.clip_name, gtem=gtem,
                                       vdpd_masks=vdpd_masks, camera_name=lidar_target_cam,
                                       lidar_reader=self.lidar_reader)

    def surround_sync_valid(self, gi, gi_frameid_must_match):
        main_utc_ts = self.clip.get_utc_ts(grab_index=gi, cam='main')
        frameId = self.clip.ts_map.frame_id_by_gi(gi, 'main')
        valid = True
        for camera_name in self.camera_names_to.union(self.camera_names_from):
            if gi_frameid_must_match:
                valid &= (frameId == self.clip.ts_map._ts_dict[camera_name].frame_id_by_gi(gi, camera_name))
            valid &= (abs(self.clip.get_utc_ts(frame_id=frameId, cam=camera_name) - main_utc_ts) < 65)
        return valid, frameId

    def get_gi_ts(self, gi, camera_name):  # TODO: this should also be supported in mono
        return self.clip.get_utc_ts(grab_index=gi, cam='main') if not self.mono else None

    def get_gi_views(self, gi, RT_modification=None, modify_views=[]):
        """
        Get data (e.g. camera images and Lidar) for views of ViewGenerator. Options are saved in the conf files in
        the 'views' folder.
        :param gi:
        :param RT_modification: 4x4 matrix to modify the calibration to view. Default - no modification (= true calibration)
        :param modify_views: list of views to modify their calibration according to `RT_modifications`
        :return:
        :rtype:
        """
        frame_id = None
        if not self.mono:
            sync, frame_id = self.surround_sync_valid(gi, self.mode == 'train')
            assert sync
        gi_views = {}
        for view_name in self.views:
            conf = self.views[view_name]['conf']
            cam_from = conf['camera_from']
            cam_to = conf['camera_to']

            tone_map_meim, K, RT_cam_to_view = \
                self.views[view_name]['image_input'].get_frame(gi, frame_id=frame_id, tone_map=conf['tone_map'])
            tone_maps_extra_meims = {}
            for tone_map in (conf['tone_maps_extra'] if 'tone_maps_extra' in conf else []):
                tone_maps_extra_meims[tone_map] = \
                    self.views[view_name]['image_input'].get_frame(gi, frame_id=frame_id, tone_map=tone_map)[0]
            car_body_meimage = self.views[view_name]['image_input'].get_car_body_mask_meimage()

            if view_name in modify_views:
                RT_cam_to_main = get_transformation_matrix(self.clip, cam_from, 'main', RT_modification=RT_modification)
            else:
                RT_cam_to_main = get_transformation_matrix(self.clip, cam_from, 'main')
            RT_view_to_cam = np.linalg.inv(RT_cam_to_view)
            RT_view_to_main = RT_cam_to_main.dot(RT_view_to_cam)
            RT_view_to_clip = []
            if self.mode == 'train' or self.mode == 'eval':
                RT_main_to_clip = gi_to_gi_RT(self.gtem[view_name]['data'], gi, self.gtem[view_name]['data']['gis'][0])
                RT_view_to_clip = RT_main_to_clip.dot(RT_cam_to_main.dot(RT_view_to_cam))

            tone_map_im, focal, origin = \
                crop_scale_meimage_to_arr(tone_map_meim, K, conf['scale'], conf['crop_rect_at_level'],
                                          dtype='uint8')  # origin: (x,y)
            tone_maps_extra_ims = {}
            for tone_map in (conf['tone_maps_extra'] if 'tone_maps_extra' in conf else []):
                tone_maps_extra_ims[tone_map] = \
                    crop_scale_meimage_to_arr(tone_maps_extra_meims[tone_map], K, conf['scale'],
                                              conf['crop_rect_at_level'], dtype='uint8')[0]
            car_body_image, _, _ = \
                crop_scale_meimage_to_arr(car_body_meimage, K, conf['scale'], conf['crop_rect_at_level'], dtype='uint8')

            masks, lidars, lidar_clusters, labels = {}, {}, {}, {}
            masks['car_body'] = car_body_image

            if self.mode == 'train' or self.mode == 'eval':
                for mask_name in conf['masks']:
                    masks[mask_name] = self.vdpd_masks[cam_to].get_mask_image(gi, cam_to, tone_map_im.shape, focal,
                                                                              origin,
                                                                              mask_name, conf['undistort'] == 1)

            if self.lidar_procs is not None:
                for lidar in conf['lidars']:
                    lidar_range = np.arange(lidar['range'][0], lidar['range'][1])
                    lidars[lidar['name']], lidar_clusters[lidar['name']] = \
                        self.lidar_procs[view_name].get_lidar_image(gi, lidar_range, tone_map_im.shape,
                                                                focal, origin, lidar, conf['undistort'])

            if self.labeler is not None:
                # Generate segmentation mask(s) for image:
                for model in conf['labels']:
                    labels[model] = {}
                    labels[model]['segmentation_mask'], labels[model]['classes'] = \
                        self.labeler.generate_segmentation(model, tone_map_im)

                # Filter Lidar missing points according to 'sky' segmentation:
                for lidar_conf in conf['lidars']:

                    # temporary: if 'filter_missing_lidar_with_sky' flag doesn't exist:
                    if 'filter_missing_lidar_with_sky' not in lidar_conf.keys():
                        lidar_conf['filter_missing_lidar_with_sky'] = 0

                    if (lidar_conf['filter_missing_lidar_with_sky']) and lidars: # an empty "lidars" dict gives False
                        lidars[lidar_conf['name']] = filter_sky_lidar_image(labels,
                                                            lidars[lidar_conf['name']], erosion=True)

            gi_views[view_name] = {
                'clip_name': self.clip_name,
                'grab_index': gi,
                'speed': self.clip.get_frame(grab_index=gi, only_meta=True)[0]['meta']['speed'],
                'image': tone_map_im,
                'images_extra': tone_maps_extra_ims,
                'focal': focal,
                'origin': origin,
                'RT_view_to_clip': RT_view_to_clip,
                'RT_view_to_main': RT_view_to_main,
                'main_ts': self.get_gi_ts(gi, 'main'),
                'view_ts': self.get_gi_ts(gi, cam_from),
                'cum_dist': self.cum_dist_by_gi.get(gi, None) if self.cum_dist_by_gi else None,
                'masks': masks,
                'lidars': lidars,
                'lidar_clusters': lidar_clusters,
                'labels': labels
            }
        return gi_views

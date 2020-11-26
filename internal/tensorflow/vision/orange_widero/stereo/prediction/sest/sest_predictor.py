import os

from stereo.prediction.predictor import Predictor
from stereo.prediction.vidar.pred_version import get_predictors_and_views_name
from stereo.prediction.vidar.pred_utils import read_conf
from stereo.interfaces.implements import load_dataset_attributes
from stereo.models.map_func_loaders import load_map_funcs
from stereo.data.sest.top_view_generator import generate_top_view_input, save_top_view_img, VIDAR_TOP_VIEW_PATH
from stereo.data.view_generator.view_generator import ViewGenerator


class SestPredictor(Predictor):
    def __init__(self, model_json_path, checkpoint_path=None, restore_iter=-1,
                 single_core=False, from_meta=True, load_loss=False):
        self.dataset = None
        self.ds_version = None
        Predictor.__init__(self, model_json_path, checkpoint_path, restore_iter, single_core,
                           from_meta, load_loss)
        self.stereo_predictors = None
        self.views_names = None
        self.vg_cahce = {}

    def get_dataset_attributes(self, **kwargs):
        self.dataset = self.model_conf['data_params']['datasets'].values()[0]
        self.ds_version = self.model_conf['data_params'].get("ds_version", self.dataset.split('_')[-1])
        return load_dataset_attributes(self.dataset)[0]

    def get_map_funcs(self, load_loss, **kwargs):
        arch_map_func, loss_map_func, _, _ = load_map_funcs(self.model_conf, self.dataset, load_loss=load_loss,
                                                            load_eval=False, pred_mode=True)
        return arch_map_func, loss_map_func

    def get_vidar_preds_cache(self, clip_name):
        if not self.stereo_predictors:
            conf = read_conf('')
            self.stereo_predictors, self.views_names = get_predictors_and_views_name(conf)
        if clip_name not in self.vg_cahce:
            self.vg_cahce[clip_name] = ViewGenerator(clip_name, view_names=self.views_names, mode='pred')
        return self.vg_cahce[clip_name], self.stereo_predictors, self.views_names

    def run_prep_func(self, clip_name, gi):
        args_dict = {"old_data_format": self.old_format_version} if self.old_format_version else {}
        args_dict['is_pred'] = True
        try:
            return self.prep_("%s@%s" % (clip_name, gi), **args_dict)
        except IOError as e:
            print("Found no top view input image on disk, generating it...")
            import errno
            if e.errno == errno.ENOENT:
                vg, predictors, views_names = self.vg_cahce.get(clip_name), self.stereo_predictors, self.views_names
                top_view_img, self.vg_cahce[clip_name], self.stereo_predictors, self.views_names = \
                    generate_top_view_input(clip_name, gi, self.ds_version, vg=vg,
                                            predictors=predictors, views_names=views_names)
                save_top_view_img(os.path.join(VIDAR_TOP_VIEW_PATH,self.ds_version), clip_name, gi, top_view_img)
                return self.prep_("%s@%s" % (clip_name, gi), **args_dict)
            else:
                raise e

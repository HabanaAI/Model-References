from stereo.prediction.predictor import Predictor
from stereo.interfaces.implements import load_dataset_attributes
from stereo.models.map_func_loaders import load_map_funcs


class StereoPredictor(Predictor):
    def __init__(self, model_json_path, checkpoint_path=None, restore_iter=-1,
                 single_core=False, from_meta=True, load_loss=False, sector_name='main', **kwargs):
        Predictor.__init__(self, model_json_path, checkpoint_path, restore_iter, single_core, from_meta, load_loss,
                           sector_name=sector_name, **kwargs)
        self.views_names = self.dataset_attributes['views_names']

    def get_dataset_attributes(self, **kwargs):
        sector_name = kwargs['sector_name']
        return load_dataset_attributes(self.model_conf['data_params']['datasets'][sector_name])[0]

    def get_map_funcs(self, load_loss, **kwargs):
        sector_name = kwargs['sector_name']
        arch_map_func, loss_map_func, _, _ = load_map_funcs(self.model_conf, sector_name, load_loss=load_loss,
                                                            load_eval=False, load_augm=False, pred_mode=True)

        return arch_map_func, loss_map_func

    def run_prep_func(self, views, **kwargs):
        return self.prep_(dataSetIndex=None, frame=None, view_names=None, inferenceOnly=True, views=views)

import os
import argparse
from tqdm import tqdm
from random import sample
from external_import import external_import

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
with external_import(PATH):
    from stereo.common.general_utils import makedir_recursive
    from stereo.prediction.vidar.pred_utils import *
    from stereo.common.netbatch_utils import run_nb
    from stereo.data.view_generator.view_generator import ViewGenerator
    from stereo.data.view_generator.simulator_view_generator import SimulatorViewGenerator
    from stereo.prediction.vidar.stereo_predictor import StereoPredictor
    from stereo.data.clip_utils import get_fof_from_pls, get_host_box, get_clip_part_t0_grabs
    from stereo.data.clip_utils import clip_path as clip2path, is_simulator_clip


CAM_ID_DICT = {
    'main': 0,
    'frontCornerLeft': 1,
    'frontCornerRight': 2,
    'rearCornerRight': 3,
    'rearCornerLeft': 4,
    'rear': 5
}


def get_predictors_and_views_name(conf, single_core=False):
    model_conf_dir = join(tree_base(), 'stereo', 'models', 'conf', 'stereo')
    predictors = {}
    views_names = []
    sectors_conf = conf['sectors']
    from_meta = conf.get('from_meta', True)
    for sector in sectors_conf.keys():
        predictors[sector] = StereoPredictor(join(model_conf_dir, sectors_conf[sector]["model"]),
                                             single_core=single_core, sector_name=sector, from_meta=from_meta,
                                             restore_iter=sectors_conf[sector]["restore_iter"])
        views_names.extend(predictors[sector].views_names)
    return predictors, views_names


def pred_gi(predictors, vg, gi, extra_tensor_names=None, with_confidence=False, confidence_thresh=0.03):
    views = vg.get_gi_views(gi)
    pcds = {}
    pcd_attrs = {}
    extra_tensors = {}
    for k in predictors.keys():
        pcd_, pcd_attr_, _, _, extra_tensors_ = pred_views(views, predictors[k], k,
                                                           extra_tensor_names=extra_tensor_names,
                                                           with_conf=with_confidence)
        if with_confidence:
            rel_error = pcd_attr_[:, 2]
            pcd_ = pcd_[rel_error < confidence_thresh]
            pcd_attr_ = pcd_attr_[rel_error < confidence_thresh, 0]
        pcds[k] = pcd_.astype(np.float32)
        pcd_attrs[k] = pcd_attr_.astype(np.float32)
        extra_tensors[k] = extra_tensors_
    return pcds, pcd_attrs, extra_tensors


def write_gi(predictors, vg, gi, out_dir, clip_name, with_confidence=False, confidence_thresh=0.03,
             extra_tensor_names=None):
    out_fn = join(out_dir, '%s@%07d.dat' % (clip_name, gi))
    pcds, pcd_attrs, extra_tensors = pred_gi(predictors, vg, gi,
                                             with_confidence=with_confidence, confidence_thresh=confidence_thresh,
                                             extra_tensor_names=extra_tensor_names)
    cam_ids = {}
    for k, pcd_attr in pcd_attrs.items():
        cam_ids[k] = np.ones_like(pcd_attr).astype(np.float32)*CAM_ID_DICT[k]
    pcd = np.concatenate(pcds.values())
    pcd_attr = np.concatenate(pcd_attrs.values())
    cam_id = np.concatenate(cam_ids.values())
    np.c_[pcd, pcd_attr, cam_id].astype(np.float32).tofile(out_fn)


def get_vg(clip_name, views_names):
    pls_mode = clip_name.endswith('.pls')
    if is_simulator_clip(clip_name):
        vg = SimulatorViewGenerator(clip_name, views_names)
        clip_path = vg.clip_path
        clip_name_ = vg.clip_name
    elif pls_mode:
        clip_name_ = clip_name.split('/')[-1]
        clip_path = clip_name
        clip = get_fof_from_pls(clip_path)
        vg = ViewGenerator(clip.clip_name, view_names=views_names, predump=None, mode='pred', clip=clip)
    else:
        clip_name_ = clip_name
        clip_path = clip2path(clip_name_)
        vg = ViewGenerator(clip_name_, view_names=views_names, predump=None, mode='pred')
    return vg, clip_name_, clip_path


def predict_and_view(predictors, vg, gi, with_confidence=False, confidence_thresh=0.03):
    from stereo.common.visualization.vis_utils import view_pptk3d
    pcds, pcd_attrs, _ = pred_gi(predictors, vg, gi, with_confidence=with_confidence,
                                 confidence_thresh=confidence_thresh)
    host_box = get_host_box(vg.clip) if vg is ViewGenerator else None
    view_pptk3d(np.concatenate(pcds.values()), np.concatenate(pcd_attrs.values()),
                host_box=host_box, fix_coords_type=1)


def single_run(args):
    is_simulator = 'Town' in args['clip_name']
    meta_dir = args['meta_dir'] if args['meta_dir'] != '' else None
    conf = read_conf(args['version_json'])
    single_core = True if args['single_core'] != '' else False
    predictors, views_names = get_predictors_and_views_name(conf, single_core=single_core)
    vg, clip_name, clip_path = get_vg(args['clip_name'], views_names)

    if args['gi'] != '':
        print('run predict on %s, %s' % (clip_name, args['gi']))
        predict_and_view(predictors, vg, int(args['gi']), with_confidence=conf.get('with_confidence', False),
                         confidence_thresh=conf.get('confidence_thresh', 0.03))
        return 0

    makedir_recursive(args['out_dir'], clip_name)
    part_ind = 0 if args['part_ind'] == '' else int(args['part_ind'])
    part_num = 1 if args['part_num'] == '' else int(args['part_num'])

    t0_grabs = get_clip_part_t0_grabs(clip_path, meta_dir, part_ind, part_num, is_simulator)
    if args['random_gis_num'] != '':
        t0_grabs = sample(t0_grabs, min(int(args['random_gis_num']), len(t0_grabs)))
    for gi in tqdm(t0_grabs):
        try:
            write_gi(predictors, vg, gi, join(args['out_dir'], clip_name), clip_name,
                     with_confidence=conf.get('with_confidence', False),
                     confidence_thresh=conf.get('confidence_thresh', 0.03))
        except Exception as e:
            print('failed at gi %d' % gi)
            if e.__class__.__name__ == "KeyboardInterrupt":
                raise e


def parse_args():
    parser = argparse.ArgumentParser(description='run inference script')
    parser.add_argument('-l', '--clip_list_file', help='clip (Front section) list for netbatch', default='')
    parser.add_argument('-m', '--meta_dir', help='meta_dir', default='')
    parser.add_argument('-c', '--clip_name', help='clip_name', default='')
    parser.add_argument('-s', '--single_core', help='single_core', default='')
    parser.add_argument('-o', '--out_dir', help='output dir', default='')
    parser.add_argument('-j', '--version_json', help='version_json', default='')
    parser.add_argument('-i', '--part_ind', default='', help='part_ind, default: %(default)s')
    parser.add_argument('-n', '--part_num', default='', help='part_num, default: %(default)s')
    parser.add_argument('-g', '--gi', default='', help='gi for single frame prediction')
    parser.add_argument('-r', '--random_gis_num', default='', help='random_gis_num')
    parser.add_argument('-nbt', '--nb_task', help='Netbatch task name. Set this to use netbatch', default='')
    parser.add_argument('-nbq', '--nb_qslot', help='nb_qslot', default='ALGO_STEREO')
    parser.add_argument('-nbm', '--nb_mem_gb', help='Netbatch requested memory in GB (default: 8)', default='')
    return vars(parser.parse_args())


def main():
    args = parse_args()
    if args['nb_task'] != '':
        exec_path = join(tree_base(), 'stereo', 'prediction', 'vidar', 'pred_version.py')
        run_nb(args, exec_path=exec_path)
        return 0
    else:
        single_run(args)


if __name__ == '__main__':
    main()


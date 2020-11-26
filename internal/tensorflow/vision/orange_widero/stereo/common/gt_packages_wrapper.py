import os
if 'USE_GTD_API' in os.environ.keys():
    from gtd_api.common.exceptions import MissingLidar
    from gtd_api.gtd_clip.four_on_four import FoFutils
    from gtd_api.gtd_clip.pls_parser import parse_pls
    from gtd_api.gtd_clip.ts_map import NonExistingGiError
    from gtd_api.gtd_db.get_tech_data import GtdDB
    from gtd_api.gtd_db.plugins import VclPlugin, PedPlugin
    from gtd_api.GTEM.parse_em import get_integralRT, get_rots_trans, apply_rts, parse_em_data, get_partial_ts, ts2ts_RT
    from gtd_api.lidar.get_lidar import get_calib as get_l2main
    from gtd_api.lidar.lidar_consts import Semantic
    from gtd_api.utils.clip_utils import get_fof_from_clip
    from gtd_api.utils.itrk_to_df import itrk2df
    from gtd_api.utils.transformations_CG import quaternion_multiply, quaternion_from_log, quaternion_conjugate
    from gtd_api.utils.transformations import RT2params, apply_rt, dehom

    from ground_truth.objects.vehicles.state_manager import VdStateManager

else:
    from gtd_db.tech.plugins import VclPlugin, PedPlugin
    from gtd_db.tech.get_tech_data import GtdDB

    from ground_truth.common_avm.exceptions import MissingLidar
    from ground_truth.common_avm.external_libs import external_import
    from ground_truth.common_avm.four_on_four import FoFutils
    from ground_truth.common_avm.itrk_tools.itrk_to_df import itrk2df
    from ground_truth.common_avm.itrk_tools.parse_em import parse_em_data, get_partial_ts, ts2ts_RT, \
        get_integralRT, get_rots_trans, apply_rts
    from ground_truth.common_avm.pls_parser import parse_pls
    from ground_truth.common_avm.ts_map import NonExistingGiError
    from ground_truth.lidar.get_lidar import get_calib as get_l2main
    from ground_truth.lidar.lidar_consts import Semantic
    from ground_truth.objects.vehicles.state_manager import VdStateManager
    from ground_truth.utils.clip_utils import get_fof_from_clip

    if 'USE_GT_UTILS' in os.environ.keys():
        from ground_truth.utils.transformations_CG import quaternion_multiply, quaternion_from_log, quaternion_conjugate
        from ground_truth.utils.transformations import apply_rt, dehom, RT2params
    else:
        from ground_truth.scripts.transformations_CG import quaternion_multiply, quaternion_from_log, quaternion_conjugate
        from ground_truth.scripts.transformations import apply_rt, dehom, RT2params



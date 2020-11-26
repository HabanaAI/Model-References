import numpy as np
from stereo.common.gt_packages_wrapper import parse_em_data, get_partial_ts, ts2ts_RT

def load_gtem_data(gtem_itrk_path, clip, cam='main', target_cam=None, technology='gtem'):
    ret_tuple = parse_em_data(gtem_itrk_path, clip, cam=cam, target_cam=target_cam, technology=technology)
    gtem_data = dict(zip(['gis', 'RTs', 'params'], ret_tuple))
    return gtem_data


def gi_to_ts(gtem_data, gi):
    if gi not in gtem_data['params']['gis_set']:
        ts = get_partial_ts(gi, gtem_data['params'])
    else:
        ts = gtem_data['params']['tss'][gtem_data['params']['gis'].searchsorted(gi)]
    return ts


def gi_to_gi_RT(gtem_data, gi1, gi2):
    ts1 = gi_to_ts(gtem_data, gi1)
    ts2 = gi_to_ts(gtem_data, gi2)
    half_grab = 0.5 / 36
    RT12 = ts2ts_RT(ts1 + half_grab, ts2 + half_grab, gtem_data['params'], gi=False)
    return RT12


def get_vehicle_accumulated_dist_gtem(gtem_data):
    XYZ = gtem_data['params']['params'][:, 3:]
    delta_XYZ = XYZ[1:] - XYZ[:-1]
    dists = np.sqrt(np.sum(delta_XYZ**2, axis=1))
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    cum_dist_by_gi = dict(zip(gtem_data['params']['gis'], cum_dists))
    return cum_dist_by_gi

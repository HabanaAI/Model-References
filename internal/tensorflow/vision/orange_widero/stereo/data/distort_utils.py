import numpy as np
from scipy.interpolate import interp1d


def get_distort_params(clip,  cams=['main', 'rear', 'parking_rear',
                                    'parking_front', 'frontCornerLeft',
                                    'parking_left', 'rearCornerLeft',
                                    'frontCornerRight', 'parking_right',
                                    'rearCornerRight']):
    distortParams = {}
    for cam in cams:
        pp = np.float64(clip.conf_files(cam)['camera'][cam]['camK'].split(' '))[-2:]
        distortParams[cam] = np.float64(clip.conf_files(cam)['camera'][cam]['distortParams'].split(' ')[1:])
        distortParams[cam] = np.concatenate(
            [pp, distortParams[cam][:-1], np.zeros(shape=(12 - distortParams[cam].shape[0],)),
             np.expand_dims(np.array(distortParams[cam][-1]), 0)])
    ru_lut = {}
    maxRd = {}
    for cam in cams:
        if 'parking' in cam:
            maxRd[cam] = 500.
        else:
            maxRd[cam] = 700.
        maxRu = 1500.
        rd = np.arange(0, maxRd[cam], .01)
        n_k = distortParams[cam].shape[0] - 5
        factor = 1.
        rdN = 1
        rd_sqr = rd * rd

        for i in np.arange(n_k):
            k = distortParams[cam][i + 4]
            k_sqr = k * k
            c = np.sign(k) * k_sqr
            for j in np.arange(i):
                c *= k_sqr
            rdN *= rd_sqr
            factor += c * rdN
        ru = rd * (1 / factor)
        ru_int = interp1d(ru, rd, fill_value='extrapolate')(np.arange(maxRu))
        ru_lut[cam] = ru_int
    return distortParams, ru_lut


def get_distort_params_arrays(clip):
    def dict2array(dic, cams_dict={0: 'main', 1: 'rear',
                                   2: 'parking_rear', 3: 'parking_front',
                                   4: 'frontCornerLeft', 5: 'parking_left',
                                   6: 'rearCornerLeft', 7: 'frontCornerRight',
                                   8: 'parking_right', 9: 'rearCornerRight'}):
        num_cams = len(cams_dict.keys())
        element_shape = dic['main'].shape
        arr = np.zeros(tuple([num_cams] + list(element_shape)), dtype=np.float32)
        for i in np.arange(num_cams):
            arr[i, ...] = np.float32(dic[cams_dict[i]])
        return arr

    distort_params, ru_lut = get_distort_params(clip)
    distort_params_array = dict2array(distort_params)
    ru_lut_array = dict2array(ru_lut)
    return distort_params_array, ru_lut_array

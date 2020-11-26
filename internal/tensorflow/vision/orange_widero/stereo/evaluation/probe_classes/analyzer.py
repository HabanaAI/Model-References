import json
import numpy as np
from os.path import join
from scipy.stats import binned_statistic, binned_statistic_2d
import heapq
from stereo.evaluation.probe import Probe
from stereo.common.general_utils import tree_base

def clean_bin_stats(hist):

    hist_mask = np.logical_not(np.isnan(hist)).astype(np.int32)
    hist[np.isnan(hist)] = 0.
    return hist, hist_mask

def average_stats(hist_sums, hist_counter):

    hist_counter[hist_counter == 0.] = 1.
    hist_sums[hist_counter == 0.] = np.nan
    return hist_sums / hist_counter

def my_heapq_append(elem, my_heapq, cmp_idx, store_bottom=5):

    approved = True
    for worst in my_heapq:
        if elem[2] == worst[1][2] :
            if elem[cmp_idx] >= worst[0]:
                my_heapq.remove(worst)
                heapq.heapify(my_heapq)
            else:
                approved = False
    if approved:
        if len(my_heapq) < store_bottom:
            heapq.heappush(my_heapq, (elem[cmp_idx], elem))
        else:
            heapq.heappushpop(my_heapq, (elem[cmp_idx], elem))

    return my_heapq

class Analyzer(Probe):

    def my_init(self, store_bottom=20):

        self.worst_loss = []
        self.worst_accuracy = []
        self.canon = []
        canon_path = join(tree_base(), 'stereo', 'evaluation', 'canonical_examples.json')
        with open(canon_path, 'rb') as f:
            canon_conf = json.load(f)
        canon_conf = canon_conf[self.cam]
        self.canon_dict = {}
        for key in canon_conf.keys():
            self.canon_dict[canon_conf[key]['clip']] = canon_conf[key]['grab_index']
        print ("There are %d canonical examples" % (len(self.canon_dict)))
        self.elem_counter = 0
        self.store_bottom = store_bottom if not self.local else 2
        self.total_loss = 0.
        self.mean_sums = None
        self.med_sums = None
        self.twoD_med_sums = None
        self.twoD_hist_sums = None
        self.mean_counter = None
        self.med_counter = None
        self.twoD_med_counter = None
        self.dist_bins = np.arange(1., 80., 1.)
        self.hist_bins = np.arange(0.0, .25, .01)
        # TODO: Fix hard-coding of px bins relative to size
        self.px_bins = np.arange(-360, 360, 15)
        self.unique_clips = set()
        self.unique_black = set()

    def my_update(self, loss, clip_name, gi, output, center_im, inp_ims, ground_truth, x):
        output = output['out']
        # mask signifying which clips in batch are not on the blacklist
        whitelist_mask = np.zeros((self.batch_size,))
        for i in range(self.batch_size):
            if self.batch_size == 1:
                elem = (loss, clip_name, gi, output, center_im, inp_ims, ground_truth)
            else:
                elem = (loss[i], clip_name[i], gi[i], output[i], center_im[i], inp_ims[i], ground_truth[i])
            self.unique_clips.add(elem[1])
            if elem[1] not in self.blacklist:
                whitelist_mask[i] = 1.
                self.add_to_heapq(elem)
            else:
                self.unique_black.add(elem[1])
        self.total_loss += np.sum(np.array([loss]) * whitelist_mask)
        self.elem_counter += self.batch_size
        self.calc_lidar_bins(output, ground_truth, x, whitelist_mask)
        print ("analyzed " + str(self.elem_counter) + " examples\n")


    def my_summarize(self):

        mean_loss = self.total_loss / self.elem_counter

        bin_means = average_stats(self.mean_sums, self.mean_counter)
        bin_meds = average_stats(self.med_sums, self.med_counter)
        bin_twoD_meds = average_stats(self.twoD_med_sums, self.twoD_med_counter)
        bin_twoD_hist = self.twoD_hist_sums

        out_dict = {'summary_loss': mean_loss, 'means': bin_means, 'meds': bin_meds,
                    'twoD_hist': bin_twoD_hist, 'twoD_meds': bin_twoD_meds,
                    'dist_bins': self.dist_bins, 'px_bins': self.px_bins, 'hist_bins': self.hist_bins,
                    'worst_loss': np.array(self.worst_loss, dtype=object),
                    'worst_accuracy': np.array(self.worst_accuracy, dtype=object),
                    'canon': np.array(self.canon, dtype=object)}

        print ("%d unique clips in dataset" % (len(self.unique_clips)))
        print ("%d of them excluded" % (len(self.unique_black)))

        return out_dict

    def my_concatenate(self, dict_lst):

        summary_loss = 0.
        means = np.zeros_like(dict_lst[0]['means'])
        meds = np.zeros_like(dict_lst[0]['meds'])
        twoD_hist = np.zeros_like(dict_lst[0]['twoD_hist'])
        twoD_meds = np.zeros_like(dict_lst[0]['twoD_meds'])
        twoD_meds_counter = np.zeros_like(dict_lst[0]['twoD_meds'])
        dist_bins = dict_lst[0]['dist_bins']
        px_bins = dict_lst[0]['px_bins']
        hist_bins = dict_lst[0]['hist_bins']

        worst_loss = []
        worst_accuracy = []
        canon = []

        for i in range(self.num_instances):
            temp_npz = dict_lst[i]
            summary_loss += temp_npz['summary_loss']
            means += temp_npz['means']
            meds += temp_npz['meds']
            twoD_hist_mask = np.logical_not(np.isnan(temp_npz['twoD_hist']))
            twoD_hist[twoD_hist_mask] += temp_npz['twoD_hist'][twoD_hist_mask]
            twoD_meds_mask = np.logical_not(np.isnan(temp_npz['twoD_meds']))
            twoD_meds[twoD_meds_mask] += temp_npz['twoD_meds'][twoD_meds_mask]
            twoD_meds_counter += twoD_meds_mask.astype(np.int32)

            canon += list(temp_npz['canon'])

            for j in range(self.store_bottom):
                elem = tuple(temp_npz['worst_accuracy'][j])
                worst_accuracy = my_heapq_append(elem[1], worst_accuracy, 0, store_bottom=self.store_bottom)
                elem = tuple(temp_npz['worst_loss'][j])
                worst_loss = my_heapq_append(elem[1], worst_loss, 1, store_bottom=self.store_bottom)

        summary_loss /= self.num_instances
        means /= self.num_instances
        meds /= self.num_instances
        twoD_meds[twoD_meds_counter == 0.] = np.nan
        twoD_meds_counter[twoD_meds_counter == 0.] = 1.
        twoD_meds /= twoD_meds_counter

        worst_accuracy = [x[1] for x in worst_accuracy]
        worst_loss = [x[1] for x in worst_loss]

        out_dict = {'summary_loss': summary_loss, 'means': means, 'meds': meds, 'twoD_hist': twoD_hist,
                    'twoD_meds': twoD_meds, 'dist_bins': dist_bins, 'px_bins': px_bins, 'hist_bins': hist_bins,
                    'worst_loss': np.array(worst_loss, dtype=object),
                    'worst_accuracy': np.array(worst_accuracy, dtype=object), 'loss': self.loss_name,
                    'dataset': self.dataset_name, 'canon': np.array(canon, dtype=object)}

        return out_dict


    def add_to_heapq(self, elem):
        lidar = elem[6]
        output = np.expand_dims(elem[3][:,:,0], 2)
        mask = lidar != 0.
        acc = np.mean(np.abs((1. / lidar[mask]) - output[mask]))
        min_gt_ratio = 0.05
        if np.mean(mask.astype(np.float32)) < min_gt_ratio:
            acc = 0.
        # Add accuracy to element
        elem = (acc,) + elem
        if elem[2] in self.canon_dict.keys() and int(self.canon_dict[elem[2]]) == int(elem[3]):
            self.canon.append(elem)

        self.worst_accuracy = my_heapq_append(elem, self.worst_accuracy, cmp_idx=0, store_bottom=self.store_bottom)
        self.worst_loss = my_heapq_append(elem, self.worst_loss, cmp_idx=1, store_bottom=self.store_bottom)


    def calc_lidar_bins(self, output, ground_truth, x, whitelist_mask):

        output = np.expand_dims(output[:,:,:,0], 3)
        ground_truth[np.logical_not(whitelist_mask), :, :, :] = 0.
        lid_mask = ground_truth != 0.
        inv_lidar_mask = 1. / ground_truth[lid_mask]
        norm_lidar_mask = ground_truth[lid_mask]
        outputs_mask = output[lid_mask]
        abs_error = np.abs(inv_lidar_mask - outputs_mask)
        percent_error = abs_error / inv_lidar_mask
        x_mask = x[lid_mask]

        # Individual
        twoD_median, _, _, _ = binned_statistic_2d(x_mask, norm_lidar_mask, percent_error,
                                                   bins=[self.px_bins, self.dist_bins], statistic='median')
        twoD_hist, _, _, _ = binned_statistic_2d(percent_error, norm_lidar_mask, percent_error,
                                                 bins=[self.hist_bins, self.dist_bins], statistic='count')
        # Compare
        out_mean, _, _ = binned_statistic(norm_lidar_mask, percent_error,
                                          statistic='mean', bins=self.dist_bins)
        out_median, _, _ = binned_statistic(norm_lidar_mask, percent_error,
                                            statistic='median', bins=self.dist_bins)

        twoD_median, twoD_median_mask = clean_bin_stats(twoD_median)
        out_mean, out_mean_mask = clean_bin_stats(out_mean)
        out_median, out_median_mask = clean_bin_stats(out_median)

        if self.mean_sums is None:
            self.mean_sums = out_mean
            self.med_sums = out_median
            self.twoD_med_sums = twoD_median
            self.twoD_hist_sums = twoD_hist
            self.mean_counter = out_mean_mask
            self.med_counter = out_median_mask
            self.twoD_med_counter = twoD_median_mask
        else:
            self.mean_sums += out_mean
            self.med_sums += out_median
            self.twoD_med_sums += twoD_median
            self.twoD_hist_sums += twoD_hist
            self.mean_counter += out_mean_mask
            self.med_counter += out_median_mask
            self.twoD_med_counter += twoD_median_mask
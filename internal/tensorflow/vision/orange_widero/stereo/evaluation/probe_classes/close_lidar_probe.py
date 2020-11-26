import numpy as np
import heapq
from stereo.evaluation.probe import Probe

'''
The goal of this probe is to find the examples with the most lidar points closer than 3 meters and save them for
further investigation because there's a good chance (in main) that many of those points come from aggregation mistakes
'''

class closeLidar(Probe):

    def my_init(self):

        self.close = []
        self.max_dist = 3.
        self.store_num = 10


    def my_update(self, loss, clip_name, gi, output, center_im, inp_ims, ground_truth, x):
        output = output['out']
        close_mask = np.logical_and(ground_truth < self.max_dist, ground_truth != 0.)
        mask_sum = np.sum(close_mask.astype(np.int32), axis=(1,2,3))
        for i in range(self.batch_size):
            elem = (float(mask_sum[i]), float(loss[i]), clip_name[i], gi[i], output[i], center_im[i], inp_ims[i],
                    ground_truth[i])
            if len(self.close) < self.store_num:
                heapq.heappush(self.close, elem)
            else:
                heapq.heappushpop(self.close, elem)


    def my_summarize(self):

        dict = {"close": self.close}
        return dict


    def my_concatenate(self, dict_lst):

        final_close = []
        for dict in dict_lst:
            for i in range(self.store_num):
                if len(final_close) < self.store_num:
                    heapq.heappush(final_close, tuple(dict["close"][i]))
                else:
                    heapq.heappushpop(final_close, tuple(dict["close"][i]))

        final_close = [x[1:] for x in final_close]

        dict = {"close": final_close}
        return dict

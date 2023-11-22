# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company

import numpy as np
import copy
import collections
from itertools import combinations
import time

class BruteForceOptimalBucketing():
    def __init__(self, inp_dist, num_buckets, numbucket_threshold=10000000, verbose=False, collect=1000, print_freq=10000):
        self.inp_dist = collections.OrderedDict(sorted(inp_dist.items()))
        # not sure if deepcopy preserves order, hence resorting
        self.inp_dist_orig = collections.OrderedDict(sorted(copy.deepcopy(self.inp_dist).items()))
        self.num_buckets = num_buckets
        self.numbucket_threshold = numbucket_threshold
        if numbucket_threshold > 0:
            self.simplify_distribution()
        self.verbose = verbose
        if self.verbose:
            print('Original distribution: ', self.inp_dist_orig)
            print('Modified distribution: ', self.inp_dist)
            print('kl divergence: ', self.kl_div())
        self.max_shape = max(self.inp_dist)
        key_col = []
        val_col = []
        for k in sorted(self.inp_dist):
            key_col += [k]
            val_col += [self.inp_dist[k]]
        self.num_shapes = len(key_col)
        self.key_col_tensor = np.array(key_col) # sorted by keys (first column)
        self.val_col_tensor = np.array(val_col)
        self.collect = collect
        self.print_freq = print_freq
        #self.key_col_tensor_tiled = np.tile(self.key_col_tensor, (num_buckets,self.collect,1)).T
    def kl_div(self):
        total = sum(self.inp_dist.values())
        kld = 0
        for k in self.inp_dist_orig:
            q = self.inp_dist_orig[k] / total
            tmp = self.inp_dist.get(k,0)
            if tmp == 0:
                term = 0
            else:
                term = tmp * np.log((tmp / total) / q)
            kld += term
        return kld
    def simplify_distribution(self):
        while self.num_possible_buckets() > self.numbucket_threshold:
            self.fuse_inp_dist()
    def fuse_inp_dist(self):
        # helper finds the smallest frequency (which will be removed)
        def helper(d):
            least_count = None
            for idx, k in enumerate(d):
                if least_count is None or d[k] < least_count:
                    least_count = d[k]
                    least_idx = idx
                    to_be_removed = k
            return least_count, least_idx, to_be_removed
        sum_vals_before = sum(self.inp_dist.values())
        assert sum_vals_before == sum(self.inp_dist_orig.values())
        # Remove the last (largest) shape from the search of lowest frequency to be deleted,
        # because that can't be deleted
        tmp = collections.OrderedDict(sorted(copy.deepcopy(self.inp_dist).items()))
        tmp.pop(max(tmp))
        # search for the shape with least frequency
        least_count, least_idx, to_be_removed = helper(tmp)
        # fuse the shape with least frequency with its right neighbour (next bigger shape)
        fuse_with = least_idx+1
        for idx, k in enumerate(self.inp_dist):
            if fuse_with == idx:
                self.inp_dist[k] = self.inp_dist[k]+least_count
        # Remove the shape with least frequency
        self.inp_dist.pop(to_be_removed)
        sum_vals_after = sum(self.inp_dist.values())
        assert sum_vals_before == sum_vals_after
    def num_possible_buckets(self):
        from functools import reduce
        import operator as op
        n = len(self.inp_dist)-1
        r = self.num_buckets-1
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer // denom  # or / in Python 2
    # function to evaluate
    def num_padding(self, buckets):
        tot_pad = 0
        cur_bucket_idx = 0
        for k in self.inp_dist_orig: # self.inp_dist is expected to be sorted, hence we can do the cur_bucket_idx optimization
            while True:
                bucket = buckets[cur_bucket_idx]
                if k > bucket:
                    cur_bucket_idx += 1
                else:
                    break
            padding = (bucket - k) * self.inp_dist_orig[k]
            assert padding >= 0
            tot_pad += padding
        return tot_pad
    def find_optimal_buckets(self):
        result_best = None
        sizes = [k for k in self.inp_dist.keys()]
        sizes_without_largest = sizes[:-1]
        num = self.num_possible_buckets()
        if self.verbose:
            print(f'Combinations to try: {num}')
        t0 = time.time()
        collect_ctr = 0
        self.idx_collection = []
        self.bucket_boundary_collection = []
        result_best = None
        def update_helper(idx, result_best, best_padwaste_in_curr_collection, best_bucket_in_curr_collection):
            if result_best is None or result_best['wasted_padding'] > best_padwaste_in_curr_collection:
                tmp = {'wasted_padding':best_padwaste_in_curr_collection, 'idx':idx, 'buckets':copy.deepcopy(best_bucket_in_curr_collection)}
                if self.verbose:
                    print('Best till now: ', tmp)
                return tmp
            else:
                return result_best
        for idx, bucket_boundary in (enumerate(combinations(sizes_without_largest, self.num_buckets-1))):
            if collect_ctr == self.collect:
                best_padwaste_in_curr_collection, best_bucket_in_curr_collection, best_idx = self.process_collection()
                result_best = update_helper(idx - self.collect + best_idx, result_best, best_padwaste_in_curr_collection, best_bucket_in_curr_collection)
                self.idx_collection = []
                self.bucket_boundary_collection = []
                collect_ctr = 0
            self.idx_collection.append(idx)
            self.bucket_boundary_collection.append(list(bucket_boundary) + [sizes[-1]])
            collect_ctr += 1
            if idx % self.print_freq == self.print_freq-1 and self.verbose:
                curr_time = time.time()
                time_till_now = curr_time-t0
                projected_time_left = time_till_now * ((num / idx) - 1) if idx > 0 else -1
                print(f'{idx}/{num}: {(idx/num):.3f}. Time taken till now {time_till_now:.3f}. Projected time left {projected_time_left:.3f}. Best {result_best}')
        if len(self.idx_collection) > 0:
            best_padwaste_in_curr_collection, best_bucket_in_curr_collection, best_idx = self.process_collection()
            result_best = update_helper(idx - len(self.idx_collection) + best_idx, result_best, best_padwaste_in_curr_collection, best_bucket_in_curr_collection)
        return result_best
    def process_collection(self):
        # self.collect x self.num_buckets
        bucket_boundary_collection = np.array(self.bucket_boundary_collection)
        # self.num_shapes x self.collect x self.num_buckets
        buckets_tiled = np.tile(np.array(bucket_boundary_collection), (self.num_shapes, 1, 1))
        # self.num_shapes x self.collect
        key_col_tensor_tiled = np.tile(self.key_col_tensor, (self.num_buckets,bucket_boundary_collection.shape[0],1)).T
        bucket_idx = np.argmin(key_col_tensor_tiled > buckets_tiled, 2)
        bucket_for_each_shape = np.take_along_axis(bucket_boundary_collection, bucket_idx.T, 1)
        padding_waste_per_shape = bucket_for_each_shape - np.expand_dims(self.key_col_tensor, 0)
        #assert np.all(padding_waste_per_shape >= 0)
        total_padding_waste = np.sum((padding_waste_per_shape * self.val_col_tensor), 1)
        #assert len(total_padding_waste)
        best_idx = np.argmin(total_padding_waste)
        return total_padding_waste[best_idx], bucket_boundary_collection[best_idx], best_idx

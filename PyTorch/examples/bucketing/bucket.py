# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company

import numpy as np
import time
import pickle as pkl

# A decorator for input/output validation of bucketing algorithms
def get_check_bucket(allow_none_return):
    # some bucketing algos like LP can return None
    def check_bucket(bucketer):
        def helper(shapes, num_buckets, *args, **kwargs):
            for k in shapes:
                assert type(k) == type(1)
                assert k >= 0
            assert num_buckets >= 1
            assert type(num_buckets) == type(1)
            buckets = bucketer(shapes, num_buckets, *args, **kwargs)
            if allow_none_return:
                if buckets is None:
                    return None
            assert len(buckets) <= num_buckets
            assert buckets[-1] <= max(shapes)
            return buckets
        return helper
    return check_bucket

# Percentile based bucketing
@get_check_bucket(False)
def percentile_bucket(shapes, num_buckets):
    buckets = np.unique(
      np.percentile(
            shapes,
            np.linspace(0, 100, num_buckets + 1),
            interpolation="lower",
      )[1:]
    )
    return buckets

# LP based bucketing
@get_check_bucket(True)
def lp_bucket(shapes, num_buckets):
    from pulp import LpMinimize, LpProblem, lpSum, PULP_CBC_CMD, LpStatus, LpVariable, LpInteger
    def padding_overhead(bucket_size,sample_size,num_samples):
        if (sample_size-bucket_size)>0:
            return 1e32
        else:
            return (bucket_size-sample_size)*num_samples
    data_unique=np.unique(shapes)
    prob = LpProblem('OptimalBucket',LpMinimize)
    Combinations=[]; padLoss={}; Indicators={}; DeltaM={}

    for s in data_unique:
        num_samples=(shapes==s).sum()
        for b in data_unique:
            Combinations.append('ind' + '_{}b_{}s'.format(b,s))
            padLoss['ind' + '_{}b_{}s'.format(b,s)] = padding_overhead(b,s,num_samples)
            Indicators['ind' + '_{}b_{}s'.format(b,s)] = LpVariable('ind' + '_{}b_{}s'.format(b,s),0,1,LpInteger)

    prob += lpSum([Indicators[ind]*padLoss[ind] for ind in padLoss.keys()]) # Objective (minimize padding)

    for s in data_unique:
        prob += lpSum([Indicators[key] for key in Combinations if '_{}s'.format(s) in key]) == 1
    bucket_indecators=[]
    for b in data_unique:
        Indicators['ind_bucket' + '_{}'.format(b)]=LpVariable('ind_bucket' + '_{}'.format(b),0,1,LpInteger)
        bucket_indecators.append(Indicators['ind_bucket' + '_{}'.format(b)])
    for b in data_unique:
        prob += lpSum([Indicators[key] for key in Combinations if '_{}b'.format(b) in key]) <= Indicators['ind_bucket' + '_{}'.format(b)]*len(data_unique)

    prob += lpSum(bucket_indecators)==num_buckets

    prob.solve(PULP_CBC_CMD(msg=0))
    LpStatus[prob.status]

    ip_buckets=[]
    for v in prob.variables():
        if 'ind_bucket' in v.name and v.value() > 0:
            ip_buckets.append(int(v.name.split('_')[-1]))

    if (prob.status==-1):
        print('Infeasable')
        return None
    else:
        return tuple(sorted(ip_buckets))

# Pad to max or constant bucketing
@get_check_bucket(False)
def const_bucket(shapes, num_buckets):
    return [max(shapes)]

# Uniform intervals bucketing
@get_check_bucket(False)
def uniform_bucket(shapes, num_buckets):
    mn = min(shapes)
    mx = max(shapes)
    step = (mx - mn) / num_buckets
    buckets = [mx]
    curr_bucket = mx
    step = (mx - mn) / num_buckets
    for i in range(num_buckets-1):
        curr_bucket = curr_bucket - step
        buckets = [curr_bucket] + buckets
    buckets = [round(k) for k in buckets]
    return buckets

# Brute force min pad waste bucketing
@get_check_bucket(False)
def brute_force_min_pad_waste(shapes, num_buckets, max_elems=10000000):
    from brute_force_min_pad_waste import BruteForceOptimalBucketing
    size_freq = {}
    for k in shapes:
        size_freq[k] = size_freq.get(k,0)+1
    ob = BruteForceOptimalBucketing(size_freq, num_buckets, numbucket_threshold=max_elems)
    res = ob.find_optimal_buckets()
    return res['buckets']

# Lloyd-Max quantization based bucketing
@get_check_bucket(False)
def lloyd_max_bucketing(shapes, num_buckets, max_steps=20):
    from lloyd_max_bucket import lloydmax
    from scipy.interpolate import CubicSpline
    hist = {}
    for k in shapes:
        hist[k] = hist.get(k,0) + 1
    x = []
    y = []
    for k in sorted(hist.keys()):
        x += [k]
        y += [hist[k]/sum(hist.values())]
    pdf = CubicSpline(x, y)
    repr = uniform_bucket(shapes, num_buckets)
    thresholds = list((np.array(repr[:-1]) + np.array(repr[1:]))/2)
    x,t,e = lloydmax(thresholds,repr,0.01, pdf, min(shapes), max(shapes), max_steps)
    buckets = [int(k) for k in t] + [max(shapes)]
    return buckets

def normalize_trial_buckets(trial_buckets):
    if trial_buckets is None:
        trial_buckets = 10
    if type(trial_buckets) == type(1):
        trial_buckets = range(1,trial_buckets)
    return trial_buckets

def eval_bucketing(buckets, shapes):
    tot = sum([min([i for i in buckets if i >= k]) for k in shapes])
    return tot, tot/len(shapes)


def bucket_analysis(shapes, bucket_algos, trial_buckets=None):
    trial_buckets = normalize_trial_buckets(trial_buckets)
    results = {}
    for algoidx, (bucket_algo_name, bucket_algo) in enumerate(bucket_algos):
        print(f'Processing {bucket_algo_name}')
        res = {}
        for num_bucket in trial_buckets:
            print(f'Processing bucket={num_bucket}')
            t0 = time.time()
            buckets = bucket_algo(shapes, num_bucket)
            t1 = time.time()
            if buckets is None:
                print(f'Failed to generate buckets for {bucket_algo_name} for {num_bucket} buckets. Falling back to const bucketing')
                buckets = const_bucket(shapes, num_bucket)
            totwaste, avgwaste = eval_bucketing(buckets, shapes)
            res[num_bucket] = {'totwaste':totwaste, 'avgwaste':avgwaste, 'time':t1-t0}
            print(algoidx, num_bucket, totwaste, avgwaste, t1-t0)
        assert bucket_algo_name not in results
        results[bucket_algo_name] = res
    pkl.dump(results, open('res.pkl', 'wb'), protocol=pkl.HIGHEST_PROTOCOL)
    return results

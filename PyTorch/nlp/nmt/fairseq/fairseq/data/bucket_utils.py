# Copyright (c) 2023, Habana Labs Ltd.  All rights reserved.

from pulp import *
import numpy as np
import logging
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

def padding_overhead(bucket_size,sample_size,num_samples):
    if (sample_size-bucket_size)>0:
        return 1e32
    else:
        return (bucket_size-sample_size)*num_samples

def get_optimal_bucket_ip(data=None,max_bucket_size=10):
    logger.info("Calculating the optimal bucket")

    data_unique=np.unique(data)
    prob = LpProblem('OptimalBucket',LpMinimize)
    Combinations=[]; padLoss={}; Indicators={}; DeltaM={}

    for s in data_unique:
        num_samples=(data==s).sum()
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

    prob += lpSum(bucket_indecators)==max_bucket_size

    prob.solve(PULP_CBC_CMD(msg=0))
    LpStatus[prob.status]

    ip_buckets=[]
    for v in prob.variables():
        if 'ind_bucket' in v.name and v.value() > 0:
            ip_buckets.append(int(v.name.split('_')[-1]))

    if (prob.status==-1):
        logger.info('Infeasable')
        return None
    else:
        return tuple(sorted(ip_buckets))

def get_stored_optimal_bucket(size, num_buckets=10):
    """
    Retrieve stored optimal bucket for the size and num_buckets
    """
    bucket_cache_file = Path('optimal_buckets.npy')
    if bucket_cache_file.is_file():
        dict = np.load('optimal_buckets.npy', allow_pickle=True).item()
        if dict:
            hash_id = hashlib.sha256(size).hexdigest().upper()
            return dict.get((hash_id, num_buckets))
    else:
        logger.info("No cached optimal buckets file found")
    return None

def store_optimal_bucket(size, num_buckets, bucket):
    """
    Store optimal bucket for the size and num_buckets to a file
    """
    bucket_cache_file = Path('optimal_buckets.npy')
    if bucket_cache_file.is_file():
        dict = np.load('optimal_buckets.npy', allow_pickle=True).item()
        if dict:
            dict[(hashlib.sha256(size).hexdigest().upper(), num_buckets)] = bucket
        else:
            dict = {(hashlib.sha256(size).hexdigest().upper(), num_buckets):bucket}
    else:
        dict = {(hashlib.sha256(size).hexdigest().upper(), num_buckets): bucket}
    np.save('optimal_buckets.npy', dict, allow_pickle=True)

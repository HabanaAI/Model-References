# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company

from datasets_library import gaussian, batched_gaussian, batch_by_formula, sample_from_pdf, generate_random_gaussian
import numpy as np
np.random.seed(0)
from bucket import bucket_analysis, lp_bucket, const_bucket, uniform_bucket, percentile_bucket, lloyd_max_bucketing, brute_force_min_pad_waste
import itertools

def test_squad_batching():
    print("Plotting gaussian")
    num_samples = 100000
    bs = 4
    gs = gaussian(num_samples)
    print(sum(gs))
    assert len(gs) == 100000
    assert np.abs(np.mean(gs) - 500) <= 2
    assert np.abs(np.var(gs) - 2486.8457) <= 20
    orig = batched_gaussian(gs, 1, max)
    assert len(orig) == 100000
    assert set(orig) == set(gs)
    max_batch4 = batched_gaussian(gs, bs, max)
    assert len(max_batch4) == 25000
    assert np.mean(max_batch4) > np.mean(gs)
    assert  np.var(max_batch4) <  np.var(gs)
    min_batch4 = batched_gaussian(gs, bs, min)
    assert len(min_batch4) == 25000
    assert np.mean(min_batch4) < np.mean(gs)
    assert np.var(min_batch4) <  np.var(gs)
    max_formula_batch4 = sample_from_pdf(batch_by_formula(gs, bs, 'max'), num_samples)
    assert len(max_formula_batch4) == 100000
    assert np.abs(np.mean(max_formula_batch4) - np.mean(max_batch4)) < 5
    assert np.abs(np.var(max_formula_batch4) - np.var(max_batch4)) < 40
    min_formula_batch4 = sample_from_pdf(batch_by_formula(gs, bs, 'min'), num_samples)
    assert len(min_formula_batch4) == 100000
    assert np.abs(np.mean(min_formula_batch4) - np.mean(min_batch4)) < 5
    assert np.abs(np.var(min_formula_batch4) - np.var(min_batch4)) < 40


def test_bucketing():
    shapes = list(itertools.islice(generate_random_gaussian(), 1000))
    assert sum(shapes) == 498957
    assert len(set(shapes)) == 229
    #("lp_bucket", lp_bucket) # this takes quite long, so skipping its test
    results = bucket_analysis(shapes, [("const_bucket", const_bucket), ("uniform_bucket", uniform_bucket), \
        ("percentile_bucket", percentile_bucket), ("lloyd_max_bucketing", lloyd_max_bucketing), \
            ("brute_force_min_pad_waste", brute_force_min_pad_waste)], [2,20])
    expected = {'const_bucket': {2: {'totwaste': 662000, 'avgwaste': 662.0}, 20: {'totwaste': 662000, 'avgwaste': 662.0}}, \
            'uniform_bucket': {2: {'totwaste': 579218, 'avgwaste': 579.218}, 20: {'totwaste': 506507, 'avgwaste': 506.507}}, \
            'percentile_bucket': {2: {'totwaste': 579522, 'avgwaste': 579.522}, 20: {'totwaste': 506122, 'avgwaste': 506.122}}, \
            'lloyd_max_bucketing': {2: {'totwaste': 594004, 'avgwaste': 594.004}, 20: {'totwaste': 506218, 'avgwaste': 506.218}}, \
            'brute_force_min_pad_waste': {2: {'totwaste': 562739, 'avgwaste': 562.739,}, 20: {'totwaste': 504726, 'avgwaste': 504.726}}}
    for algo_name in ["const_bucket", "uniform_bucket", "percentile_bucket", "lloyd_max_bucketing", "brute_force_min_pad_waste"]:
        assert algo_name in results
        val = results[algo_name]
        for bkt in [2,20]:
            assert bkt in val
            bkt_result = val[bkt]
            bkt_result.pop('time')
            assert bkt_result == expected[algo_name][bkt]


def test_numsteps():

    shapes = list(itertools.islice(generate_random_gaussian(), 10000))
    lloyd_max_set_step = lambda step : (lambda shp, num_buckets : lloyd_max_bucketing(shp, num_buckets, step))

    results = bucket_analysis(shapes, [("lloyd_max_02", lloyd_max_set_step(2)), ("lloyd_max_10", lloyd_max_set_step(10)), ("lloyd_max_20", lloyd_max_set_step(20))], [6, 10])
    expected = {'lloyd_max_02': {6: {'totwaste': 5284440, 'avgwaste': 528.444}, \
                10: {'totwaste': 5172300, 'avgwaste': 517.23,}}, \
                'lloyd_max_10': {6: {'totwaste': 5226954, 'avgwaste': 522.6954}, \
                10: {'totwaste': 5149487, 'avgwaste': 514.9487}}, \
                'lloyd_max_20': {6: {'totwaste': 5209336, 'avgwaste': 520.9336}, \
                10: {'totwaste': 5137341, 'avgwaste': 513.7341}}, \
                'lloyd_max_30': {6: {'totwaste': 5203774, 'avgwaste': 520.3774}, \
                10: {'totwaste': 5131550, 'avgwaste': 513.155}}}
    expected = {'lloyd_max_02': {6: {'totwaste': 5284440, 'avgwaste': 528.444}, 10: {'totwaste': 5172300, 'avgwaste': 517.23}}, \
        'lloyd_max_10': {6: {'totwaste': 5226191, 'avgwaste': 522.6191}, 10: {'totwaste': 5147715, 'avgwaste': 514.7715}}, \
        'lloyd_max_20': {6: {'totwaste': 5209807, 'avgwaste': 520.9807}, 10: {'totwaste': 5135907, 'avgwaste': 513.5907}}}
    for algo_name in ["lloyd_max_02", "lloyd_max_10", "lloyd_max_20"]:
        assert algo_name in results
        val = results[algo_name]
        for bkt in [6,10]:
            assert bkt in val
            bkt_result = val[bkt]
            bkt_result.pop('time')
            #print(algo_name, bkt, bkt_result, expected[algo_name][bkt])
            assert bkt_result == expected[algo_name][bkt]


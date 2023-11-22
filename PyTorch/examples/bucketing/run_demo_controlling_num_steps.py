# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company

import itertools
from plotting import plot_bucket_analysis_results
from bucket import bucket_analysis, lloyd_max_bucketing, brute_force_min_pad_waste
from datasets_library import generate_random_gaussian

shapes = list(itertools.islice(generate_random_gaussian(), 10000))

lloyd_max_set_step = lambda step : (lambda shp, num_buckets : lloyd_max_bucketing(shp, num_buckets, step))
bruteforce_set_threshold = lambda th : (lambda shp, num_buckets : brute_force_min_pad_waste(shp, num_buckets, th))


results = bucket_analysis(shapes, [("lloyd_max_02", lloyd_max_set_step(2)), ("lloyd_max_10", lloyd_max_set_step(10)), ("lloyd_max_20", lloyd_max_set_step(20)), \
            ("lloyd_max_30", lloyd_max_set_step(30)), ("bruteforce_100k", bruteforce_set_threshold(100000)), ("bruteforce_1M", bruteforce_set_threshold(1000000))], [6, 8, 10])
plot_bucket_analysis_results(results, 'bucket_analysis_num_steps_gaussian.svg')

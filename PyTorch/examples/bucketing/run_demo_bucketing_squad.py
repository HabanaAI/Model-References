# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company

import itertools
from plotting import plot_bucket_analysis_results
from bucket import bucket_analysis, lp_bucket, const_bucket, uniform_bucket, percentile_bucket, lloyd_max_bucketing, brute_force_min_pad_waste
from datasets_library import squad

shapes = squad(4)
results = bucket_analysis(shapes, [("const_bucket", const_bucket), ("uniform_bucket", uniform_bucket), \
    ("percentile_bucket", percentile_bucket), ("lloyd_max_bucketing", lloyd_max_bucketing), \
        ("brute_force_min_pad_waste", brute_force_min_pad_waste)], [2,3,4,5,6,10,20])

plot_bucket_analysis_results(results, "bucket_analysis_bar_squad.svg")



###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import habana_generation_utils as hgu


default_options = {
    "early_stopping": True,
    "early_stopping_delay": 2, # schedule an extra step before checking early_stopping, i.e. schedule-0, skip-check-1, schedule-1, check-0, schedule-0, check-1
    "max_iterations": 128,
    "num_beams": 4,
    "static_shapes": True,
    "use_cache": True,
    "use_graphs": True,
    "limit_graphs": False,
    "use_rolling_position_ids": True,
    "reuse_cache": True,
    "kv_cache_fp8": False,
    "trim_logits": True,
    "kv_cache_margin": 129,
}


def get_options_dict(options_str: str = None) -> dict:
    options = {}
    if options_str is not None:
        options = dict(
            [hgu.parse_key_type_value(ktv) for ktv in options_str.split(',')]
        )
    return {**default_options, **options}

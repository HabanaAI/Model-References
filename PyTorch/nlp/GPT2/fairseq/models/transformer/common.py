# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.distributed import fsdp_wrap
from fairseq.modules.checkpoint_activations import checkpoint_wrapper


def handle_transformer_checkpoint_activations(layer, cfg, use_rng_tracker=False):
    checkpoint = cfg.checkpoint_activations
    layer = checkpoint_wrapper(
        layer,
        offload_to_cpu=cfg.offload_activations,
        use_rng_tracker=use_rng_tracker) if checkpoint else layer
    return layer


def handle_transformer_fsdp_wrapping(layer, cfg):
    # if we are checkpointing, enforce that FSDP always wraps the
    # checkpointed layer, regardless of layer size
    checkpoint = cfg.checkpoint_activations
    min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
    layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
    return layer

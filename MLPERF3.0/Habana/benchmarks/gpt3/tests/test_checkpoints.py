# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
import pytest
import shutil

from parameterized import parameterized
from megatron.testing_utils import (
    CaptureStdout,
    TestCasePlus,
    execute_subprocess_async,
    get_gpu_count,
    require_deepspeed,
    require_torch_gpu,
    require_torch_multi_gpu,
    set_seed
)

set_seed(42)


def parameterized_custom_name_func(func, param_num, param):
    # customize the test name generator function as we want both params to appear in the subtest
    # name, as by default it shows only the first param
    param_based_name = parameterized.to_safe_name("_to_".join(str(x) for x in param.args))
    return f"{func.__name__}_{param_based_name}"


params = [
    # TP_PP_DP
    ["1_1_1", "1_1_1"],
    ["2_1_1", "1_1_1"],
    ["1_2_1", "1_1_1"],
    ["1_1_2", "1_1_1"],

    ["1_1_1", "2_1_1"],
    ["1_1_1", "1_2_1"],
    ["1_1_1", "1_1_2"],

    ["1_1_2", "1_1_2"],
    ["1_1_2", "2_1_1"],
    ["1_1_2", "1_2_1"],

    ["1_2_1", "1_2_1"],
    ["1_2_1", "2_1_1"],
    ["1_2_1", "1_1_2"],

    ["2_1_1", "2_1_1"],
    ["2_1_1", "1_2_1"],
    ["2_1_1", "1_1_2"],

    ["2_2_2", "1_1_1"],
    ["2_2_2", "2_2_2"],
    ["1_1_1", "2_2_2"],

    ["1_1_8", "2_2_2"],

]


def get_launcher(num_gpus):
    # 1. explicitly set --num_nodes=1 just in case these tests end up run on a multi-node setup
    # - it won't be able to handle that
    return f"deepspeed --num_nodes 1 --num_gpus {num_gpus}".split()


@require_deepspeed
@require_torch_gpu
class MegDSTestCheckpoints(TestCasePlus):
    """ """

    def setUp(self):
        super().setUp()

        # at times magatron fails to build kernels and doesn't remove the lock file, which makes
        # subsequent runs hang - so make sure there is no lock when starting the testing
        meg_lock_file_path = self.repo_root_dir_str + "/megatron/fused_kernels/build/lock"
        if os.path.exists(meg_lock_file_path):
            os.unlink(meg_lock_file_path)

    @staticmethod
    def find_lines_with_pattern_in_buffer(buffer, pattern):
        lines = buffer.splitlines()
        res = []
        for line in lines:
            if line.find(pattern) != -1:
                res.append(line)
        return res

    def get_config(self, output_dir, tp_size, pp_size, dp_size, n_iters=None,
                   exit_interval=None, save_interval= None, skip_train=False,
                   use_bloom=False):

        data_dir = f"{self.data_dir}/gpt2"

        num_gpus = pp_size * tp_size * dp_size
        print(f"Using {num_gpus} GPUs")

        n_iters = 8 if n_iters is None else n_iters
        exit_interval = n_iters // 2 if exit_interval is None else exit_interval
        save_interval = 1 if save_interval is None else save_interval
        seq_len = 8

        # common/shared configs

        ds_args = f"""
                --deepspeed
                --deepspeed_config {self.test_file_dir_str}/ds_config_bf16.json
                --zero-stage 0
                --deepspeed-activation-checkpointing
        """.split()

        args = f"""
                --tensor-model-parallel-size {tp_size}
                --pipeline-model-parallel-size {pp_size}
                --distributed-backend nccl

                --log-interval 1
                --save-interval {save_interval}
                --eval-interval 10
                --eval-iters 1
                --exit-interval {exit_interval}

                --merge-file {data_dir}/gpt2-tiny-merges.txt
                --vocab-file {data_dir}/gpt2-tiny-vocab.json
                --split 99,0,1
                --save {output_dir}/checkpoints
                --load {output_dir}/checkpoints
                --data-path {data_dir}/meg-gpt2-openwebtext_text_document

                --num-layers 2
                --hidden-size 8
                --num-attention-heads 2
                --seq-length {seq_len}
                --max-position-embeddings 8
                --micro-batch-size 1
                --global-batch-size 16
                --train-iters {n_iters}

                --checkpoint-activations
                --partition-activations

                --optimizer adam
                --adam-beta1 0.9
                --adam-beta2 0.95
                --adam-eps 1e-8
                --lr 1e-4
                --lr-warmup-iters 1
                --lr-decay-iters 6
                --clip-grad 1.0
                --weight-decay 1e-1
                --bf16
        """

        # removed below args to speedup test
        _ = f"""
                --tensorboard-dir {output_dir}/tensorboard
                --tensorboard-queue-size 5
                --log-timers-to-tensorboard
                --log-batch-size-to-tensorboard
                --log-validation-ppl-to-tensorboard
        """

        if skip_train:
            args += "--skip-train"

        args = args.split()

        if use_bloom:
            bloom_args = f"""
                --embed-layernorm
                --position-embedding-type alibi
                """.split()
            args.extend(bloom_args)

        return args, ds_args, num_gpus

    def train_checkpoint(self, output_dir, tp_size=1, pp_size=1, dp_size=1,
                         n_iters=None, exit_interval=None, save_interval=None,
                         skip_train=False):
        src_dir = self.src_dir
        script = [f"{src_dir}/pretrain_gpt.py"]

        args, ds_args, num_gpus = self.get_config(output_dir, tp_size, pp_size, dp_size,
                                                  n_iters=n_iters, exit_interval=exit_interval,
                                                  save_interval=save_interval,
                                                  skip_train=skip_train)
        launcher = get_launcher(num_gpus)
        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] + cmd)); die

        # 1. test training from scratch (no checkpoint)
        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # test deepspeed is running
        self.assertIn("DeepSpeed info", cs.out)

        # test reports
        self.assertIn("consumed samples", cs.out)

        # test there should be no checkpoint this round
        self.assertIn(f"Unable to find latest file at {output_dir}/checkpoints/latest", cs.out)

        # test checkpoint saving
        self.assertIn("successfully saved checkpoint at iteration", cs.out)
        return cs.out

    def convert_checkpoint_to_universal(self, output_dir, step):
        cmd = f"""
            python tools/convert_checkpoint/ds_to_universal.py
            --input_folder  {output_dir}/checkpoints/global_step{step}
            --output_folder {output_dir}/checkpoints/global_step{step}_universal
        """.split()
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] + cmd)); die

        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        self.assertIn("Convert DeepSpeed Checkpoint to Universal Checkpoint", cs.out)

    def resume_from_checkpoint(self, output_dir, tp_size=1, pp_size=1, dp_size=1):
        src_dir = self.src_dir
        script = [f"{src_dir}/pretrain_gpt.py"]

        args, ds_args, num_gpus = self.get_config(output_dir, tp_size, pp_size, dp_size)
        launcher = get_launcher(num_gpus)
        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] + cmd)); die

        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # test checkpoint loading
        self.assertIn(f"successfully loaded checkpoint from {output_dir}/checkpoints", cs.out)

        # test reports
        self.assertIn("consumed samples", cs.out)

        # test checkpoint saving
        self.assertIn("successfully saved checkpoint at iteration", cs.out)
        return cs.out

    def resume_from_universal_checkpoint(self, output_dir, tp_size=1, pp_size=1, dp_size=1,
                                         n_iters=None, exit_interval=None, save_interval=None,
                                         skip_train=False):
        src_dir = self.src_dir
        script = [f"{src_dir}/pretrain_gpt.py"]

        args, ds_args, num_gpus = self.get_config(output_dir, tp_size, pp_size, dp_size,
                                                  n_iters=n_iters, exit_interval=exit_interval,
                                                  save_interval=save_interval,
                                                  skip_train=skip_train)
        launcher = get_launcher(num_gpus)
        extra_args = ["--universal-checkpoint"]
        if skip_train:
            extra_args.append("--skip-train")

        cmd = launcher + script + args + ds_args + extra_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] + cmd)); die

        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # test checkpoint loading
        self.assertIn(f"successfully loaded checkpoint from {output_dir}/checkpoints", cs.out)

        # test reports
        if not skip_train:
            self.assertIn("consumed samples", cs.out)

        # test checkpoint saving
        self.assertIn("successfully saved checkpoint at iteration", cs.out)
        return cs.out

    @staticmethod
    def copy_checkpoint(src_ckp_root, dst_ckp_root, ckp_name, is_universal=False):
        src_root = os.path.join(src_ckp_root, 'checkpoints')
        dst_root = os.path.join(dst_ckp_root, 'checkpoints')
        os.makedirs(dst_root, exist_ok=True)
        src_folder = os.path.join(src_root, ckp_name)
        dst_folder = os.path.join(dst_root, ckp_name)
        shutil.copytree(src=src_folder, dst=dst_folder)
        latest_filename = 'latest_universal' if is_universal else 'latest'
        dst_latest = os.path.join(dst_root, latest_filename)
        with open(dst_latest, "w") as f:
            f.write(ckp_name)

    @require_torch_multi_gpu
    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def _test_checkpoint_reshaping_main(self, src, tgt):
        # this test needs at least 2 gpus - if there are more gpus it will do more extensive testing

        tp_size_src, pp_size_src, dp_size_src = list(map(int, src.split('_')))
        tp_size_tgt, pp_size_tgt, dp_size_tgt = list(map(int, tgt.split('_')))

        n_gpus = get_gpu_count()
        n_gpus_src = tp_size_src * pp_size_src * dp_size_src
        n_gpus_tgt = tp_size_tgt * pp_size_tgt * dp_size_tgt

        if n_gpus_src > n_gpus:
            pytest.skip(f"the test requires {n_gpus_src} gpus for source topology but have only {n_gpus}")
        if n_gpus_tgt > n_gpus:
            pytest.skip(f"the test requires {n_gpus_tgt} gpus for target topology but have only {n_gpus}")

        output_dir = self.get_auto_remove_tmp_dir("./xxx", after=False)

        # 1. train with initial topology defined in the first arg of params
        self.train_checkpoint(output_dir, tp_size=tp_size_src, pp_size=pp_size_src, dp_size=dp_size_src)

        # 2. convert checkpoint to universal checkpoint (topology )
        self.convert_checkpoint_to_universal(output_dir=output_dir, step=1)

        # 3. check we can resume training from a reshaped checkpoint to the target topology - the last arg of params
        self.resume_from_universal_checkpoint(output_dir, tp_size=tp_size_tgt, pp_size=pp_size_tgt, dp_size=dp_size_tgt)

    @require_torch_multi_gpu
    def _test_checkpoint_reshaping_empty_dir(self):

        output_dir = self.get_auto_remove_tmp_dir()
        with self.assertRaises(RuntimeError):
            self.convert_checkpoint_to_universal(output_dir=output_dir, step=1)

    @require_torch_multi_gpu
    def test_checkpoint_reshaping_2x2x2_to_2x2x1_to_2x2x2(self):
        # this test needs at least 8 gpus

        tp_size_src, pp_size_src, dp_size_src = 2, 2, 2
        tp_size_tgt, pp_size_tgt, dp_size_tgt = 2, 2, 1

        n_gpus = get_gpu_count()
        n_gpus_src = tp_size_src * pp_size_src * dp_size_src
        n_gpus_tgt = tp_size_tgt * pp_size_tgt * dp_size_tgt
        n_required_gpus = max(n_gpus_src, n_gpus_tgt)
        if n_required_gpus > n_gpus:
            pytest.skip(f"the test requires {n_required_gpus} gpus but have only {n_gpus}")

        root_dir = self.get_auto_remove_tmp_dir(after=True)
        output_2x2x2_dir = os.path.join(root_dir, 'topo_2x2x2')
        output_2x2x1_dir = os.path.join(root_dir, 'topo_2x2x1')
        output_2x2x2_final_dir = os.path.join(root_dir, 'topo_2x2x2_final')

        total_n_iters = 20
        checkpoint_iter = total_n_iters // 2

        # 1. train with initial 2x2x2 topology
        out = self.train_checkpoint(output_2x2x2_dir,
                                    tp_size=tp_size_src,
                                    pp_size=pp_size_src,
                                    dp_size=dp_size_src,
                                    n_iters=total_n_iters,
                                    exit_interval=total_n_iters + 1,
                                    save_interval=checkpoint_iter)

        try:
            orig_2x2x2_test_loss = float(re.search(
                'test data \| lm loss value: (\d+\.\d+E+\++\d+)', out).group(1))
        except AttributeError:
            assert False, 'Not found test data loss in original 2x2x2 training'

        # 2. convert 2x2x2 checkpoint to universal checkpoint
        self.convert_checkpoint_to_universal(output_dir=output_2x2x2_dir, step=checkpoint_iter)

        # 3. copy 2x2x2 universal checkpoint (step 10) to 2x2x1
        univ_ckp_name = f'global_step{checkpoint_iter}_universal'
        self.copy_checkpoint(src_ckp_root=output_2x2x2_dir,
                             dst_ckp_root=output_2x2x1_dir,
                             ckp_name=univ_ckp_name,
                             is_universal=True)

        # 3. use trainer to convert from universal to 2x2x1:
        #   3.1. load universal checkpoint
        #   3.1. skip actual training
        #   3.1. save checkpoint for 2x2x1 topology
        self.resume_from_universal_checkpoint(output_2x2x1_dir,
                                              tp_size=tp_size_tgt,
                                              pp_size=pp_size_tgt,
                                              dp_size=dp_size_tgt,
                                              n_iters=total_n_iters,
                                              exit_interval=checkpoint_iter,
                                              save_interval=total_n_iters,
                                              skip_train=True)

        # 4. copy 2x2x1 checkpoint (step 10) to 2x2x2_final
        ckp_name = f'global_step{checkpoint_iter}'
        self.copy_checkpoint(src_ckp_root=output_2x2x1_dir,
                             dst_ckp_root=output_2x2x2_final_dir,
                             ckp_name=ckp_name,
                             is_universal=False)

        # 5. convert 2x2x1 step 10 checkpoint to universal checkpoint
        self.convert_checkpoint_to_universal(output_dir=output_2x2x2_final_dir, step=checkpoint_iter)

        # 6. Load from universal created from 2x2x1 and resume training till end
        out = self.resume_from_universal_checkpoint(output_2x2x2_final_dir,
                                                    tp_size=tp_size_src,
                                                    pp_size=pp_size_src,
                                                    dp_size=dp_size_src,
                                                    n_iters=total_n_iters,
                                                    exit_interval=total_n_iters + 1,
                                                    save_interval=total_n_iters)
        try:
            final_2x2x2_test_loss = float(re.search(
                'test data \| lm loss value: (\d+\.\d+E+\++\d+)', out).group(1))
        except AttributeError:
            assert False, 'Not found test data loss in final 2x2x2 training'

        # 7. Verify same test loss for original training and final training
        assert orig_2x2x2_test_loss == final_2x2x2_test_loss

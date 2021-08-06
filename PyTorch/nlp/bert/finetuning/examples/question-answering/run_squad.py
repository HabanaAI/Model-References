# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import glob
import logging
import os
import sys
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

transformers_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../src"))
sys.path.append(transformers_path)

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def compute_position_ids(input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]
    position_ids_seq = torch.arange(seq_length, dtype=torch.int32)
    position_ids_ = position_ids_seq.unsqueeze(0).expand(input_shape)
    position_ids = position_ids_.contiguous()
    return position_ids

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def enable_tracing():
    torch._C._debug_set_autodiff_subgraph_inlining(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    try:
            import habana_frameworks.torch.core as htcore
    except ImportError:
            assert False,"Could Not import habana_frameworks.torch.core"

    htcore.enable()
    htcore.remove_inplace_ops()


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if args.use_habana and args.use_fused_adam:
        try:
            from hb_custom import FusedAdamW
        except ImportError:
            raise ImportError("Please install hb_custom.")
        optimizer = FusedAdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.use_jit_trace:
        enable_tracing()

    if args.use_lazy_mode:
        try:
            import habana_frameworks.torch.core as htcore
        except ImportError:
            assert False, "Could Not import habana_frameworks.torch.core"

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1 and not args.use_jit_trace:
        if args.use_habana:
            model = torch.nn.parallel.DistributedDataParallel(
            model, bucket_cap_mb=230, find_unused_parameters=True, gradient_as_bucket_view=True
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
            )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    is_model_traced = False
    loss_list = []

    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    for params in model.parameters():
        params.grad = None
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)
    if args.use_habana and not args.use_jit_trace:
        if args.use_fused_clip_norm:
            try:
                from hb_custom import FusedClipNorm
            except ImportError:
                raise ImportError("Please install hb_custom.")
            FusedNorm = FusedClipNorm(model.parameters(), args.max_grad_norm)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            ## Habana doesn't support Long tensors
            ## Hence we need to convert start and end positions to int
            if args.use_habana:
                batch[0] = batch[0].to(dtype=torch.int32)
                batch[1] = batch[1].to(dtype=torch.int32)
                batch[2] = batch[2].to(dtype=torch.int32)
                batch[3] = batch[3].to(dtype=torch.int32)
                batch[4] = batch[4].to(dtype=torch.int32)

            position_ids_cpu = compute_position_ids(batch[0])
            device = args.device
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            position_ids = position_ids_cpu.to(args.device)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
                "position_ids": position_ids
            }

            input_keys = ('input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions', "position_ids")
            input_dict = {k: inputs[k] for k in input_keys if k in inputs}
            target = inputs['end_positions']
            tensor_dummy = torch.zeros(1).to(device)


            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if args.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )
            if args.use_jit_trace and is_model_traced == False:
                model_trace = torch.jit.trace(model, (batch[0], batch[1], batch[2], position_ids, tensor_dummy, tensor_dummy, batch[3], batch[4], tensor_dummy, tensor_dummy), check_trace=False)
                is_model_traced = True
                if args.use_habana:
                    if args.use_fused_clip_norm:
                        try:
                            from hb_custom import FusedClipNorm
                        except ImportError:
                            raise ImportError("Please install hb_custom.")
                        FusedNorm = FusedClipNorm(model_trace.parameters(), args.max_grad_norm)
                if args.local_rank != -1:
                    if args.use_habana:
                        model_trace = torch.nn.parallel.DistributedDataParallel(
                            model_trace, bucket_cap_mb=230, find_unused_parameters=True, gradient_as_bucket_view=True
                        )
            if args.use_jit_trace:
                outputs = model_trace(batch[0], batch[1], batch[2], position_ids, tensor_dummy, tensor_dummy, batch[3], batch[4], tensor_dummy, tensor_dummy)
            else:
                outputs = model(**inputs)            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.local_rank != -1:
                if mpi_comm is not None:
                  mpi_comm.Barrier()
                else:
                  torch.distributed.barrier()

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if args.use_lazy_mode:
                htcore.mark_step()

            if args.local_rank != -1:
                if mpi_comm is not None:
                  mpi_comm.Barrier()
                else:
                  torch.distributed.barrier()

            loss_list.append(loss)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Increment the global step
                global_step += 1

                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    if args.use_habana:
                        if args.use_fused_clip_norm:
                            if args.use_jit_trace:
                                FusedNorm.clip_norm(model_trace.parameters())
                            else:
                                FusedNorm.clip_norm(model.parameters())
                        else:
                            if args.hmp:
                                from hmp import hmp
                                with hmp.disable_casts():
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if args.use_habana and args.hmp and not(args.use_fused_adam):
                    from hmp import hmp
                    with hmp.disable_casts():
                        optimizer.step()
                else:
                    optimizer.step()
                scheduler.step()  # Update learning rate schedule
                if args.use_jit_trace:
                    for param in model_trace.parameters():
                        param.grad = None
                else:
                    for param in model.parameters():
                        param.grad = None

                if args.use_lazy_mode:
                    htcore.mark_step()


                if args.local_rank != -1:
                    if mpi_comm is not None:
                        mpi_comm.Barrier()
                    else:
                        torch.distributed.barrier()

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    for loss_t in loss_list :
                        tr_loss += loss_t.item()

                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        result = evaluate(args, model, tokenizer)
                        for key, value in result.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        result = dict(("global_step-{}_".format(global_step)+k, v) for k, v in result.items())
                        logger.info("Results: {}".format(result))
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
                    # Report the loss
                    logger.info("Global Step: %s, Loss: %s", global_step, loss.item())
                    loss_list.clear()

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))

                    ## Saving model, optimizer and scheduler checkpoints for HPU
                    if args.use_habana:
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, "module") else model
                        d = next(model_to_save.parameters()).device
                        if (d != torch.device("cpu")):
                            import copy
                            model_to_save_clone = copy.deepcopy(model_to_save)
                            model_to_save_clone.to(torch.device("cpu"))
                            model_to_save_clone.save_pretrained(output_dir)
                            torch.save(model_to_save_clone.state_dict(), os.path.join(output_dir, "model.pt"))
                        logger.info("Saving HPU model checkpoint to %s", output_dir)
                        param_groups_copy = optimizer.state_dict()['param_groups']
                        state_dict_copy = {}
                        for st_key, st_val in optimizer.state_dict()['state'].items():
                            st_val_copy={}
                            for k, v in st_val.items():
                                if isinstance(v, torch.Tensor):
                                    st_val_copy[k] = v.to('cpu')
                                else:
                                    st_val_copy[k] = v
                                state_dict_copy[st_key] = st_val_copy
                        optim_dict = {}
                        optim_dict['state'] = state_dict_copy
                        optim_dict['param_groups'] = param_groups_copy
                        torch.save(optim_dict, os.path.join(output_dir, "optimizer.pt"))
                        logger.info("Saving HPU optimizer state to %s", output_dir)
                    else:
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    if args.use_jit_trace:
        enable_tracing()

    if args.use_lazy_mode:
        try:
            import habana_frameworks.torch.core as htcore
        except ImportError:
            assert False, "Could Not import habana_frameworks.torch.core"

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    is_eval_traced = False

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        ## Habana doesn't support Long tensors
        ## Hence we need to convert start and end positions to int
        if args.use_habana:
            batch[0] = batch[0].to(dtype=torch.int32)
            batch[1] = batch[1].to(dtype=torch.int32)
            batch[2] = batch[2].to(dtype=torch.int32)

        position_ids_cpu = compute_position_ids(batch[0])
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        position_ids = position_ids_cpu.to(args.device)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "position_ids": position_ids,
            }
            tensor_dummy = torch.zeros(1).to(args.device)

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )
            if args.use_jit_trace and is_eval_traced == False:
                model_trace = torch.jit.trace(model, (batch[0], batch[1], batch[2], position_ids, tensor_dummy, tensor_dummy, tensor_dummy, tensor_dummy, tensor_dummy, tensor_dummy), check_trace=False)
                is_eval_traced = True
                model_trace.eval()
            if args.use_jit_trace:
                outputs = model_trace(batch[0], batch[1], batch[2], position_ids, tensor_dummy, tensor_dummy, tensor_dummy, tensor_dummy, tensor_dummy, tensor_dummy)
            else:
                outputs = model(**inputs)

            if args.use_lazy_mode:
                htcore.mark_step()

            feature_indices = feature_indices.to("cpu")


        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
            else:
                examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--use_habana", action="store_true", help="Whether not to use Habana device when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--use_jit_trace", action='store_true', default=False, help='run with torch jit trace mode')
    parser.add_argument('--hmp', dest='hmp', action='store_true', help='enable hmp mode')
    parser.add_argument('--hmp_bf16', default='', help='path to bf16 ops list in hmp O1 mode')
    parser.add_argument('--hmp_fp32', default='', help='path to fp32 ops list in hmp O1 mode')
    parser.add_argument('--hmp_opt_level', default='O1', help='choose optimization level for hmp')
    parser.add_argument('--hmp_verbose', action='store_true', help='enable verbose mode for hmp')
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

    parser.add_argument("--use_fused_adam", action="store_true", help="Whether to use fused adamw on habana device")
    parser.add_argument("--use_fused_clip_norm", action="store_true", help="Whether to use fused clip norm on habana device")
    parser.add_argument('--use_lazy_mode', action='store_true', help='run model in lazy execution mode')

    args = parser.parse_args()

    os.environ["MAX_WAIT_ATTEMPTS"] = "50"
    os.environ["RUN_TPC_FUSER"] = "1"
    os.environ["HCL_CPU_AFFINITY"] = "1"
    os.environ["PT_HPU_PGM_ENABLE_CACHE"] = "15"

    if args.use_jit_trace:
        os.environ["PT_HPU_ENABLE_GRAPHMODE_LAYERNORM_FUSION"] = "1"
        real_path = os.path.realpath(__file__)
        demo_config_path = os.path.dirname(real_path)
        os.environ["PT_HPU_GRAPH_FUSION_OPS_FILE"] = demo_config_path + "/../../../BERT_Fusion_Ops.txt"

    if args.use_lazy_mode:
        os.environ["PT_HPU_LAZY_MODE"] = "1"
        os.environ["PT_HPU_LOWER_AS_STRIDED"] = "1"

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training

    if args.use_habana:
        from habana_frameworks.torch.utils.library_loader import load_habana_module
        load_habana_module()
        device = torch.device("hpu")

        try:
            global mpi_comm
            from mpi4py import MPI
            mpi_comm = MPI.COMM_WORLD
            args.world_size = mpi_comm.Get_size()
            if args.world_size > 1:
                args.rank = mpi_comm.Get_rank()
                if args.local_rank == -1:
                    args.local_rank = args.rank
            else:
                mpi_comm = None
                raise('Not an MPI run')
        except Exception as e:
            mpi_comm = None
            if 'WORLD_SIZE' in os.environ and 'RANK' in os.environ and 'LOCAL_RANK' in os.environ:
                args.world_size = int(os.environ["WORLD_SIZE"])
                args.rank       = int(os.environ["RANK"])
                args.local_rank = int(os.environ["LOCAL_RANK"])
            elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ and 'OMPI_COMM_WORLD_SIZE' in os.environ:
                args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
                args.local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
                args.rank       = args.local_rank
            else:
                print("Single node run")

        if args.local_rank == -1:
            args.n_gpu = 0
        else:
            if os.getenv('MASTER_ADDR') is None:
                os.environ['MASTER_ADDR'] = 'localhost'
            if os.getenv('MASTER_PORT') is None:
                os.environ['MASTER_PORT'] = '12355'
            args.dist_backend = 'hcl'
            os.environ["ID"] = str(args.local_rank)
            import habana_torch_hcl
            torch.distributed.init_process_group(args.dist_backend, rank=args.local_rank, world_size=args.world_size)
            args.n_gpu = 1


        if args.hmp:
            print(args.hmp_bf16)
            from hmp import hmp
            hmp.convert(opt_level=args.hmp_opt_level, bf16_file_path=args.hmp_bf16,
                    fp32_file_path=args.hmp_fp32, isVerbose=args.hmp_verbose)


    elif args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        if os.getenv('MASTER_ADDR') is None:
            os.environ['MASTER_ADDR'] = 'localhost'
        if os.getenv('MASTER_PORT') is None:
            os.environ['MASTER_PORT'] = '12355'
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForQuestionAnswering.from_pretrained(args.output_dir)  # , force_download=True)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            if global_step.isnumeric() == False:
                global_step = ""
            model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)  # , force_download=True)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    if args.use_lazy_mode:
        os.environ.pop("PT_HPU_LAZY_MODE")

    return results


if __name__ == "__main__":
    main()

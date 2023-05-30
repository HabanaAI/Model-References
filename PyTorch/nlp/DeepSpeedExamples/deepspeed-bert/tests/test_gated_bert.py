# coding=utf-8
# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

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

""" Test Gated BERT """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import h5py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import modeling
from torch.optim.adamw import AdamW as AdamW

try:
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu

    HTCORE_EXISTS = True
except:
    HTCORE_EXISTS = False


def create_pretraining_dataset(input_file, max_pred_length, max_samples, batch_size):
    train_data = PretrainingDataset(input_file=input_file, max_pred_length=max_pred_length, max_samples=max_samples)
    train_dataloader = DataLoader(train_data, sampler=None,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  pin_memory=True)
    return train_dataloader


class PretrainingDataset(Dataset):
    def __init__(self, input_file, max_pred_length, max_samples=-1):
        super().__init__()
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:max_samples]) for key in keys]
        f.close()

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.inputs[0])

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, next_sentence_labels]


class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels):
        masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
        next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .hdf5 files  for the task.")
    parser.add_argument("--use_hpu",
                        action='store_true',
                        help='Use GPU. If not specified, use GPU')
    parser.add_argument("--use_lazy_mode",
                        action='store_true',
                        help='Use HPU lazy mode. If not specified, use HPU eager mode')
    args = parser.parse_args()
    return args


def create_model(args, device, config):
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForPreTraining(config)
    model.to(device=device)

    # BERT modeling  uses weight sharing between word embedding and prediction decoder.
    # So make sure the storage is pointing properly even after model is moved to device.
    if args.use_hpu:
        model.cls.predictions.decoder.weight = model.bert.embeddings.word_embeddings.weight
    return model


def setup_training(args):
    if args.use_hpu:
        if not args.use_lazy_mode:
            os.environ["PT_HPU_LAZY_MODE"] = "2"

    device = torch.device("hpu") if args.use_hpu else torch.device("cuda")
    return device


def get_first_data_file(args):
    data_file = os.path.join(args.input_dir, "books_wiki_en_corpus_training_0.hdf5")
    return data_file


def train(args, model, config, device, n_steps, batch_size):
    # setup data loader
    data_file = get_first_data_file(args)
    n_samples = n_steps * batch_size
    train_dataloader = create_pretraining_dataset(data_file,
                                                  max_pred_length=20,
                                                  max_samples=n_samples,
                                                  batch_size=batch_size)

    # create optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=0.0001)

    # setup criterion
    criterion = BertPretrainingCriterion(config.vocab_size)

    model.train()
    loss_list = []
    for step, batch in enumerate(train_dataloader):
        batch = [t.to(device) for t in batch]
        input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
        prediction_scores, seq_relationship_score = model(input_ids=input_ids,
                                                          token_type_ids=segment_ids,
                                                          attention_mask=input_mask)
        loss = criterion(prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels)
        loss_list.append(loss.item())
        loss.backward()

        htcore.mark_step() if args.use_hpu and args.use_lazy_mode else None
        optimizer.step()
        htcore.mark_step() if args.use_hpu and args.use_lazy_mode else None
    return loss_list


def create_bert_tiny_config(n_layers, gated_layers):
    config = {
        "hidden_size": 64,
        "hidden_act": "gelu",
        "initializer_range": 0.02,
        "vocab_size": 30522,
        "hidden_dropout_prob": 0.,
        "num_attention_heads": 1,
        "type_vocab_size": 2,
        "max_position_embeddings": 512,
        "num_hidden_layers": n_layers,
        "intermediate_size": 64,
        "attention_probs_dropout_prob": 0.,
        "layer_norm_large_model": True,
        "gated_layers": gated_layers
    }
    return modeling.BertConfig.from_dict(config)


def create_layer_mapping(n_layers, gated_model_n_layers, gated_layers, verbose =False):
    gated_model_all_layers = list(range(gated_model_n_layers))
    gated_model_non_gated_layers = list(set(gated_model_all_layers) - set(gated_layers))
    assert len(gated_model_non_gated_layers) == n_layers, "Invalid non-gated vs. gated configuration"
    gated_model_non_gated_layers.sort()
    mapping = {i: gated_model_non_gated_layers[i] for i in range(n_layers)}
    if verbose:
        print(f"Layer mapping': {mapping}")
    return mapping


def copy_params(src_model, dst_model, layer_map, verbose=False):
    layer_prefix = 'bert.encoder.layer.'
    layer_id_token_idx = len(layer_prefix.split('.')) - 1
    dst_model_params = dict(dst_model.named_parameters())
    for name, p in src_model.named_parameters():
        if name.startswith(layer_prefix):
            tokens = name.split('.')
            src_idx = int(tokens[layer_id_token_idx])
            tgt_idx = layer_map[src_idx]
            tokens[layer_id_token_idx] = str(tgt_idx)
            dst_name = '.'.join(tokens)
        else:
            dst_name = name
        dst_model_params[dst_name].data = p.data.clone()
        if verbose:
            print(f"Copy data from {name} to {dst_name}")


def reset_random(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def test_configuration(args, device, model_n_layers, model_g_n_layers, model_g_gated_layers):
    n_steps = 10
    batch_size = 16

    # create both gated and non-gated models; copy initial weights to non-gated
    reset_random()
    config = create_bert_tiny_config(n_layers=model_n_layers, gated_layers=())
    model = create_model(args, device, config)
    config_g = create_bert_tiny_config(n_layers=model_g_n_layers, gated_layers=model_g_gated_layers)
    model_g = create_model(args, device, config_g)
    layer_map = create_layer_mapping(config.num_hidden_layers, config_g.num_hidden_layers, config_g.gated_layers)
    copy_params(model, model_g, layer_map)

    # train non-gated
    reset_random()
    losses = train(args, model, config, device, n_steps=n_steps, batch_size=batch_size)

    # train gated
    reset_random()
    losses_g = train(args, model_g, config_g, device, n_steps=n_steps, batch_size=batch_size)

    # verify
    ok = losses == losses_g
    return ok


def test_gated_bert_tiny():
    args = parse_arguments()

    if args.use_hpu:
        assert HTCORE_EXISTS, 'Habana htcore not imported'

    device = setup_training(args)

    test_configs = [
        # model_n_layers        model_g_n_layers            model_g_gated_layers
        (1,                     2,                          (1,)),
        (2,                     4,                          (1, 2,)),
        (3,                     10,                         (1, 2, 3, 4, 5, 6, 8, )),
    ]

    # test and verify
    for test_config in test_configs:
        model_n_layers, model_g_n_layers, model_g_gated_layers = test_config
        ok = test_configuration(args, device, model_n_layers, model_g_n_layers, model_g_gated_layers)
        res = "PASS" if ok else "FAIL"
        print(f"n_layers={model_n_layers} gated_model_n_layers={model_g_n_layers} gated={model_g_gated_layers} - {res}")

    if args.use_lazy_mode:
        os.environ.pop("PT_HPU_LAZY_MODE")


if __name__ == "__main__":
    test_gated_bert_tiny()

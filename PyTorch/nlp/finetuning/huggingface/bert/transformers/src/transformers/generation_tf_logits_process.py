# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team
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

import inspect
from typing import List

import numpy as np
import tensorflow as tf

from .file_utils import add_start_docstrings
from .tf_utils import set_tensor_by_indices_to_value
from .utils.logging import get_logger


logger = get_logger(__name__)


TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        scores (`tf.Tensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search
        kwargs:
            Additional logits processor specific kwargs.

    Return:
        `tf.Tensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.
"""


class TFLogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    @add_start_docstrings(TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor) -> tf.Tensor:
        """TF method for processing logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class TFLogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    @add_start_docstrings(TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor) -> tf.Tensor:
        """TF method for warping logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class TFLogitsProcessorList(list):
    """
    This class can be used to create a list of [`TFLogitsProcessor`] to subsequently process a `scores` input tensor.
    This class inherits from list and adds a specific *__call__* method to apply each [`TFLogitsProcessor`] to the
    inputs.
    """

    @add_start_docstrings(TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, **kwargs) -> tf.Tensor:
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
        return scores


class TFTemperatureLogitsWarper(TFLogitsWarper):
    r"""
    [`TFLogitsWarper`] for temperature (exponential scaling output probability distribution).

    Args:
        temperature (`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")

        self.temperature = temperature

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor) -> tf.Tensor:
        scores = scores / self.temperature
        return scores


class TFTopKLogitsWarper(TFLogitsWarper):
    r"""
    [`TFLogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor) -> tf.Tensor:
        top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.shape[-1])  # Safety check
        # Boolean mask containing all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < tf.math.top_k(scores, k=top_k)[0][..., -1:]
        next_scores = tf.where(indices_to_remove, self.filter_value, scores)
        return next_scores


class TFTopPLogitsWarper(TFLogitsWarper):
    """
    [`TFLogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to <= prob_cut_off.

    Args:
        top_p (`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to `top_p` or higher are kept
            for generation.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_p, float) or (top_p < 0 or top_p > 1.0):
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor) -> tf.Tensor:
        topk_scores, topk_indices = tf.math.top_k(scores, scores.shape[-1])

        mask_scores = tf.fill(scores.shape, self.filter_value)
        cumulative_probs = tf.math.cumsum(tf.nn.softmax(topk_scores, axis=-1), axis=-1)
        score_mask = cumulative_probs < self.top_p

        # Also include the token that is higher than top_p (the first false = shift and insert a True on the left)
        score_mask = tf.concat((tf.ones([score_mask.shape[0], 1], dtype=tf.bool), score_mask[:, :-1]), axis=-1)

        # Ensure min tokens to keep
        score_mask = tf.concat(
            (
                tf.ones([score_mask.shape[0], self.min_tokens_to_keep], dtype=tf.bool),
                score_mask[:, self.min_tokens_to_keep :],
            ),
            axis=-1,
        )

        # Mask the values that do not fit the criteria
        topk_next_scores = tf.where(score_mask, topk_scores, mask_scores)

        # Undo the topk sorting: converts the 2D matrix of per-row original indices of shape (batch_size, vocab_size)
        # to a 3D tensor of shape (batch_size, vocab_size, 2) containing the original score coordinate, from which we
        # can scatter (i.e. `scatter_indices[row, col, :]` is a tensor containing `[row, topk_indices[row, col]]`)
        scatter_rows = tf.tile(tf.expand_dims(tf.range(topk_indices.shape[0]), axis=-1), [1, topk_indices.shape[-1]])
        scatter_indices = tf.stack((scatter_rows, topk_indices), axis=-1)
        next_scores = tf.scatter_nd(scatter_indices, topk_next_scores, shape=topk_next_scores.shape)

        return next_scores


class TFMinLengthLogitsProcessor(TFLogitsProcessor):
    r"""
    [`TFLogitsProcessor`] enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`int`):
            The id of the *end-of-sequence* token.
    """

    def __init__(self, min_length: int, eos_token_id: int):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a positive integer, but is {min_length}")

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor) -> tf.Tensor:
        # create boolean flag to decide if min length penalty should be applied
        cur_len = input_ids.shape[-1]
        apply_penalty = 1 - tf.clip_by_value(cur_len - self.min_length, 0, 1)

        # TODO(Matt) - this if statement has to be rewritten for XLA. Leaving it now though since
        # generate is not XLA - compileable anyways
        if apply_penalty:
            eos_token_id_mask = tf.broadcast_to(tf.range(scores.shape[-1]) == self.eos_token_id, scores.shape)
            scores = set_tensor_by_indices_to_value(scores, eos_token_id_mask, float("-inf"))

        return scores


class TFRepetitionPenaltyLogitsProcessor(TFLogitsProcessor):
    r"""
    [`TFLogitsProcessor`] enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    """

    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    def _create_score_penalties(self, input_ids, logits):
        # create logit penalties for already seen input_ids
        token_penalties = np.ones(logits.shape)
        prev_input_ids = [np.unique(input_id) for input_id in input_ids.numpy()]
        for i, prev_input_id in enumerate(prev_input_ids):
            logit_penalized = logits[i].numpy()[prev_input_id]
            logit_penalties = np.zeros(logit_penalized.shape)
            # if previous logit score is < 0 then multiply repetition penalty else divide
            logit_penalties[logit_penalized < 0] = self.penalty
            logit_penalties[logit_penalized > 0] = 1 / self.penalty
            np.put(token_penalties[i], prev_input_id, logit_penalties)
        return tf.convert_to_tensor(token_penalties, dtype=tf.float32)

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor) -> tf.Tensor:

        score_penalties = self._create_score_penalties(input_ids, scores)

        scores = tf.math.multiply(scores, score_penalties)

        return scores


class TFNoBadWordsLogitsProcessor(TFLogitsProcessor):
    """
    [`TFLogitsProcessor`] that enforces that specified sequences will never be sampled.

    Args:
        bad_words_ids (`List[List[int]]`):
            List of list of token ids that are not allowed to be generated. In order to get the tokens of the words
            that should not appear in the generated text, use `tokenizer(bad_word, add_prefix_space=True).input_ids`.
        eos_token_id (`int`):
            The id of the *end-of-sequence* token.
    """

    def __init__(self, bad_words_ids: List[List[int]], eos_token_id: int):

        if not isinstance(bad_words_ids, List) or len(bad_words_ids) == 0:
            raise ValueError(f"`bad_words_ids` has to be a non-emtpy list, but is {bad_words_ids}.")
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in bad_words_ids):
            raise ValueError(f"`bad_words_ids` has to be a list of lists, but is {bad_words_ids}.")
        if any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in bad_word_ids)
            for bad_word_ids in bad_words_ids
        ):
            raise ValueError(
                f"Each list in `bad_words_ids` has to be a list of positive integers, but is {bad_words_ids}."
            )

        self.bad_words_ids = bad_words_ids

    def calc_banned_bad_words_ids(self, prev_input_ids):
        banned_tokens = []

        def _tokens_match(prev_tokens, tokens):
            if len(tokens) == 0:
                # if bad word tokens is just one token always ban it
                return True
            if len(tokens) > len(prev_tokens):
                # if bad word tokens are longer than prev tokens they can't be equal
                return False

            if prev_tokens[-len(tokens) :] == tokens:
                # if tokens match
                return True
            else:
                return False

        for prev_input_ids_slice in prev_input_ids:
            banned_tokens_slice = []

            for banned_token_seq in self.bad_words_ids:
                assert (
                    len(banned_token_seq) > 0
                ), f"Banned words token sequences {self.bad_words_ids} cannot have an empty list"

                if _tokens_match(prev_input_ids_slice.numpy().tolist(), banned_token_seq[:-1]) is False:
                    # if tokens do not match continue
                    continue

                banned_tokens_slice.append(banned_token_seq[-1])

            banned_tokens.append(banned_tokens_slice)

        return banned_tokens

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor) -> tf.Tensor:

        vocab_size = scores.shape[-1]

        # calculate a list of banned tokens according to bad words
        banned_tokens = self.calc_banned_bad_words_ids(input_ids)

        banned_tokens_indices_mask = []
        for banned_tokens_slice in banned_tokens:
            banned_tokens_indices_mask.append(
                [True if token in banned_tokens_slice else False for token in range(vocab_size)]
            )

        scores = set_tensor_by_indices_to_value(
            scores, tf.convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool), -float("inf")
        )

        return scores


class TFNoRepeatNGramLogitsProcessor(TFLogitsProcessor):
    r"""
    [`TFLogitsProcessor`] that enforces no repetition of n-grams. See
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.
    """

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    def calc_banned_ngram_tokens(self, prev_input_ids, num_hypos, cur_len):
        # Copied from fairseq for no_repeat_ngram in beam_search
        if cur_len + 1 < self.ngram_size:
            # return no banned tokens if we haven't generated ngram_size tokens yet
            return [[] for _ in range(num_hypos)]
        generated_ngrams = [{} for _ in range(num_hypos)]
        for idx in range(num_hypos):
            gen_tokens = prev_input_ids[idx].numpy().tolist()
            generated_ngram = generated_ngrams[idx]
            for ngram in zip(*[gen_tokens[i:] for i in range(self.ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

        def _get_generated_ngrams(hypo_idx):
            # Before decoding the next token, prevent decoding of ngrams that have already appeared
            start_idx = cur_len + 1 - self.ngram_size
            ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].numpy().tolist())
            return generated_ngrams[hypo_idx].get(ngram_idx, [])

        banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]

        return banned_tokens

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor) -> tf.Tensor:

        batch_size, vocab_size = scores.shape
        cur_len = input_ids.shape[-1]
        banned_tokens = self.calc_banned_ngram_tokens(input_ids, batch_size, cur_len)

        # create banned_tokens boolean mask
        banned_tokens_indices_mask = []
        for banned_tokens_slice in banned_tokens:
            banned_tokens_indices_mask.append(
                [True if token in banned_tokens_slice else False for token in range(vocab_size)]
            )

        scores = set_tensor_by_indices_to_value(
            scores, tf.convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool), -float("inf")
        )

        return scores

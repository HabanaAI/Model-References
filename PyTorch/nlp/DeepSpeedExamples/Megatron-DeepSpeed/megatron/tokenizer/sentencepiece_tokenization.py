"""SentencePiece tokenizer."""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import os
import logging
import sentencepiece
import numpy as np

logger = logging.getLogger(__name__)


class SentencePieceTokenizer(object):
    def __init__(self, model_path):
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"model_path: {model_path} is invalid")

        logger.info(f"Loading SentencePiece model from {model_path}")
        self.tokenizer = sentencepiece.SentencePieceProcessor()
        self.tokenizer.Load(model_path)
        self.original_vocab_size = self.tokenizer.get_piece_size()
        self.vocab_size = self.tokenizer.get_piece_size()
        self.vocab = {_id: self.tokenizer.id_to_piece(_id) for _id in range(self.vocab_size)}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        """ Text to token ids """
        return self.text_to_ids(text)

    def decode(self, ids):
        """ Token ids to text """
        return self.ids_to_text(ids)

    def text_to_ids(self, text):
        return self.tokenizer.encode_as_ids(text)

    def ids_to_text(self, ids):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        return self.tokenizer.decode_ids(ids)

    def token_to_id(self, token):
        return self.tokenizer.piece_to_id(token)

    def ids_to_tokens(self, ids):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        return [self.tokenizer.id_to_piece(_id) for _id in ids]

    def tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            tokens = [tokens]
        ids = []
        for token in tokens:
            ids.append(self.token_to_id(token))
        return ids

    def tokens_to_text(self, tokens):
        return self.tokenizer.decode_pieces(tokens)

    def text_to_tokens(self, text):
        return self.tokenizer.encode_as_pieces(text)

    @property
    def pad_id(self):
        return self.tokenizer.pad_id()

    @property
    def bos_id(self):
        return self.tokenizer.bos_id()

    @property
    def eos_id(self):
        return self.tokenizer.eos_id()

    @property
    def unk_id(self):
        return self.tokenizer.unk_id()

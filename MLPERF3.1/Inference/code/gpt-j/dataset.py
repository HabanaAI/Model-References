from transformers import AutoTokenizer, BatchEncoding
from torch.nn.functional import pad

import utils
import torch

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class Dataset():
    def __init__(self, model_path, dataset_path, total_count_override=None, perf_count_override=None, add_padding=True, fake_data=False):
        print("Constructing QSL")

        self.model_path = model_path
        self.dataset_path = dataset_path
        self.add_padding = add_padding
        self.fake_data = fake_data

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=2048,
            padding_side="left",
            use_fast=True,)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.list_data_dict = utils.jload(self.dataset_path)

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        self.sources = [prompt_input.format_map(
            example) for example in self.list_data_dict]
        self.targets = [
            f"{example['output']}" for example in self.list_data_dict]

        self.source_encoded_input_ids, self.source_encoded_attn_masks = self.encode_samples()

        self.count = total_count_override or len(self.sources)
        self.perf_count = perf_count_override or self.count

    def encode_samples(self):
        def pad_tensor(tensor, value=0):
            max_length = 1919
            return pad(tensor, (max_length - tensor.shape[-1], 0), value=value)

        print("Encoding Samples")

        max_length = 1919
        min_length = 30
        total_samples = len(self.sources)

        source_encoded_input_ids = []
        source_encoded_attn_masks = []

        for i in range(total_samples):
            if not self.fake_data:
                source_encoded = self.tokenizer(self.sources[i], return_tensors="pt",
                                                padding=True, truncation=True,
                                                max_length=max_length)
            else:
                # Hack to generate a deterministic semi-random sequence without using random.*
                length = min_length + len(self.sources[i]) % (max_length - min_length)
                source_encoded = BatchEncoding({
                    'input_ids': torch.ones((1, length), dtype=torch.int64),
                    'attention_mask': torch.ones((1, length), dtype=torch.int64)})
            if self.add_padding:
                source_encoded.input_ids = pad_tensor(source_encoded.input_ids, self.tokenizer.pad_token_id)
                source_encoded.attention_mask = pad_tensor(source_encoded.attention_mask)
            source_encoded_input_ids.append(source_encoded.input_ids)
            source_encoded_attn_masks.append(source_encoded.attention_mask)

        return source_encoded_input_ids, source_encoded_attn_masks

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass

    def __del__(self):
        print("Finished destroying QSL.")

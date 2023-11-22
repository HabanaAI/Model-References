# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company

import random, itertools
from tqdm import tqdm
import numpy as np

def get_cdf(pdf):
    # list of tuples, each tuple is a k-v pair (k is the number, v its probability)
    sorted_key_pdf = sorted(pdf, key=lambda x:x[0]) # sort by keys
    run_sum = 0
    cdf = []
    for i in range(len(sorted_key_pdf)):
        k, v = sorted_key_pdf[i]
        run_sum += v
        assert run_sum <= 1 or (1-run_sum) < 0.000000001
        cdf += [(k, 1 if run_sum > 1 else run_sum)]
    last_elem = cdf[-1]
    assert last_elem[1] <= 1.0
    assert (1.0 - last_elem[1]) < 0.000001
    cdf[-1] = (cdf[-1][0], 1.0) # set the last elem to 1
    return cdf

def hist_to_tuplelist(pdf):
    inp_is_hist = type(pdf) == type({1:2})
    if inp_is_hist:
        # pdf is a histograms. values add to 1 and are positive
        pdf = [(k,pdf[k]) for k in pdf]
    return pdf, inp_is_hist

def format_convert(f):
    def helper(pdf, bs, aggregator):
        pdf, inp_is_hist = hist_to_tuplelist(pdf)
        out = f(pdf, bs, aggregator)
        if inp_is_hist:
            return {k:v for k,v in out}
        else:
            return out
    return helper

@format_convert
def aggregator_batch(pdf, bs, aggregator):
    assert aggregator in ['min', 'max']
    pdf = sorted(pdf, key=lambda x:x[0])
    cdf = get_cdf(pdf)
    assert len(cdf) == len(pdf)
    result = []
    for p, c in zip(pdf, cdf):
        kp, pval = p
        kc, cval = c
        assert kp == kc
        val = bs * pval * ((cval if aggregator == 'max' else (1-cval)) ** (bs-1))
        result += [(kp, val)]
    # the resulting pdf might be unnormalized, probably due to computational issues? normalizing it
    result_val_tot = sum([k[1] for k in result])
    result = [(k[0], k[1]/result_val_tot) for k in result]
    return result


def generate_random_gaussian():
    import numpy as np
    while True:
        x = np.random.normal(500, 50)
        if x < 2: # truncating it so that its not negative
            x = 2
        x = round(x) # its a discrete distribution, so rounding it off
        yield x

def gaussian(num_samples):
    return list(itertools.islice(generate_random_gaussian(), num_samples))

def batched_gaussian(orig_list, bs, aggregator):
    return [aggregator(orig_list[i * bs : (i+1) * bs]) for i in range(len(orig_list) // bs)]


def batch_by_formula(orig_list, bs, aggregator):
    count_hist = {}
    for item in orig_list:
        count_hist[item] = count_hist.get(item, 0) + 1
    total = sum(list(count_hist.values()))
    pdf_hist = {k:count_hist[k]/total for k in count_hist}
    return aggregator_batch(pdf_hist, bs, aggregator)

def sample_from_pdf(pdf, num_samples):
    pdf, _ = hist_to_tuplelist(pdf)
    nums = [k[0] for k in pdf]
    prob = [k[1] for k in pdf]
    return np.random.choice(nums, num_samples, p=prob)


def squad(bs=1, clip=None):
    print('Start squad bs =',bs)
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    import torch

    # Pad to max length sentence in each batch
    def collate(batch):
        def pad(item, val, maxlen):
            return torch.tensor([i + [val]*(maxlen-len(i)) for i in item])
        token = [k['token_type_ids'] for k in batch]
        attention = [k['attention_mask'] for k in batch]
        inp = [k['input_ids'] for k in batch]
        token_lens = [len(i) for i in token]
        # Find the max length sentence in this batch
        max_len = max(token_lens)
        assert token_lens == [len(i) for i in attention] == [len(i) for i in inp]
        return {'token_type_ids': pad(token, 0, max_len), 'attention_mask': pad(attention, 0, max_len), 'input_ids': pad(inp, 0, max_len)}


    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    squad_dataset = load_dataset('squad')
    tokenized_dataset = squad_dataset.map(lambda x: tokenizer(x['context']), batched=True)

    dt = DataLoader(tokenized_dataset['train'], batch_size=bs, num_workers=2, collate_fn=collate)
    lens = []
    for idx, data in tqdm(enumerate(dt)):
        lens += [data['input_ids'].shape[1]]
        if clip is not None and len(lens) >= clip:
            break
    print('Done squad bs =', bs)
    return lens
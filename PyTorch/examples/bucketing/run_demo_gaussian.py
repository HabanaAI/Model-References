# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company

from datasets_library import gaussian, batched_gaussian, batch_by_formula, sample_from_pdf
from plotting import plotter


if __name__ == '__main__':
    print("Plotting gaussian")
    num_samples = 100000
    bs = 4
    gs = gaussian(num_samples)
    orig = batched_gaussian(gs, 1, max)
    max_batch4 = batched_gaussian(gs, bs, max)
    min_batch4 = batched_gaussian(gs, bs, min)
    max_formula_batch4 = sample_from_pdf(batch_by_formula(gs, bs, 'max'), num_samples)
    min_formula_batch4 = sample_from_pdf(batch_by_formula(gs, bs, 'min'), num_samples)
    max_batch32 = batched_gaussian(gs, 8*bs, max)
    min_batch32 = batched_gaussian(gs, 8*bs, min)
    max_formula_batch32 = sample_from_pdf(batch_by_formula(gs, 8*bs, 'max'), num_samples)
    min_formula_batch32 = sample_from_pdf(batch_by_formula(gs, 8*bs, 'min'), num_samples)
    plotter([orig, max_batch4, max_formula_batch4, min_batch4, min_formula_batch4, max_batch32, max_formula_batch32, min_batch32, min_formula_batch32], 'gaussian.svg', ['original', 'bs4_max', 'bs4_max_formula', 'bs4_min', 'bs4_min_formula', 'bs32_max', 'bs32_max_formula', 'bs32_min', 'bs32_min_formula'])
## Table of Contents

* [Model-References](../../../../README.md)
* [Applying SDPA CustomOp to Bert NV](#applying-customops-to-a-real-training-model-example)

A brief description of Scaled Dot Product Attention (SDPA) kernel is provided in 
[FusedSDPA section](https://docs.habana.ai/en/latest/PyTorch/Python_Packages.html#hpex-kernels-fusedsdpa).

The usage of the SDPA is demonstrated through the BERT Fine tuning training model.
The changes required to invoke SDPA are available in `custom_fusedsdpa_op.patch`.
The BERT FT model can be patched with `custom_fusedsdpa_op.patch` and trained using SDPA.

Below are the steps to patch and run the BERT FT training script. The commands to run the 
training remain unmodified.

## Applying SDPA CustomOp to BERT Fine-Tuning

1. Apply the patch `custom_fusedsdpa_op.patch` to PyTorch/nlp/bert/modeling.py:
   - Go to the main directory in the repository.
   - Run `git apply --verbose PyTorch/examples/custom_op/custom_fusedsdpa/custom_fusedsdpa_op.patch`
2. Run the model.

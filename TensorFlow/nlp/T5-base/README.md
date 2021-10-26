## Model overview

T5, or Text-to-Text Transfer Transformer, is a Transformer based architecture that uses a text-to-text approach.
Every task – including translation, question answering, and classification – is cast as feeding the model text
as input and training it to generate some target text. This allows for the use of the same model,
loss function, hyperparameters, etc. across our diverse set of tasks. The changes compared to BERT include:

* adding a causal decoder to the bidirectional architecture.
* replacing the fill-in-the-blank cloze task with a mix of alternative pre-training tasks.

Source: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)

Code taken from [SnapThat's Github repo](https://github.com/snapthat/TF-T5-text-to-text/blob/master/snapthatT5/notebooks/TF-T5-%20Training.ipynb)

## Model changes
* patched `compute_loss` function due to missing cast to fp32 (the issue is solved by [transformers#12332](https://github.com/huggingface/transformers/pull/12332))
* removed unused `boolean_mask`
* added `load_habana_module`
* added TensorBoard training hooks
* added script for downloading the dataset and the pretrained model


# Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Setup and
Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please
answer the questions in the guide according to your preferences. This guide will
walk you through the process of setting up your system to run the model on
Gaudi.

Topology uses SQUAD dataset and pretrained weights provided by Huggingface. To download and preprocess the dataset run following command. It will also download pretrained model.

```bash
cd /root/Model-References/TensorFlow/nlp/T5-base
$PYTHON prepare_data.py /data/huggingface
```

When this command finishes the `/data/huggingface` should contain `squad` and `t5_base` subdirectories and we are ready to fine-tune the model.


## Training/Fine-Tuning

Running following command will fine-tune T5-base model on SQUAD dataset.

```bash
cd /root/Model-References/TensorFlow/nlp/T5-base
$PYTHON train.py --dtype bf16 --data_dir /data/huggingface --model_dir ./model
```

For more info about possible arguments run: `$PYTHON train.py --help`.


## Inference

When model is fine-tuned we can use it for inference. Run the following command to explore how model behaves:

```bash
$ $PYTHON inference.py --data_dir /data/huggingface --model_dir ./model
Provide context and ask model a question, for example:
Context: In 2019 Habana Labs announced its first AI accelerator. Gaudi, named after famous Catalan architect, was designed to accelerate training of deep neural networks in data centers.
Question: What is the name of the chip?
Answer:  <pad> Gaudi</s>
...
```

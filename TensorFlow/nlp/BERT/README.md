
# BERT Demo

Bidirectional Encoder Representations from Transformers (BERT) is a technique for natural language processing (NLP) pre-training developed by Google.
BERT was created and published in 2018 by Jacob Devlin and his colleagues from Google.
Google is leveraging BERT to better understand user searches.

The original English-language BERT model comes with two pre-trained general types: (1) the BERTBASE model, a 12-layer, 768-hidden, 12-heads, 110M parameter neural network architecture, and (2) the BERTLARGE model, a 24-layer, 1024-hidden, 16-heads, 340M parameter neural network architecture; both of which were trained on the BooksCorpus with 800M words, and a version of the English Wikipedia with 2,500M words.

There are 2 distinct sets of scripts for exercising BERT training.

## BERT Pre-Training

- Located in: `tensorflow-training/demo/bert/pretaining`
- Is a fork of: [NVIDIA BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT)
- Suited for datasets:
  - `bookswiki`
  - `overfit`
- Uses optimizer: **LAMB** ("Layer-wise Adaptive Moments optimizer for Batch training").
- Consists of 2 phases:
  - Phase 1 - **Masked Language Model** - where given a sentence, a randomly chosen word is guessed.
  - Phase 2 - **Next Sentence Prediction** - where the model guesses whether sentence B comes after sentence A
- The resulting (trained) model weights are language-specific (here: english) and has to be further "fitted" to do a specific task (with finetuning).
- Heavy-weight: the training takes several hours or days.

## BERT Fine-Tuning
- Located in: `tensorflow-training/demo/bert`
- Is a fork of: [Google Research BERT](https://github.com/google-research/bert)
- Suited for tasks:
  - `mrpc`: Microsoft Research Paraphrase Corpus (**MRPC**) is a paraphrase identification dataset, where systems aim to identify if two sentences are paraphrases of each other.
  - `squad`: Stanford Question Answering Dataset (**SQuAD**) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
- Uses optimizer: **AdamW** ("ADAM with Weight Decay Regularization").
- Bases on model weights trained with pretraining.
- Light-weight: the training takes a minute or so.

## Running the Training and Evaluation

```bash
./demo_bert -h  # for help
```

| Model      | Task & Dataset                           | Batch Size                                                | Command Line                                                       |
|------------|------------------------------------------|-----------------------------------------------------------|--------------------------------------------------------------------|
| BERT BASE  | Fine-Tuning for MRPC                     | 256 (32 per worker)                                       | `./demo_bert finetuning -d bf16 -m base -t mrpc -b 32 -s 128 -v`   |
| BERT LARGE | Fine-Tuning for MRPC                     | 512 (32 per worker)                                       | `./demo_bert finetuning -d bf16 -m large -t mrpc -b 64 -s 128 -v`  |
| BERT BASE  | Fine-Tuning for SQuAD                    | 256 (32 per worker)                                       | `./demo_bert finetuning -d bf16 -m base -t squad -b 32 -s 384 -v`  |
| BERT LARGE | Fine-Tuning for SQuAD                    | 192 (24 per worker)                                       | `./demo_bert finetuning -d bf16 -m large -t squad -b 24 -s 384 -v` |
| BERT LARGE | Pre-Training on BookCorpus and Wikipedia | Phase I: 512 (64 per worker), Phase II: 64 (8 per worker) | `./demo_bert pretraining -d bf16 -m large -b 64 8 -v`              |

Keep an eye on the opening log messages, as they show variables, which can be overridden by the environment. For instance, if you want to enable ART in HCL, just:
```bash
HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE=false ./demo_bert ...
```

Or to enable profiling with Synapse traces with SynapseLoggerHook:
```bash
HABANA_SYNAPSE_LOGGER=range ./demo_bert ...
```

## Multi-HLS Support

Let's assume you want to run BERT finetuning mrpc large on 2x HLS-1, i.e. _igk-kvm015-001-u18_ (10.211.169.54) and _igk-kvm016-001-u18_ (10.211.162.123), invoking a script from your VM.

- Make sure you that every of those three machines have the other's public keyes in _authorized_keys_, so every one of them can ssh to every other without need of entering the credentials (user/password).
- Define the distributed machine pool for OpenMPI:
```bash
export MULTI_HLS_IPS='10.211.169.54,10.211.162.123'
```
- Invoke BERT training script normally, remembering about `-v` parameter:
```
./demo_bert finetuning -d bf16 -m large -t mrpc -b 64 -s 128 -v
```

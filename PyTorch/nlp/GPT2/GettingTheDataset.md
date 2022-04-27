## GPT-2 train on OpenWebtext with gpt-2 tokenizer dict

### Requirements
>NOTE: Make sure all the requirements in the `requirements.txt` files are installed before downloading dataset.

### OpenWebText Dataset

(1) Create a directory to download and process the dataset
```bash
$ mkdir ./GPT2-data
$ cd GPT-data
```

(2) Download OpenWebText.tar.xz in above directory from:<br>
https://skylion007.github.io/OpenWebTextCorpus/

The above site provides a link to the below google drive:<br>
https://drive.google.com/drive/folders/1IaD_SIIB-K3Sij_-JjWoPy_UrWqQRdjx

(2) Once the OpenWebText.tar.xz file is available, untar it to `openwebtext` directory
```bash
tar -xf openwebtext.tar.xz
```
(3) Partition the xz files obtained in step-2 in to three directories:<br>
* Use the `partion_data.py` in GPT2 directory in below format and sequence which will move the file from input openwebtext directory to output `openwebtext-train`, `openwebtext-valid` and `openwebtext-test` directories
    ```bash
    # Example just showing the format
    $PYTHON Model-References/PyTorch/nlp/GPT2/partition_data.py /path/to/input/openwebtext/directory /path/to/output/openwebtext-train/directory partion-pecent
    ```
* `openwebtext-train` - which will contain 95% of the xz files.
    ```bash
    $ mkdir openwebtext-train
    $ $PYTHON Model-References/PyTorch/nlp/GPT2/partition_data.py /ath/to/GPT2-data/openwebtext /path/to/GPT2-data/openwebtext-train 0.95
    ```
* `openwebtext-valid` - which will contain 2.5% of the xz files.<br>
    ```bash
    $ mkdir openwebtext-train
    $ $PYTHON Model-References/PyTorch/nlp/GPT2/partition_data.py /path/to/GPT2-data/openwebtext /path/to/GPT2-data/openwebtext-valid 0.5
    ```
* `openwebtext-test`  - while will contain 2.5% of the xz files.
    ```bash
    $ mkdir openwebtext-train
    $ $PYTHON Model-References/PyTorch/nlp/GPT2/partition_data.py /path/to/GPT2-data/openwebtext /path/to/GPT2-data/openwebtext-test 1.0
    ```
>NOTE: The files in `openwebtext-train` directory shall not contain files from `openwebtext-valid` and `openwebtext-test` directories. Also, the above three partitions must be run in same sequence and partion-percent values respectively to obtain the correct data partition.

(4) In each of the `openwebtext-train`, `openwebtext-valid` and `openwebtext-test` directories, untar the xz files. To untar the xz files, run the below command in respective directory.

* In `openwebtext-train` :
    ```bash
    find -name \*.txt -exec sh -c 'cat {} >> train.raw' \;
    ```
* In `openwebtext-valid` :
    ```bash
    find -name \*.txt -exec sh -c 'cat {} >> valid.raw' \;
    ```
* In `openwebtext-test`:
    ```bash
    find -name \*.txt -exec sh -c 'cat {} >> test.raw' \;
    ```

(5) Download encoder and vocab files.
* Obtain the encoder.json file in GPT2-data directory
    ```bash
    wget -P GPT2-data/ https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    ```
* Obtain the vocab.bpe file in GPT2-data directory
    ```bash
    wget -P GPT2-data/ https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    ```
(6) Tokeninzing openwebtext
* Tokenizing openwebtext-train with gpt2 pretrained tokenizer
    ```bash
    $PYTHON Model-References/PyTorch/nlp/GPT2/examples/roberta/multiprocessing_bpe_encoder.py \
     --encoder-json GPT2-data/encoder.json \
     --vocab-bpe GPT2-data/vocab.bpe \
     --inputs GPT2-data/openwebtext-train/train.raw \
     --outputs GPT2-data/openwebtext-train/train.gpt2_bpe \
     --keep-empty \
     --workers 80
    ```
* Tokenizing openwebtext-valid with gpt2 pretrained tokenizer
    ```bash
    $PYTHON Model-References/PyTorch/nlp/GPT2/examples/roberta/multiprocessing_bpe_encoder.py \
     --encoder-json GPT2-data/encoder.json \
     --vocab-bpe GPT2-data/vocab.bpe \
     --inputs GPT2-data/openwebtext-test/test.raw \
     --outputs GPT2-data/openwebtext-test/test.gpt2_bpe \
     --keep-empty \
     --workers 80
    ```
* Tokenizing openwebtext-test with gpt2 pretrained tokenizer
    ```bash
    $PYTHON Model-References/PyTorch/nlp/GPT2/examples/roberta/multiprocessing_bpe_encoder.py \
    --encoder-json GPT2-data/encoder.json \
    --vocab-bpe GPT2-data/vocab.bpe \
    --inputs GPT2-data/openwebtext-valid/valid.raw \
    --outputs GPT2-data/openwebtext-valid/valid.gpt2_bpe \
    --keep-empty \
    --workers 80
    ```

(7) Binarizing the train, valid, test gpt2 tokenized files

* Create a destination directory to process and obtain the final openwebtext dataset.
    ```bash
    mkdir /data/OpenWebText_gpt2
    ```
* Binarize the tokenized file and store the files in above directory
    ```bash
    $PYTHON Model-References/PyTorch/nlp/GPT2/fairseq_cli/preprocess.py \
    --only-source \
    --trainpref GPT2-data/openwebtext-train/train.gpt2_bpe \
    --validpref GPT2-data/openwebtext-valid/valid.gpt2_bpe \
    --testpref GPT2-data/openwebtext-test/test.gpt2_bpe \
    --destdir data/OpenWebText_gpt2/ \
    --workers 80
    ```
    >NOTE: Since in above command there is no dict.txt file provided via --srcdict parameter, a new dict.txt file will be created, which is recommended.

# Train BERT with Data Packed Mode

## Data preparation
To pack data use the following command:
```bash
python pack_pretraining_data_tfrec.py \
--input-glob=<path to tfrecord directory> \
--output-dir=<path to packed tfrecord directory> \
--max-files 30
```
Output: (Old packed dataset) <br>
- Number of records (multi-samples): 4746826
- Average of packed samples per record: 2.0325640754474676
- Total number of samples in records: 9648228

####Data format before packing:
- **input_ids**: size=512, list of token ids, where:
    - 101 - start
    - 102 - end
    - 103 - mask  <br>
    padded to 512 with zeros.
- **input_mask**: size=512, 111...111000...000, where the number of ones corresponds to the effective sample length, padded to 512 with zeros.
- **segment_ids**: size=512, 000...000111...111000...000, where first zeros correspond to the first sentence, ones to second sentence, and  padded to 512 with zeros.
- **masked_lm_positions**: size=76, positions of masked tokens (103),  padded to 76 with zeros.
- **masked_lm_ids**: size=76, token ids of masked tokens,  padded to 76 with zeros.
- **masked_lm_weights**: size=76, 111...111000...000, number of ones equals to number of masked tokens.
- **next_sentence_labels**: size=1, 0 or 1, where 1 if sentence 2 is the next sentence of sentence 1.

####Data format after packing:
- **input_ids**: size=512, list of token ids where:
    - 101 - start
    - 102 - end
    - 103 - mask  <br>
    padded to 512 with zeros. <br>
    Example of 2 packed samples: 101,...,102,...,102,101,...,102,...,102,0...0 <br>
    where 101,...,102 first sentence, ,...,102 second sentence, <br>
    101,...,102 third sentence and ,...,102 forth sentence.
- **input_mask**: size=512, 111...111222...222000...000,
where the number of ones corresponds to the first sample length,
and the number of twos corresponds to the second sample length. (If there are 3 samples 1...12...23...30...0.)
- **segment_ids**: size=512, 000...000111...111000...000111...111000...000
where 000...000111...111 the first and the second samples, padded to 512 with zeros.
- **positions**: size=512, 0,1,2,3,...,\<length of first sample\> - 1,0,1,2,3,...,\<length of second sample\> - 1,0,0,...,0
- **masked_lm_positions**: size=79, positions of masked tokens (103), padded to 79 with zeros.
- **masked_lm_ids**: size=79, token ids of masked tokens, padded to 76 with zeros.
- **masked_lm_weights**: size=79, 111...111222...222000...000
where 111...111 corresponds to the first sample
and 222...222 to the second. (If there are 3 samples 1...12...23...30...0.)
- **next_sentence_positions**: size=3, \<position of first sample\>,\<position of second sample\>,0 (corresponds to 101 positions)
- **next_sentence_labels**: size=3, 0 or 1, where 1 if sentence 2 is the next sentence of sentence 1.
- **next_sentence_weights**: size=3, 110 (If there are 3 samples 111)

## Data packed mode

To run BERT pre-training on habana device use the following flag: <br>
`--enable_habana_backend` <br>

To run BERT pre-training with data packed mode use the following flags: <br>
- `--enable_packed_data_mode` - To enable packed data packed mode.
- `--avg_seq_per_pack=2` - Average number of packed samples per record, should be integer.
- `--compute_lm_loss_per_seq` - (Optional, can reduce accuracy) Compute lm loss per samples in packed record. If not used, lm loss is multiplied by avg_seq_per_pack.

Full command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONPATH=<path to model_garden root>/internal/MLPERF/Habana/benchmarks/bert/implementations/bert-tf-gaudi-8 \
mpiexec --allow-run-as-root -np 8 --bind-to socket \
python3 run_pretraining.py \
--input_files_dir=<path to training records dir> \
--init_checkpoint=<path to checkpoint dir>/model.ckpt-28252 \
--eval_files_dir=<path to eval records dir> \
--output_dir=<path to output dir> \
--bert_config_file=<path to bert config dir>/bert_config.json \
--do_train=True \
--do_eval=False \
--is_dist_eval_enabled=true \
--train_batch_size=14 \
--eval_batch_size=125 \
--max_eval_steps=80 \
--max_seq_length=512 \
--max_predictions_per_seq=76 \
--num_train_steps=6365 \
--num_accumulation_steps=4 \
--num_warmup_steps=0 \
--save_checkpoints_steps=335 \
--learning_rate=5.000000e-05 \
--horovod --noamp --nouse_xla \
--allreduce_post_accumulation=True \
--enable_device_warmup=True \
--samples_between_eval=150080 \
--stop_threshold=7.200000e-01 \
--samples_start_eval=0 \
--dllog_path=<path to log>/bert_dllog.json \
--enable_habana_backend \
--enable_packed_data_mode \
--avg_seq_per_pack=2 \
--compute_lm_loss_per_seq
```

####When packed data mode enabled the following parts of script are affected:
1) **input_fn_builder**
2) **model_fn_builder**
3) **embedding_postprocessor**
4) **create_attention_mask_from_input_mask**
5) **pooler inside BertMode.\__init\__**
6) **get_masked_lm_output**
7) **get_next_sentence_output**

####Jupyter notebook
For records investigation use following notebook: `notebooks/records investigation.ipynb`
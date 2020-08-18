# PyTorch Models for Gaudi

This page contains a general description on how to use and optmize models for PyTorch on Gaudi. All supported models are based on PyTorch v1.5.

Definitions: 
- Eager mode: refers to op by op execution as defined in standard PyTorch eager mode scripts
- Graph mode: refers to Torchscript based execution as defined in PyTorch 
- Lazy mode: refers to eager mode experience and scripts with a flag for performance on Gaudi

## ResNet50
Graph mode: Graph mode is supported in FP32 and BF16. Single node performance for BS 256, BF16 mixed precision is 679 images/sec. ResNet50 Top1 accuracy is measured as 75.85% (starting from 80th epoch, Train BS 256 & Test BS 32, BF16 Mixed precision).
Eager mode is supported for BS 256 with BF16 mixed precision and BS128 with FP32 data type.
   
## DLRM
Lazy mode: Supports Medium configuration with Adagrad optimizer for FP32 and BF16 mixed precision. Performance for medium configuration, measured as 36957 queries/sec, with random data, BS 512 and BF16 mixed precision configuration. DLRM Medium configuration accuracy and loss trend matches GPU.
Lazy mode refers to deferred execution of graphs, comprising of ops delivered from script op by op like eager mode. It gives eager mode experience with performance on Gaudi. 
Eager mode: Medium configuration for FP32 and BF16 is supported.
Medium configuration details are available in [DLRM README](./recommendation/dlrm/#Medium-Configuration).

# BERT Base  (Fine tuning)
Eager mode: BERT base is supported on both MRPC and SQuAD dataset using FP32 data type.
   
# BERT Large  (Fine tuning)
Graph mode: BERT Large supports SQuAD dataset with FP32 configurations for BS10, BS12 and SL 384. For BF16 mixed precision, supported configurations are BS16, BS24 and SL 384. The performance for BF16 mixed precision, BS24, SL384 is 30.2 sent/sec and loss trend matches GPU with accuracy of 87.5%. The performance for FP32, BS10, SL384 is 11.8 sent/sec and loss trend matches GPU with accuracy of 86.6%. MRPC dataset is supported for FP32, BS32 and SL 128.
Eager mode: BERT Large is supported on both MRPC and SQuAD dataset using FP32 data type.

# BERT Large Pre-training
Graph mode: BERT Large supports Wiki and BookWiki (combination of BooksCorpus and Wiki) datasets with FP32 and BF16 mixed precision. For FP32, per chip BS32 and SL128 for phase 1 and BS4, SL512 for phase 2 are supported. The performance is 43 sps for phase 1 and 8.5 sps for phase 2. For BF16 mixed precision, per chip BS64 and SL128 for phase 1 and BS8, SL512 for phase 2 are supported. The performance is 109.4 sent/sec for phase 1 and 20 sent/sec for phase 2. Convergence is verified by loss comparison with GPU and the loss trends match for all cases.

# Scale up (Pytorch + HCL)
- Scaling with BERT Large fine tuning on 8 Gaudis (single HLS) is at 86.4% for Graph mode with SQuAD dataset, for BF16 mixed precision. For FP32, the scaling is at 86.6% for Graph mode with SQuAD dataset. Per chip batch sizes are as mentioned for single card above.
- Scaling with BERT Large pre-training on 8 Gaudis (single HLS) is at 96.4% for phase1 and 85.6% for phase2 for Graph mode with Wikipedia bookcorpus dataset, for BF16 mixed precision. Per chip batch sizes are as mentioned for single card above. Corresponding scaling numbers for FP32 for configurations as mentioned in single node case above, are 94.5% for phase1 and 87% for phase2. 
- Scaling with ResNet50 scale-out training is measured as 88.5% on 8 Gaudis (single HLS) for Graph mode, BS 256 & Mixed precision.

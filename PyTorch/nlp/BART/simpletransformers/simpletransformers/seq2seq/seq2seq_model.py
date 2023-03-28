import json
import logging
import math
import os
import random
import warnings
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import time


import sys
import random
from sklearn.utils import shuffle
import torch
model_references_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../../../../.."))
os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ":" +  model_references_path
sys.path.append(model_references_path)
import hb_utils
import datetime

import transformers
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    #get_polynomial_decay_schedule_with_warmup,
)
from transformers.optimization import AdamW, Adafactor
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizerFast,
    BertConfig,
    BertModel,
    BertTokenizerFast,
    DistilBertConfig,
    DistilBertModel,
    DistilBertTokenizerFast,
    ElectraConfig,
    ElectraModel,
    ElectraTokenizerFast,
    EncoderDecoderConfig,
    EncoderDecoderModel,
    LongformerConfig,
    LongformerModel,
    LongformerTokenizerFast,
    MarianConfig,
    MarianMTModel,
    MarianTokenizer,
    MobileBertConfig,
    MobileBertModel,
    MobileBertTokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizerFast,
)
import datasets
from datasets import load_from_disk

from simpletransformers.config.global_args import global_args
from simpletransformers.config.model_args import Seq2SeqArgs
from simpletransformers.config.utils import sweep_config_to_sweep_values
from simpletransformers.seq2seq.seq2seq_utils import (
    Seq2SeqDataset,
    SimpleSummarizationDataset,
    add_faiss_index_to_dataset,
    generate_faiss_index_dataset,
    load_hf_dataset,
)




try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "auto": (AutoConfig, AutoModel, AutoTokenizer),
    "bart": (BartConfig, BartForConditionalGeneration, BartTokenizerFast),
    "bert": (BertConfig, BertModel, BertTokenizerFast),
    "distilbert": (DistilBertConfig, DistilBertModel, DistilBertTokenizerFast),
    "electra": (ElectraConfig, ElectraModel, ElectraTokenizerFast),
    "longformer": (LongformerConfig, LongformerModel, LongformerTokenizerFast),
    "mobilebert": (MobileBertConfig, MobileBertModel, MobileBertTokenizerFast),
    "marian": (MarianConfig, MarianMTModel, MarianTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizerFast),
}


class Seq2SeqModel:
    def __init__(
        self,
        encoder_type=None,
        encoder_name=None,
        decoder_name=None,
        encoder_decoder_type=None,
        encoder_decoder_name=None,
        additional_special_tokens_encoder=None,
        additional_special_tokens_decoder=None,
        index_name=None,
        knowledge_dataset=None,
        index_path=None,
        dpr_ctx_encoder_model_name=None,
        config=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        **kwargs,
    ):

        """
        Initializes a Seq2SeqModel.

        Args:
            encoder_type (optional): The type of model to use as the encoder.
            encoder_name (optional): The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            decoder_name (optional): The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
                                    Must be the same "size" as the encoder model (base/base, large/large, etc.)
            encoder_decoder_type (optional): The type of encoder-decoder model. (E.g. bart)
            encoder_decoder_name (optional): The path to a directory containing the saved encoder and decoder of a Seq2SeqModel. (E.g. "outputs/") OR a valid BART or MarianMT model.
            additional_special_tokens_encoder (optional): dict of special tokens to add to encoder tokenizer
            additional_special_tokens_decoder (optional): dict of special tokens to add to decoder tokenizer
            index_name (optional): Name of the index to use: 'hf' for a canonical dataset from the datasets library, 'custom' for a local index, or 'legacy' for the original one
            knowledge_dataset (optional): Path to a TSV file (two columns - title, text) containing a knowledge dataset for RAG or the path to a directory containing a saved Huggingface dataset for RAG.
                                        If this is not given for a RAG model, a dummy dataset will be used.
            index_path (optional): Path to the faiss index of the custom knowledge dataset. If this is not given and knowledge_dataset is given, it will be computed.
            dpr_ctx_encoder_model_name (optional): The DPR context encoder model to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
                                                This is required when using a custom knowledge_dataset.
            config (optional): A configuration file to build an EncoderDecoderModel.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        if not config:
            # if not ((encoder_name and decoder_name) or encoder_decoder_name) and not encoder_type:
            if not ((encoder_name and decoder_name) or encoder_decoder_name):
                raise ValueError(
                    "You must specify a Seq2Seq config \t OR \t"
                    "encoder_type, encoder_name, and decoder_name OR \t \t"
                    "encoder_type and encoder_decoder_name"
                )
            elif not (encoder_type or encoder_decoder_type):
                raise ValueError(
                    "You must specify a Seq2Seq config \t OR \t"
                    "encoder_type, encoder_name, and decoder_name \t OR \t"
                    "encoder_type and encoder_decoder_name"
                )

        self.args = self._load_model_args(encoder_decoder_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, Seq2SeqArgs):
            self.args = args

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if self.args.local_rank in [-1, 0] else logging.WARN,
        )

        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

        self.set_seed()

        self.device = args.device
        self.results = {}

        if not use_cuda:
            self.args.fp16 = False

        if self.args.local_rank not in [-1, 0]:
            if args.use_habana:
                hb_utils.barrier()
            else:
                torch.distributed.barrier()

        if encoder_decoder_type in ["rag-token", "rag-sequence"]:
            config_class, model_class, tokenizer_class, retriever_class = MODEL_CLASSES[
                encoder_decoder_type
            ]

            if self.args.config is not None:
                config = config_class.from_pretrained(
                    encoder_decoder_name, **self.args.config
                )
            elif config is not None:
                config = config_class.from_pretrained(encoder_decoder_name, **config)

            self.encoder_tokenizer = tokenizer_class.from_pretrained(
                encoder_decoder_name, config=config
            )
            self.decoder_tokenizer = self.encoder_tokenizer
            if knowledge_dataset is None:
                if index_name is None:
                    self.retriever = retriever_class.from_pretrained(
                        encoder_decoder_name,
                        index_name="exact",
                        use_dummy_dataset=True,
                        config=config,
                    )
                elif index_name == "legacy":
                    self.retriever = retriever_class.from_pretrained(
                        encoder_decoder_name,
                        index_name="legacy",
                        use_dummy_dataset=False,
                        config=config,
                    )
            else:
                if os.path.isdir(knowledge_dataset):
                    self.dataset = load_from_disk(knowledge_dataset)
                    if index_path:
                        self.dataset.load_faiss_index("embeddings", index_path)
                    elif os.path.isfile(
                        os.path.join(knowledge_dataset, "hf_dataset_index.faiss")
                    ):
                        self.dataset.load_faiss_index(
                            "embeddings",
                            os.path.join(knowledge_dataset, "hf_dataset_index.faiss"),
                        )
                    else:
                        self.dataset = add_faiss_index_to_dataset(self.dataset)
                        if self.args.save_knowledge_dataset:
                            self.dataset.get_index("embeddings").save(
                                os.path.join(
                                    knowledge_dataset, "hf_dataset_index.faiss"
                                )
                            )
                else:
                    self.dataset = generate_faiss_index_dataset(
                        knowledge_dataset,
                        ctx_encoder_name=dpr_ctx_encoder_model_name,
                        args=self.args,
                        device=self.device,
                    )
                    if self.args.save_knowledge_dataset:
                        self.dataset.get_index("embeddings").save(
                            os.path.join(
                                self.args.output_dir,
                                "knowledge_dataset",
                                "hf_dataset_index.faiss",
                            )
                        )
                self.retriever = RagRetriever.from_pretrained(
                    encoder_decoder_name,
                    index_name="custom",
                    indexed_dataset=self.dataset,
                    config=config,
                )

            self.model = model_class.from_pretrained(
                encoder_decoder_name, retriever=self.retriever, config=config
            )
            self.config = self.model.config
        else:
            if encoder_decoder_type:
                config_class, model_class, tokenizer_class = MODEL_CLASSES[
                    encoder_decoder_type
                ]
            else:
                config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_type]

            if encoder_decoder_type in ["bart", "mbart", "marian"]:
                self.model = model_class.from_pretrained(encoder_decoder_name, cache_dir=self.args.cache_dir)
                if encoder_decoder_type in ["bart", "mbart"]:
                    self.encoder_tokenizer = tokenizer_class.from_pretrained(
                        encoder_decoder_name, cache_dir=self.args.cache_dir
                    )
                elif encoder_decoder_type == "marian":
                    if self.args.base_marian_model_name:
                        self.encoder_tokenizer = tokenizer_class.from_pretrained(
                            self.args.base_marian_model_name
                        )
                    else:
                        self.encoder_tokenizer = tokenizer_class.from_pretrained(
                            encoder_decoder_name
                        )
                self.decoder_tokenizer = self.encoder_tokenizer
                self.config = self.model.config
            else:
                if encoder_decoder_name:
                    # self.model = EncoderDecoderModel.from_pretrained(encoder_decoder_name)
                    self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                        os.path.join(encoder_decoder_name, "encoder"),
                        os.path.join(encoder_decoder_name, "decoder"),
                    )
                    self.encoder_tokenizer = tokenizer_class.from_pretrained(
                        os.path.join(encoder_decoder_name, "encoder")
                    )
                    self.decoder_tokenizer = AutoTokenizer.from_pretrained(
                        os.path.join(encoder_decoder_name, "decoder")
                    )
                else:
                    self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                        encoder_name, decoder_name, config=config
                    )
                    self.encoder_tokenizer = tokenizer_class.from_pretrained(
                        encoder_name
                    )
                    self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name)
                self.encoder_config = self.model.config.encoder
                self.decoder_config = self.model.config.decoder

        if self.args.local_rank == 0:
            if args.use_habana:
                hb_utils.barrier()
            else:
                torch.distributed.barrier()

        if additional_special_tokens_encoder is not None:
            self.encoder_tokenizer.add_special_tokens(additional_special_tokens_encoder)

        if additional_special_tokens_decoder is not None:
            self.decoder_tokenizer.add_special_tokens(additional_special_tokens_decoder)

        if self.args.wandb_project and not wandb_available:
            warnings.warn(
                "wandb_project specified but wandb is not available. Wandb disabled."
            )
            self.args.wandb_project = None

        # `model_name` could be provided in args
        if self.args.model_name is None:
            if encoder_decoder_name:
                self.args.model_name = encoder_decoder_name

                # # Checking if we are loading from a saved model or using a pre-trained model
                # if not saved_model_args and encoder_decoder_type == "marian":
                # Need to store base pre-trained model name to get the tokenizer when loading a saved model
                self.args.base_marian_model_name = encoder_decoder_name

            elif encoder_name and decoder_name:
                self.args.model_name = encoder_name + "-" + decoder_name
            else:
                self.args.model_name = "encoder-decoder"

            if encoder_decoder_type:
                self.args.model_type = encoder_decoder_type
            elif encoder_type:
                self.args.model_type = encoder_type + "-bert"
            else:
                self.args.model_type = "encoder-decoder"

    def train_model(
        self,
        train_data,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_data=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model using 'train_data'

        Args:
            train_data: Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.
                        - `input_text`: The input text sequence.
                        - `target_text`: The target text sequence
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_data (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down training significantly as the predicted sequences need to be generated.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"


        if args:
            self.args.update_from_dict(args)

        # if self.args.silent:
        #     show_running_loss = False

        if self.args.evaluate_during_training and eval_data is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_data is not specified."
                " Pass eval_data to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        if self.args.local_rank in [-1, 0]:
            if (
                os.path.exists(output_dir)
                and os.listdir(output_dir)
                and not self.args.overwrite_output_dir
            ):
                raise ValueError(
                    "Output directory ({}) already exists and is not empty."
                    " Set args.overwrite_output_dir = True to overcome.".format(output_dir)
                )

        trainMetaData = None

        self._move_model_to_device()

        train_dataset = self.load_and_cache_examples(train_data, verbose=verbose)


        os.makedirs(output_dir, exist_ok=True)

        print("Start training")
        start_time = time.time()

        global_step, training_details = self.train(
            train_dataset,
            output_dir,
            show_running_loss=show_running_loss,
            eval_data=eval_data,
            verbose=verbose,
            trainMetaData=trainMetaData,
            **kwargs,
        )
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        if self.args.local_rank in [-1, 0]:
            self.save_model(self.args.output_dir, model=self.model)

        # model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        # model_to_save.save_pretrained(output_dir)
        # self.encoder_tokenizer.save_pretrained(output_dir)
        # self.decoder_tokenizer.save_pretrained(output_dir)
        # torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        if verbose and self.args.is_master:
            logger.info(
                " Training of {} model complete. Saved to {}.".format(
                    self.args.model_name, output_dir
                )
            )

        return global_step, training_details

    def set_seed(self):
        args = self.args
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    def train(
        self,
        train_dataset,
        output_dir,
        show_running_loss=True,
        eval_data=None,
        verbose=True,
        trainMetaData=None,
        **kwargs,
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args

        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter(logdir=args.tensorboard_dir)
        print("####################### gradient_accumulation_steps ########################", self.args.gradient_accumulation_steps)
        ################# distributed training ##################
        if args.distributed:
            if args.use_habana:
                train_sampler = DistributedSampler(train_dataset)
                train_dataloader = DataLoader(
                    train_dataset,
                    sampler=train_sampler,
                    batch_size=args.train_batch_size,
                    num_workers=self.args.dataloader_num_workers, pin_memory=True, drop_last=True
                )
            else:
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=args.world_size,
                    rank=args.local_rank,
                )
                train_dataloader = DataLoader(
                    train_dataset,
                    sampler=train_sampler,
                    batch_size=args.train_batch_size,
                )
            print("####################### distributed training with ", self.device, self.args.dataloader_num_workers)
        else:
            train_sampler = RandomSampler(train_dataset)
            data_loader_type = DataLoader
            train_dataloader = data_loader_type(
                train_dataset,
                sampler=train_sampler,
                batch_size=args.train_batch_size,
                num_workers=self.args.dataloader_num_workers,
            )

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = (
                args.max_steps
                // (len(train_dataloader) // args.gradient_accumulation_steps)
                + 1
            )
        else:
            t_total = (
                len(train_dataloader)
                // args.gradient_accumulation_steps
                * args.num_train_epochs
            )
        if args.debug:
            print('args.max_steps acc t_total', args.max_steps, args.gradient_accumulation_steps, t_total, args.output_dir)
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [
                p for n, p in model.named_parameters() if n in params
            ]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                            and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                            and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = (
            warmup_steps if args.warmup_steps == 0 else args.warmup_steps
        )

        # TODO: Use custom optimizer like with BertSum?
        if args.optimizer == "FusedAdamW":
            try:
                #from hb_custom import FusedAdamW
                from habana_frameworks.torch.hpex.optimizers import FusedAdamW
            except ImportError:
                raise ImportError("Please install habana_torch.")
            optimizer = FusedAdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adam_epsilon,
            )
        elif args.optimizer == "AdamW":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adam_epsilon,
            )
        elif args.optimizer == "Adafactor":
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adafactor_eps,
                clip_threshold=args.adafactor_clip_threshold,
                decay_rate=args.adafactor_decay_rate,
                beta1=args.adafactor_beta1,
                weight_decay=args.weight_decay,
                scale_parameter=args.adafactor_scale_parameter,
                relative_step=args.adafactor_relative_step,
                warmup_init=args.adafactor_warmup_init,
            )
            print("Using Adafactor for T5")
        else:
            raise ValueError(
                "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                    args.optimizer
                )
            )
        if args.debug:
            print('args.scheduler', args.scheduler)
        if args.scheduler == "constant_schedule":
            scheduler = get_constant_schedule(optimizer)

        elif args.scheduler == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps
            )

        elif args.scheduler == "linear_schedule_with_warmup":
            print('args.warmup_steps t_total', args.warmup_steps, t_total)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
            )

        elif args.scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "polynomial_decay_schedule_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                lr_end=args.polynomial_decay_schedule_lr_end,
                power=args.polynomial_decay_schedule_power,
            )

        else:
            raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

        if (
            args.model_name
            and os.path.isfile(os.path.join(args.model_name, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(args.model_name, "optimizer.pt"))
            )
            scheduler.load_state_dict(
                torch.load(os.path.join(args.model_name, "scheduler.pt"))
            )

        if args.n_gpu > 1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
        if args.distributed and args.use_habana:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, bucket_cap_mb=200, broadcast_buffers=False,
                gradient_as_bucket_view=True, find_unused_parameters=True)

        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            int(args.num_train_epochs), disable=True, mininterval=0
        )
        self.set_seed()
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0
        loss_list = []

        if args.model_name and os.path.exists(args.model_name):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )
                if args.local_rank in [-1, 0]:
                    logger.info(
                        "   Continuing training from checkpoint, will skip to saved global_step"
                    )
                    logger.info("   Continuing training from epoch %d", epochs_trained)
                    logger.info("   Continuing training from global step %d", global_step)
                    logger.info(
                        "   Will skip the first %d steps in the current epoch",
                        steps_trained_in_current_epoch,
                    )
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            kwmetrics = {}
            if args.evaluate_generated_text:
                kwmetrics = {"f1":[], "precision":[], "recall":[]}
            training_progress_scores = self._create_training_progress_scores(**kwmetrics)

        if args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                config={**asdict(args)},
                **args.wandb_kwargs,
            )
            wandb.run._label(repo="simpletransformers")
            wandb.watch(self.model)

        if args.fp16:
            from torch.cuda import amp
            scaler = amp.GradScaler()
        
        if args.bf16 == "hmp":
            from habana_frameworks.torch.hpex import hmp
            hmp.convert(opt_level=args.hmp_opt_level, bf16_file_path=args.hmp_bf16,
                        fp32_file_path=args.hmp_fp32, isVerbose=args.hmp_verbose)

        model.train()
        for current_epoch in train_iterator:
            if args.distributed:
                train_sampler.set_epoch(current_epoch)

            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(
                f"Epoch {epoch_number + 1} of {args.num_train_epochs}"
            )

            batch_iterator = tqdm(
                train_dataloader,
                disable=not args.is_master,
                mininterval=0,
            )
            model.train()
            for step, batch in enumerate(batch_iterator):
                model.train()
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if args.use_habana:
                    for key in batch.keys():
                        batch[key] = batch[key].to(dtype=torch.int32)
                #batch = tuple(t.to(args.device) for t in batch)

                if args.use_habana and args.use_fused_clip_norm:
                    try:
                        from habana_frameworks.torch.hpex.normalization import FusedClipNorm
                    except ImportError:
                        raise ImportError("Please install habana_torch.")
                    FusedNorm = FusedClipNorm(model.parameters(), args.max_grad_norm)

                inputs = self._get_inputs_dict(batch)


                if args.distributed:
                    if args.use_habana:
                        hb_utils.barrier()
                    else:
                        torch.distributed.barrier()
                #optimizer.zero_grad()

                if args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        # model outputs are always tuple in pytorch-transformers (see doc)
                        loss = outputs[0]
                else:
                    with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=(args.bf16 == "autocast")):
                        outputs = model(**inputs)
                        loss = outputs[0]

                if args.n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training

                if args.debug:
                    print('######################### loss', loss.device, loss.dtype, loss.shape)

                if show_running_loss and (args.logging_steps > 0 and global_step % args.logging_steps) == 0:
                    current_loss = loss.item()
                    batch_iterator.set_description(
                        f"Epochs {epoch_number+1}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    scaler.scale(loss).backward()
                elif args.bf16 == "hmp":
                    from habana_frameworks.torch.hpex import hmp
                    with hmp.disable_casts():
                            optimizer.step()
                else:
                    loss.backward()

                if args.lazy_mode:
                    import habana_frameworks.torch.core as htcore
                    htcore.mark_step()

                loss_list.append(loss)
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)

                    if args.use_habana:
                        if args.use_fused_clip_norm:
                            FusedNorm.clip_norm(model.parameters())
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), args.max_grad_norm
                            )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )


                    if args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    model.zero_grad(True)

                    scheduler.step()  # Update learning rate schedule
                    global_step += 1

                    if args.lazy_mode:
                        import habana_frameworks.torch.core as htcore
                        htcore.mark_step()

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        for loss_t in loss_list :
                            tr_loss += loss_t.item()
                        # Log metrics
                        tb_writer.add_scalar(
                            "lr", scheduler.get_last_lr()[0], global_step
                        )
                        tb_writer.add_scalar(
                            "loss",
                            (tr_loss - logging_loss) / args.logging_steps,
                            global_step,
                        )
                        logging_loss = tr_loss
                        if args.wandb_project or self.is_sweeping:
                            wandb.log(
                                {
                                    "Training loss": loss.item(),
                                    "lr": scheduler.get_last_lr()[0],
                                    "global_step": global_step,
                                }
                            )
                        loss_list.clear()

                    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(global_step)
                        )

                        self.save_model(
                            output_dir_current, optimizer, scheduler, model=model
                        )

                    if args.local_rank in [-1, 0] and args.evaluate_during_training and (
                        args.evaluate_during_training_steps > 0
                        and global_step % args.evaluate_during_training_steps == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = self.eval_model(
                            eval_data,
                            verbose=verbose and args.evaluate_during_training_verbose,
                            silent=args.evaluate_during_training_silent,
                            **kwargs,
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step
                            )

                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(global_step)
                        )

                        if args.save_eval_checkpoints:
                            self.save_model(
                                output_dir_current,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(loss.item())
                        for key in results:
                            training_progress_scores[key].append(results[key])
                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(
                                args.output_dir, "training_progress_scores.csv"
                            ),
                            index=False,
                        )

                        if args.wandb_project or self.is_sweeping:
                            wandb.log(self._get_last_metrics(training_progress_scores))

                        if not best_eval_metric:
                            best_eval_metric = results[args.early_stopping_metric]
                            if args.save_best_model:
                                self.save_model(
                                    args.best_model_dir,
                                    optimizer,
                                    scheduler,
                                    model=model,
                                    results=results,
                                )
                        if best_eval_metric and args.early_stopping_metric_minimize:
                            if (
                                results[args.early_stopping_metric] - best_eval_metric
                                < args.early_stopping_delta
                            ):
                                best_eval_metric = results[args.early_stopping_metric]
                                if args.save_best_model:
                                    self.save_model(
                                        args.best_model_dir,
                                        optimizer,
                                        scheduler,
                                        model=model,
                                        results=results,
                                    )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if (
                                        early_stopping_counter
                                        < args.early_stopping_patience
                                    ):
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(
                                                f" No improvement in {args.early_stopping_metric}"
                                            )
                                            logger.info(
                                                f" Current step: {early_stopping_counter}"
                                            )
                                            logger.info(
                                                f" Early stopping patience: {args.early_stopping_patience}"
                                            )
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {args.early_stopping_patience} steps reached"
                                            )
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        else:
                            if (
                                results[args.early_stopping_metric] - best_eval_metric
                                > args.early_stopping_delta
                            ):
                                best_eval_metric = results[args.early_stopping_metric]
                                if args.save_best_model:
                                    self.save_model(
                                        args.best_model_dir,
                                        optimizer,
                                        scheduler,
                                        model=model,
                                        results=results,
                                    )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if (
                                        early_stopping_counter
                                        < args.early_stopping_patience
                                    ):
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(
                                                f" No improvement in {args.early_stopping_metric}"
                                            )
                                            logger.info(
                                                f" Current step: {early_stopping_counter}"
                                            )
                                            logger.info(
                                                f" Early stopping patience: {args.early_stopping_patience}"
                                            )
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {args.early_stopping_patience} steps reached"
                                            )
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        model.train()

                if args.max_steps > 0 and global_step > args.max_steps:
                    batch_iterator.close()
                    break

            epoch_number += 1
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

            output_dir_current = os.path.join(
                output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number)
            )

            if args.local_rank in [-1, 0] and args.save_model_every_epoch or args.evaluate_during_training:
                os.makedirs(output_dir_current, exist_ok=True)

            if args.local_rank in [-1, 0] and args.save_model_every_epoch:
                self.save_model(output_dir_current, optimizer, scheduler, model=model)

            if args.local_rank in [-1, 0] and args.evaluate_during_training and args.evaluate_each_epoch:
                results = self.eval_model(
                    eval_data,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    silent=args.evaluate_during_training_silent,
                    **kwargs,
                )

                if args.save_eval_checkpoints:
                    self.save_model(
                        output_dir_current, optimizer, scheduler, results=results, model=model
                    )

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                for key in training_progress_scores:
                    print('this is training_progress_scores: ', key, training_progress_scores[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(
                    os.path.join(args.output_dir, "training_progress_scores.csv"),
                    index=False,
                )

                if args.wandb_project or self.is_sweeping:
                    wandb.log(self._get_last_metrics(training_progress_scores))

                if not best_eval_metric:
                    best_eval_metric = results[args.early_stopping_metric]
                    if args.save_best_model:
                        self.save_model(
                            args.best_model_dir,
                            optimizer,
                            scheduler,
                            model=model,
                            results=results,
                        )
                if best_eval_metric and args.early_stopping_metric_minimize:
                    if (
                        results[args.early_stopping_metric] - best_eval_metric
                        < args.early_stopping_delta
                    ):
                        best_eval_metric = results[args.early_stopping_metric]
                        if args.save_best_model:
                            self.save_model(
                                args.best_model_dir,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )
                        early_stopping_counter = 0
                    else:
                        if (
                            args.use_early_stopping
                            and args.early_stopping_consider_epochs
                        ):
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(
                                        f" No improvement in {args.early_stopping_metric}"
                                    )
                                    logger.info(
                                        f" Current step: {early_stopping_counter}"
                                    )
                                    logger.info(
                                        f" Early stopping patience: {args.early_stopping_patience}"
                                    )
                            else:
                                if verbose:
                                    logger.info(
                                        f" Patience of {args.early_stopping_patience} steps reached"
                                    )
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )
                else:
                    if (
                        results[args.early_stopping_metric] - best_eval_metric
                        > args.early_stopping_delta
                    ):
                        best_eval_metric = results[args.early_stopping_metric]
                        if args.save_best_model:
                            self.save_model(
                                args.best_model_dir,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )
                        early_stopping_counter = 0
                    else:
                        if (
                            args.use_early_stopping
                            and args.early_stopping_consider_epochs
                        ):
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(
                                        f" No improvement in {args.early_stopping_metric}"
                                    )
                                    logger.info(
                                        f" Current step: {early_stopping_counter}"
                                    )
                                    logger.info(
                                        f" Early stopping patience: {args.early_stopping_patience}"
                                    )
                            else:
                                if verbose:
                                    logger.info(
                                        f" Patience of {args.early_stopping_patience} steps reached"
                                    )
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )

        results = self.eval_model(
                    eval_data,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    silent=args.evaluate_during_training_silent,
                    **kwargs,
            )

        return (
            global_step,
            tr_loss / global_step
            if not self.args.evaluate_during_training
            else training_progress_scores,
        )

    def eval_model(
        self, eval_data, output_dir=None, verbose=True, silent=False, **kwargs
    ):
        """
        Evaluates the model on eval_data. Saves results to output_dir.

        Args:
            eval_data: Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.
                        - `input_text`: The input text sequence.
                        - `target_text`: The target text sequence.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            silent: If silent, tqdm progress bars will be hidden.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down evaluation significantly as the predicted sequences need to be generated.
        Returns:
            results: Dictionary containing evaluation results.
        """  # noqa: ignore flake8"

        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()

        eval_dataset = self.load_and_cache_examples(
            eval_data, evaluate=True, verbose=verbose, silent=silent
        )
        os.makedirs(output_dir, exist_ok=True)

        result = self.evaluate(
            eval_dataset, output_dir, verbose=verbose, silent=silent, **kwargs
        )
        self.results.update(result)

        if self.args.evaluate_generated_text:
            to_predict = eval_data["input_text"].tolist()
            if self.args.model_type in ["rag-token", "rag-sequence"]:
                preds, _ = self.predict(to_predict)
            else:
                preds = self.predict(to_predict)

            result = self.compute_metrics(
                eval_data["target_text"].tolist(), preds, **kwargs
            )
            self.results.update(result)

        return self.results

    def evaluate(self, eval_dataset, output_dir, verbose=True, silent=False, **kwargs):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args
        eval_output_dir = output_dir

        results = {}
        if args.distributed:
            if args.use_habana:
                eval_sampler = DistributedSampler(eval_dataset)
                eval_dataloader = DataLoader(
                    eval_dataset,
                    sampler=eval_sampler,
                    batch_size=args.eval_batch_size,
                    num_workers=self.args.dataloader_num_workers, pin_memory=True, drop_last=True
                )
            else:
                eval_sampler = DistributedSampler(
                    eval_dataset,
                    num_replicas=args.world_size,
                    rank=args.local_rank,
                )
                eval_dataloader = DataLoader(
                    eval_dataset,
                    sampler=eval_sampler,
                    batch_size=args.eval_batch_size,
                )
            print("####################### distributed eval with ", self.device, self.args.dataloader_num_workers)
        else:
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(
                eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
            )

        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()

        if args.n_gpu > 1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.local_rank], output_device=self.args.local_rank)

        if self.args.fp16:
            from torch.cuda import amp
        if self.args.bf16 == "hmp":
            from habana_frameworks.torch.hpex import hmp
            hmp.convert(opt_level=self.args.hmp_opt_level, bf16_file_path=self.args.hmp_bf16,
                        fp32_file_path=self.args.hmp_fp32, isVerbose=self.args.hmp_verbose)

        print('################ evaluate ###############')
        start_time = time.time()
        for batch in tqdm(
            eval_dataloader, disable=args.local_rank not in [-1, 0], desc="Running Evaluation"
        ):
            # batch = tuple(t.to(device) for t in batch)

            inputs = self._get_inputs_dict(batch)
            with torch.no_grad(), torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=(args.bf16 == "autocast")):
                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        tmp_eval_loss = outputs[0]
                else:
                    outputs = model(**inputs)
                    tmp_eval_loss = outputs[0]
                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps

        results["eval_loss"] = eval_loss

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Evaluation time {}'.format(total_time_str))
        print('eval_loss:', eval_loss)

        if args.local_rank in [-1, 0]:
            output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

        return results

    def predict(self, to_predict):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction. Note that the prefix should be prepended to the text.

        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8"

        print("################# predict, model is on: #################", next(self.model.parameters()).device)

        all_outputs = []
        all_retrieved = []
        all_doc_scores = []
        # Batching
        for idx_pred, batch in enumerate(tqdm(
            [
                to_predict[i : i + self.args.eval_batch_size]
                for i in range(0, len(to_predict), self.args.eval_batch_size)
            ],
            desc="Generating outputs",
            disable=self.args.local_rank not in [-1, 0],
        )):
            if self.args.model_type == "marian":
                input_ids = self.encoder_tokenizer.prepare_seq2seq_batch(
                    batch,
                    max_length=self.args.max_seq_length,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                )["input_ids"]
            elif self.args.model_type in ["mbart"]:
                input_ids = self.encoder_tokenizer.prepare_seq2seq_batch(
                    src_texts=batch,
                    max_length=self.args.max_seq_length,
                    pad_to_max_length=True,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                    src_lang=self.args.src_lang,
                )["input_ids"]
            elif self.args.model_type in ["rag-token", "rag-sequence"]:
                input_ids = self.encoder_tokenizer(
                    batch, truncation=True, padding="longest", return_tensors="pt"
                )["input_ids"].to(self.device)

                question_hidden_states = self.model.question_encoder(input_ids)[0]

                docs_dict = self.retriever(
                    input_ids.cpu().numpy(),
                    question_hidden_states.detach().cpu().numpy(),
                    return_tensors="pt",
                )
                doc_scores = torch.bmm(
                    question_hidden_states.unsqueeze(1),
                    docs_dict["retrieved_doc_embeds"]
                    .float()
                    .transpose(1, 2)
                    .to(self.device),
                ).squeeze(1)
            else:
                input_ids = self.encoder_tokenizer.batch_encode_plus(
                    batch,
                    max_length=self.args.max_seq_length,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                )["input_ids"]
            input_ids = input_ids.to(self.device)
            if self.args.debug:
                print("#####################  in predict, batch idx_pred, before generate(), inpu_ids is on: #########################", idx_pred, input_ids.device)
            if self.args.model_type in ["bart", "marian"]:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_length,
                    length_penalty=self.args.length_penalty,
                    early_stopping=self.args.early_stopping,
                    repetition_penalty=self.args.repetition_penalty,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    num_return_sequences=self.args.num_return_sequences,
                )
            elif self.args.model_type in ["mbart"]:
                tgt_lang_token = self.decoder_tokenizer._convert_token_to_id(
                    self.args.tgt_lang
                )

                outputs = self.model.generate(
                    input_ids=input_ids,
                    decoder_start_token_id=tgt_lang_token,
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_length,
                    length_penalty=self.args.length_penalty,
                    early_stopping=self.args.early_stopping,
                    repetition_penalty=self.args.repetition_penalty,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    num_return_sequences=self.args.num_return_sequences,
                )
            elif self.args.model_type in ["rag-token", "rag-sequence"]:
                outputs = self.model.generate(
                    context_input_ids=docs_dict["context_input_ids"].to(self.device),
                    context_attention_mask=docs_dict["context_attention_mask"].to(
                        self.device
                    ),
                    doc_scores=doc_scores,
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_length,
                    length_penalty=self.args.length_penalty,
                    early_stopping=self.args.early_stopping,
                    repetition_penalty=self.args.repetition_penalty,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    num_return_sequences=self.args.num_return_sequences,
                )
                retrieved_docs = [
                    doc
                    for doc in self.retriever.index.get_doc_dicts(docs_dict["doc_ids"])
                ]
            else:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    decoder_start_token_id=self.model.config.decoder.pad_token_id,
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_length,
                    length_penalty=self.args.length_penalty,
                    early_stopping=self.args.early_stopping,
                    repetition_penalty=self.args.repetition_penalty,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    num_return_sequences=self.args.num_return_sequences,
                )
            if self.args.debug:
                print("################################## predict: batch idx_pred, after generate, done ##############", idx_pred)
            all_outputs.extend(outputs.cpu().numpy())
            if self.args.model_type in ["rag-token", "rag-sequence"]:
                all_retrieved.extend(retrieved_docs)
                all_doc_scores.extend(doc_scores.detach().cpu())
        if self.args.debug:
            print("################################## predict: after all batches, before decode ##############")
        if self.args.model_type in ["rag-token", "rag-sequence"]:
            outputs = self.encoder_tokenizer.batch_decode(
                all_outputs,
                skip_special_tokens=self.args.skip_special_tokens,
                clean_up_tokenization_spaces=True,
            )
        elif self.args.use_multiprocessed_decoding:
            if self.args.multiprocessing_chunksize == -1:
                chunksize = max(len(all_outputs) // (self.args.process_count * 2), 500)
            else:
                chunksize = self.args.multiprocessing_chunksize

            self.model.to("cpu")
            with Pool(self.args.process_count) as p:
                outputs = list(
                    tqdm(
                        p.imap(self._decode, all_outputs, chunksize=chunksize),
                        total=len(all_outputs),
                        desc="Decoding outputs",
                        disable=self.args.local_rank not in [-1, 0],
                    )
                )
            self._move_model_to_device()
        else:
            outputs = [
                self.decoder_tokenizer.decode(
                    output_id,
                    skip_special_tokens=self.args.skip_special_tokens,
                    clean_up_tokenization_spaces=True,
                )
                for output_id in all_outputs
            ]
        if self.args.debug:
            print("################################## predict: after all batches, after decode ##############")
        if self.args.num_return_sequences > 1:
            if self.args.model_type in ["rag-token", "rag-sequence"]:
                return (
                    [
                        outputs[i : i + self.args.num_return_sequences]
                        for i in range(0, len(outputs), self.args.num_return_sequences)
                    ],
                    [
                        all_retrieved[i : i + self.args.num_return_sequences]
                        for i in range(0, len(outputs), self.args.num_return_sequences)
                    ],
                    [
                        all_doc_scores[i : i + self.args.num_return_sequences]
                        for i in range(0, len(outputs), self.args.num_return_sequences)
                    ],
                )
            else:
                self._move_model_to_device()
                if self.args.debug:
                    print("######################### end of predict, model is on:", next(self.model.parameters()).device)
                return [
                    outputs[i : i + self.args.num_return_sequences]
                    for i in range(0, len(outputs), self.args.num_return_sequences)
                ]
        else:
            if self.args.model_type in ["rag-token", "rag-sequence"]:
                return outputs, all_retrieved, all_doc_scores
            else:
                return outputs

    def _decode(self, output_id):
        return self.decoder_tokenizer.decode(
            output_id,
            skip_special_tokens=self.args.skip_special_tokens,
            clean_up_tokenization_spaces=True,
        )

    def compute_metrics(self, labels, preds, **kwargs):
        """
        Computes the evaluation metrics for the model predictions.

        Args:
            labels: List of target sequences
            preds: List of model generated outputs
            **kwargs: Custom metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down evaluation significantly as the predicted sequences need to be generated.

        Returns:
            result: Dictionary containing evaluation results.
        """  # noqa: ignore flake8"
        # assert len(labels) == len(preds)
        if self.args.debug:
            print('################# labels #################')
            print(labels)

            print('################## preds #################')
            print(preds)

        results = {}
        from datasets import load_dataset, load_metric
        metric = load_metric("bertscore", keep_in_memory=True)
        for r_idx, ref_lab in enumerate(labels):
            metric.add_batch(predictions=[preds[r_idx]], references=[ref_lab])
        eval_metric = metric.compute(lang="en")
        if self.args.debug:
            print('############## eval_metric keys: ####################', eval_metric.keys())
        results["f1"] = np.average([v for v in eval_metric["f1"]])
        results["precision"] = np.average([v for v in eval_metric["precision"]])
        results["recall"] = np.average([v for v in eval_metric["recall"]])

        #for metric, func in kwargs.items():
        #    results[metric] = func(labels, preds)

        return results

    def load_and_cache_examples(
        self, data, evaluate=False, no_cache=False, verbose=True, silent=False
    ):
        """
        Creates a T5Dataset from data.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        encoder_tokenizer = self.encoder_tokenizer
        decoder_tokenizer = self.decoder_tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache
        print('####################### load_and_cache_examples no_cache', no_cache, args.no_cache)
        if args.local_rank not in [-1, 0]:
            if args.use_habana:
                hb_utils.barrier()
            else:
                torch.distributed.barrier()

        if not no_cache and args.local_rank in [-1, 0]:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"

        if self.args.use_hf_datasets:
            dataset = load_hf_dataset(
                data, encoder_tokenizer, decoder_tokenizer, self.args
            )
        else:
            if args.dataset_class:
                CustomDataset = args.dataset_class
                dataset = CustomDataset(
                    encoder_tokenizer, decoder_tokenizer, args, data, mode
                )
            else:
                if args.model_type in ["bart", "mbart", "marian"]:
                    dataset = SimpleSummarizationDataset(
                        encoder_tokenizer, self.args, data, mode
                    )
                else:
                    dataset = Seq2SeqDataset(
                        encoder_tokenizer, decoder_tokenizer, self.args, data, mode,
                    )

        if args.local_rank == 0:
            if args.use_habana:
                hb_utils.barrier()
            else:
                torch.distributed.barrier()
        return dataset

    def _create_training_progress_scores(self, **kwargs):
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "eval_loss": [],
            "train_loss": [],
            **extra_metrics,
        }

        return training_progress_scores

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def save_model(
        self,
        output_dir=None,
        optimizer=None,
        scheduler=None,
        model=None,
        results=None,
        dataset=None,
    ):
        if self.args.local_rank not in [-1, 0]:
            return
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving model into {output_dir}")

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            self.save_model_args(output_dir)

            if self.args.model_type in [
                "bart",
                "mbart",
                "marian",
                "rag-token",
                "rag-sequence",
            ]:
                os.makedirs(os.path.join(output_dir), exist_ok=True)
                if self.args.use_habana:
                    import copy
                    model_to_save_clone = copy.deepcopy(model_to_save)
                    model_to_save_clone.to(torch.device("cpu"))
                    model_to_save_clone.save_pretrained(output_dir)
                    torch.save(model_to_save_clone.state_dict(), os.path.join(output_dir, "model.pt"))
                else:
                    model_to_save.save_pretrained(output_dir)
                self.config.save_pretrained(output_dir)

                if self.args.model_type in [
                    "bart",
                    "mbart",
                    "rag-token",
                    "rag-sequence",
                ]:
                    self.encoder_tokenizer.save_pretrained(output_dir)

                if (
                    self.args.model_type in ["rag-token", "rag-sequence"]
                    and self.args.save_knowledge_dataset_with_checkpoints
                ):
                    output_dataset_directory = os.path.join(
                        output_dir, "knowledge_dataset"
                    )
                    os.makedirs(output_dataset_directory, exist_ok=True)
                    self.retriever.save_pretrained(output_dataset_directory)
            else:
                os.makedirs(os.path.join(output_dir, "encoder"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "decoder"), exist_ok=True)
                self.encoder_config.save_pretrained(os.path.join(output_dir, "encoder"))
                self.decoder_config.save_pretrained(os.path.join(output_dir, "decoder"))

                model_to_save = (
                    self.model.encoder.module
                    if hasattr(self.model.encoder, "module")
                    else self.model.encoder
                )
                model_to_save.save_pretrained(os.path.join(output_dir, "encoder"))

                model_to_save = (
                    self.model.decoder.module
                    if hasattr(self.model.decoder, "module")
                    else self.model.decoder
                )

                model_to_save.save_pretrained(os.path.join(output_dir, "decoder"))

                self.encoder_tokenizer.save_pretrained(
                    os.path.join(output_dir, "encoder")
                )
                self.decoder_tokenizer.save_pretrained(
                    os.path.join(output_dir, "decoder")
                )

            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                if self.args.use_habana:
                        param_groups_copy = optimizer.state_dict()['param_groups']
                        state_dict_copy = {}
                        for st_key, st_val in optimizer.state_dict()['state'].items():
                            st_val_copy={}
                            for k, v in st_val.items():
                                st_val_copy[k] = v.to('cpu') if isinstance(v, torch.Tensor) else v
                            state_dict_copy[st_key] = st_val_copy
                        optim_dict = {}
                        optim_dict['state'] = state_dict_copy
                        optim_dict['param_groups'] = param_groups_copy
                        torch.save(optim_dict, os.path.join(output_dir, "optimizer.pt"))
                else:
                    torch.save(
                        optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                    )
                torch.save(
                    scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                )

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def _move_model_to_device(self):
        self.model.to(self.device)
        print(f"_move_model_to_device {self.device}")

    def _get_inputs_dict(self, batch):
        device = self.device
        if self.args.model_type in ["bart", "marian"]:
            pad_token_id = self.encoder_tokenizer.pad_token_id
            if self.args.debug:
                print('########### pad_token_id #############', pad_token_id)
            source_ids, source_mask, y = (
                batch["source_ids"],
                batch["source_mask"],
                batch["target_ids"],
            )
            y_ids = y[:, :-1].contiguous()
            labels = y[:, 1:].clone()
            labels[y[:, 1:] == pad_token_id] = -100
            ttnan = torch.isnan(labels)
            if self.args.debug and len(ttnan.nonzero()) > 0:
                print('############## labels have nan ############')
            if self.args.use_habana:
                y_ids = y_ids.to(dtype=torch.int32)
                labels = labels.to(dtype=torch.int32)
                if self.args.debug:
                    print('############## use_habana int32 y_ids labels ############')
            inputs = {
                "input_ids": source_ids.to(device),
                "attention_mask": source_mask.to(device),
                "decoder_input_ids": y_ids.to(device),
                "labels": labels.to(device),
            }
        elif self.args.model_type in ["mbart"]:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "decoder_input_ids": batch["decoder_input_ids"].to(device),
                "labels": batch["labels"].to(device),
            }
        elif self.args.model_type in ["rag-token", "rag-sequence"]:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "decoder_input_ids": batch["decoder_input_ids"].to(device),
                "labels": batch["decoder_input_ids"].to(device),
                "reduce_loss": True,
            }
        elif self.args.use_hf_datasets:
            labels = batch["decoder_input_ids"]
            labels_masked = labels.clone()
            labels_masked[labels_masked == self.decoder_tokenizer.pad_token_id] = -100

            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "decoder_input_ids": batch["decoder_input_ids"].to(device),
                "labels": labels_masked.to(device),
            }
        else:
            labels = batch[1]
            labels_masked = labels.clone()
            labels_masked[labels_masked == self.decoder_tokenizer.pad_token_id] = -100

            inputs = {
                "input_ids": batch[0].to(device),
                "decoder_input_ids": labels.to(device),
                "labels": labels_masked.to(device),
            }

        return inputs

    def save_model_args(self, output_dir):
        if self.args.local_rank not in [-1, 0]:
            return
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = Seq2SeqArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]

###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

from functools import partial
from itertools import chain

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM


def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    if "labels" not in result:
        result["labels"] = result["input_ids"].copy()
    return result


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.cache_dir,
        use_auth_token=True,
        num_proc=args.num_workers,
    )
    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]
    column_names = train_dataset.features

    def tokenize_function(example, eval=False):
        output_texts = []
        mask_labels_sizes = []
        for i in range(len(example["input"])):
            if "gov_report" in args.dataset_config_name:
                output_texts.append(
                    f"### Summarize the following text:\n {example['input'][i]}\n ### Summary:\n {example['output'][i]}{tokenizer.eos_token}"
                )
                if eval:
                    mask_labels_sizes.append(
                        f"### Summarize the following text:\n {example['input'][i]}\n ### Summary:\n"
                    )
            else:
                output_texts.append(
                    f"### {example['input'][i]}\n ### The answer is:\n {example['output'][i]}{tokenizer.eos_token}"
                )


        #input_ids = tokenizer(output_texts).input_ids
        input_ids = tokenizer(
            output_texts, padding="max_length", max_length=args.max_seq_length
        ).input_ids

        if eval:
            labels_ids = tokenizer(mask_labels_sizes).input_ids
            masked_labels = []
            for out, lb in zip(input_ids, labels_ids):
                ml = out.copy()
                ml[: len(lb)] = [-100] * len(lb)
                ml[-1] = -100
                masked_labels.append(ml)
            return {"input_ids": input_ids, "labels": masked_labels}
        else:
            return {"input_ids": input_ids, "labels": input_ids.copy()}

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=column_names,
    )
    valid_dataset = valid_dataset.map(
        partial(tokenize_function, eval=True),
        batched=True,
        num_proc=2,
        remove_columns=column_names,
    )

    def filter_function(example):
        to_keep = []
        for i in range(len(example["input_ids"])):
            if len(example["input_ids"][i]) > args.max_seq_length:
                to_keep.append(False)
            else:
                to_keep.append(True)
        return to_keep

    train_dataset = train_dataset.filter(
        filter_function,
        batched=True,
        # with_indices=True,
        num_proc=8,
        # remove_columns=column_names,
    )
    valid_dataset = valid_dataset.filter(
        filter_function,
        batched=True,
        # with_indices=True,
        num_proc=2,
        # remove_columns=column_names,
    )
    print(
        f"Before packing, Size of the train set: {len(train_dataset)}. Size of the validation set: {len(valid_dataset)}"
    )

    packing_method = partial(group_texts, block_size=args.max_seq_length)
    # Packing
    train_dataset = train_dataset.map(
        packing_method,
        batched=True,
        num_proc=8,
    )
    valid_dataset = valid_dataset.map(
        packing_method,
        batched=True,
        num_proc=2,
    )

    print(
        f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(valid_dataset)}"
    )

    return train_dataset, valid_dataset


def create_and_prepare_model(script_args, training_args):

    device_map = None
    bnb_config = None
    load_in_8bit = script_args.use_8bit_qunatization

    if script_args.use_4bit_qunatization:
        compute_dtype = getattr(torch, script_args.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=script_args.use_4bit_qunatization,
            bnb_4bit_quant_type=script_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=script_args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and script_args.use_4bit_qunatization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print(
                    "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
                )
                print("=" * 80)

    if script_args.use_4bit_qunatization or script_args.use_8bit_qunatization:
        device_map = "auto"  # {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_path,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map=device_map,
        use_cache=not training_args.gradient_checkpointing,
        trust_remote_code=True,
        use_flash_attention_2=True if script_args.use_flash_attn else False,
        torch_dtype=torch.bfloat16,
        max_position_embeddings=8192,
    )
    model.generation_config.attn_softmax_bf16 = True
    model.generation_config.use_flash_attention = True
    model.generation_config.flash_attention_recompute = script_args.flash_attention_recompute_enable
    model.generation_config.flash_attention_causal_mask = script_args.flash_attention_causal_mask_enable
    model.generation_config.flash_attention_fast_softmax = script_args.flash_attention_fast_softmax_enable
    model.generation_config.flash_attention_fp8 = script_args.flash_attention_fp8
    peft_config = None
    if script_args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            r=script_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=(
                None
                if script_args.lora_target_modules is None
                else script_args.lora_target_modules.split(",")
            ),
        )
        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model


def peft_module_casting_to_bf16(model, args):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

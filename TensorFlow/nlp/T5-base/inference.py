#!/usr/bin/env python3
#
# Copyright (c) 2020 Snapthat
# Source: https://github.com/snapthat/TF-T5-text-to-text/blob/master/snapthatT5/notebooks/TF-T5-%20Training.ipynb
#
###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import argparse
import transformers

from model import T5, encoder_max_len, decoder_max_len


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='T5-base Q&A inference')
parser.add_argument('--model_dir', type=str, default='/tmp/t5_base',
                    help='directory with saved model')
parser.add_argument('--data_dir', type=str, default='/data/huggingface',
                    help='directory with data and tokenizer')
parser.add_argument('--no_hpu', action='store_true',
                    help='do not load Habana modules = inference on CPU/GPU')
args = parser.parse_args()


def answer(model, tokenizer, context, question):
    input_text = f'answer_me: {question} context: {context} </s>'
    encoded_query = tokenizer(
        input_text, return_tensors='tf', padding='max_length',
        truncation=True, max_length=encoder_max_len)

    input_ids = encoded_query["input_ids"]
    attention_mask = encoded_query["attention_mask"]
    generated_answer = model.generate(
        input_ids, attention_mask=attention_mask,
        max_length=decoder_max_len, top_p=0.95, top_k=50, repetition_penalty=2)
    return tokenizer.decode(generated_answer.numpy()[0])


def main():
    if not args.no_hpu:
        # Load Habana module in order to do inference on HPU (Gaudi)
        from habana_frameworks.tensorflow import load_habana_module
        load_habana_module()

    checkpoint_path = os.path.join(args.model_dir, 'checkpoints')
    model = T5.from_pretrained(checkpoint_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.join(args.data_dir, 't5_base', 'tokenizer'))

    print('\nProvide context and ask model a question, for example:')
    context = ('In 2019 Habana Labs announced its first AI accelerator. Gaudi, '
               'named after famous Catalan architect, was designed to accelerate '
               'training of deep neural networks in data centers.')
    question = 'What is the name of the chip?'
    print('Context:', context)
    print('Question:', question)
    print("Answer: ", answer(model, tokenizer, context, question))

    while True:
        print('\nProvide context and ask model a question (to exit use Ctrl+C)')
        context = input('Context: ')
        question = input('Question: ')
        print("Answer: ", answer(model, tokenizer, context, question))


if __name__ == '__main__':
    main()

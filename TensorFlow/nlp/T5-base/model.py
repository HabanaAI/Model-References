# Copyright (c) 2020 Snapthat
# Source: https://github.com/snapthat/TF-T5-text-to-text/blob/master/snapthatT5/notebooks/TF-T5-%20Training.ipynb
#
###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import transformers
import tensorflow as tf

encoder_max_len = 250
decoder_max_len = 54


def compute_loss(self, labels, logits):
    logits = tf.cast(logits, tf.float32)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    return loss_fn(labels, logits)


# The original model does not return float32 logits which breaks accuracy
# Also, boolean_mask is added but not used at all which affects performance
transformers.modeling_tf_utils.TFCausalLanguageModelingLoss.compute_loss = compute_loss


class T5(transformers.TFT5ForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(
            name='accuracy')

    def train_step(self, data):
        x = data
        y = x["labels"]
        y = tf.reshape(y, [-1, 1])
        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            loss = outputs[0]
            logits = outputs[1]
            loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics if m != self.accuracy}

    def test_step(self, data):
        x = data
        y = x["labels"]
        y = tf.reshape(y, [-1, 1])
        output = self(x, training=False)
        loss = output[0]
        loss = tf.reduce_mean(loss)
        logits = output[1]

        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(y, logits)
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}


def encode(tokenizer, example,
           encoder_max_len=encoder_max_len, decoder_max_len=decoder_max_len):

    context = example['context']
    question = example['question']
    answer = example['answers']['text']

    question_plus = f"answer_me: {str(question)}"
    question_plus += f" context: {str(context)} </s>"

    answer_plus = ', '.join([i for i in list(answer)])
    answer_plus = f"{answer_plus} </s>"

    encoder_inputs = tokenizer(question_plus, truncation=True,
                               return_tensors='tf', max_length=encoder_max_len,
                               padding='max_length')

    decoder_inputs = tokenizer(answer_plus, truncation=True,
                               return_tensors='tf', max_length=decoder_max_len,
                               padding='max_length')

    input_ids = encoder_inputs['input_ids'][0]
    input_attention = encoder_inputs['attention_mask'][0]
    target_ids = decoder_inputs['input_ids'][0]
    target_attention = decoder_inputs['attention_mask'][0]

    outputs = {'input_ids': input_ids, 'attention_mask': input_attention,
               'labels': target_ids, 'decoder_attention_mask': target_attention}
    return outputs

# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python2, python3
###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company

# Changes:
# - Added line tf.get_logger().propagate = False
###############################################################################
"""Run ALBERT on SQuAD v1.1 using sentence piece tokenization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import os
import random
import time
from TensorFlow.nlp.albert import fine_tuning_utils
from TensorFlow.nlp.albert import modeling
from TensorFlow.nlp.albert import squad_utils
import six
import tensorflow
import tensorflow.compat.v1 as tf
from TensorFlow.common.tb_utils import write_hparams_v1, TBSummary
from habana_frameworks.tensorflow import load_habana_module
import TensorFlow.nlp.bert.utils.habana_hooks as habana_hooks

from TensorFlow.common.debug import dump_callback

try:
    import horovod.tensorflow as hvd
except ImportError:
    hvd = None

def horovod_enabled():
  return hvd is not None and hvd.is_initialized()

tf.get_logger().propagate = False

# pylint: disable=g-import-not-at-top
if six.PY2:
  import six.moves.cPickle as pickle
else:
  import pickle
# pylint: enable=g-import-not-at-top

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "albert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("spm_model_file", None,
                    "The model file for sentence piece tokenization.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string("train_feature_file", None,
                    "training feature file.")

flags.DEFINE_string(
    "predict_feature_file", None,
    "Location of predict features. If it doesn't exist, it will be written. "
    "If it does exist, it will be read.")

flags.DEFINE_string(
    "predict_feature_left_file", None,
    "Location of predict features not passed to TPU. If it doesn't exist, it "
    "will be written. If it does exist, it will be read.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "albert_hub_module_handle", None,
    "If set, the ALBERT hub module to use.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_integer("train_steps", -1,
                     "Total number of training steps to perform.")

flags.DEFINE_string("input_file", None,
                     "Squad TF_record file, generate it if not defined")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 5000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("save_summary_steps", 1,
                       "How often to save the summary data.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_einsum", False,
    "Whether to use tf.einsum or tf.reshape+tf.matmul for dense layers. Must "
    "be set to False for TFLite compatibility.")

flags.DEFINE_string(
    "export_dir",
    default=None,
    help=("The directory where the exported SavedModel will be stored."))

flags.DEFINE_bool('use_horovod', False, "Run training using horovod.")

flags.DEFINE_bool(
    "deterministic_run", False,
    "If set run will be deterministic (set random seed, read dataset in single thread, disable dropout)")

flags.DEFINE_bool('enable_scoped_allocator', False, "Enable scoped allocator optimization")


def validate_flags_or_throw(albert_config):
  """Validate the input FLAGS or throw an exception."""

  if not FLAGS.do_train and not FLAGS.do_predict and not FLAGS.export_dir:
    err_msg = "At least one of `do_train` or `do_predict` or `export_dir`" + "must be True."
    raise ValueError(err_msg)

  if FLAGS.do_train:
    if not FLAGS.train_file:
      raise ValueError(
          "If `do_train` is True, then `train_file` must be specified.")
  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")
    if not FLAGS.predict_feature_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_feature_file` must be "
          "specified.")
    if not FLAGS.predict_feature_left_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_feature_left_file` must be "
          "specified.")

  if FLAGS.max_seq_length > albert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the ALBERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, albert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def build_squad_serving_input_fn(seq_length):
  """Builds a serving input fn for raw input."""

  def _seq_serving_input_fn():
    """Serving input fn for raw images."""
    input_ids = tf.placeholder(
        shape=[1, seq_length], name="input_ids", dtype=tf.int32)
    input_mask = tf.placeholder(
        shape=[1, seq_length], name="input_mask", dtype=tf.int32)
    segment_ids = tf.placeholder(
        shape=[1, seq_length], name="segment_ids", dtype=tf.int32)

    inputs = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids
    }
    return tf.estimator.export.ServingInputReceiver(features=inputs,
                                                    receiver_tensors=inputs)

  return _seq_serving_input_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  albert_config = modeling.AlbertConfig.from_json_file(FLAGS.albert_config_file)

  if FLAGS.deterministic_run and (albert_config.attention_probs_dropout_prob or albert_config.hidden_dropout_prob):
        albert_config.attention_probs_dropout_prob = 0.0
        albert_config.hidden_dropout_prob = 0.0

  validate_flags_or_throw(albert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)
  model_dir = FLAGS.output_dir
  if horovod_enabled():
    model_dir = os.path.join(FLAGS.output_dir, "worker_" + str(hvd.rank()))

  tokenizer = fine_tuning_utils.create_vocab(
      vocab_file=FLAGS.vocab_file,
      do_lower_case=FLAGS.do_lower_case,
      spm_model_file=FLAGS.spm_model_file,
      hub_module=FLAGS.albert_hub_module_handle)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute_cluster.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  if FLAGS.do_train:
    iterations_per_loop = int(min(FLAGS.iterations_per_loop,
                                  FLAGS.save_checkpoints_steps))
  else:
    iterations_per_loop = FLAGS.iterations_per_loop

  # The Scoped Allocator Optimization is enabled by default unless disabled by a flag.
  if FLAGS.enable_scoped_allocator:
    from tensorflow.core.protobuf import rewriter_config_pb2  # pylint: disable=import-error

    session_config = tf.compat.v1.ConfigProto()
    session_config.graph_options.rewrite_options.scoped_allocator_optimization = rewriter_config_pb2.RewriterConfig.ON

    enable_op = session_config.graph_options.rewrite_options.scoped_allocator_opts.enable_op
    del enable_op[:]
    enable_op.append("HorovodAllreduce")
  else:
    session_config = None

  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=model_dir,
      keep_checkpoint_max=0,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      save_summary_steps=FLAGS.save_summary_steps,
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host),
      session_config=session_config)

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None

  train_batch_size = FLAGS.train_batch_size
  if horovod_enabled():
    train_batch_size = train_batch_size * hvd.size()

  if FLAGS.do_train:
    train_examples = squad_utils.read_squad_examples(
        input_file=FLAGS.train_file, is_training=True)
    num_train_steps = int(
        len(train_examples) / train_batch_size * FLAGS.num_train_epochs)
    if FLAGS.train_steps > 0:
      num_train_steps = FLAGS.train_steps
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    rng = random.Random(12345)
    rng.shuffle(train_examples)

  start_index = 0
  end_index = len(train_examples)
  per_worker_filenames = [os.path.join(FLAGS.output_dir, "train.tf_record")]
  worker_id = 0

  if horovod_enabled():
    per_worker_filenames = [os.path.join(FLAGS.output_dir, "train.tf_record_{}".format(i)) for i in range(hvd.local_size())]
    num_examples_per_rank = len(train_examples) // hvd.size()
    remainder = len(train_examples) % hvd.size()
    worker_id = hvd.rank()
    if worker_id < remainder:
      start_index = worker_id * (num_examples_per_rank + 1)
      end_index = start_index + num_examples_per_rank + 1
    else:
      start_index = worker_id * num_examples_per_rank + remainder
      end_index = start_index + (num_examples_per_rank)

  learning_rate = FLAGS.learning_rate

  model_fn = squad_utils.v1_model_fn_builder(
      albert_config=albert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      use_einsum=FLAGS.use_einsum,
      hub_module=FLAGS.albert_hub_module_handle)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  write_hparams_v1(FLAGS.output_dir, {
    'batch_size': FLAGS.train_batch_size,
    **{x: getattr(FLAGS, x) for x in FLAGS}
  })

  if FLAGS.do_train:
    # We write to a temporary file to avoid storing very large constant tensors
    # in memory.

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num orig examples = %d", len(train_examples))
    tf.logging.info("  Num steps = %d", num_train_steps)
    tf.logging.info("  Per-worker batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Total batch size = %d", train_batch_size)

    ## use pre-generated tf_record as input
    if FLAGS.input_file:
      if horovod_enabled():
        per_worker_filenames_temp = [os.path.join(FLAGS.input_file, "train.tf_record") for i in range(hvd.local_size())]
      else:
        per_worker_filenames_temp = [os.path.join(FLAGS.input_file, "train.tf_record")]

      if tf.gfile.Exists(per_worker_filenames_temp[hvd.local_rank() if horovod_enabled() else worker_id]):
        per_worker_filenames = per_worker_filenames_temp

    if not tf.gfile.Exists(per_worker_filenames[hvd.local_rank() if horovod_enabled() else worker_id]):
      train_writer = squad_utils.FeatureWriter(
          filename=per_worker_filenames[hvd.local_rank() if horovod_enabled() else worker_id], is_training=True)
      squad_utils.convert_examples_to_features(
          examples=train_examples[start_index:end_index],
          tokenizer=tokenizer,
          max_seq_length=FLAGS.max_seq_length,
          doc_stride=FLAGS.doc_stride,
          max_query_length=FLAGS.max_query_length,
          is_training=True,
          output_fn=train_writer.process_feature,
          do_lower_case=FLAGS.do_lower_case)
      tf.logging.info("  Num split examples = %d", train_writer.num_features)
      train_writer.close()

      del train_examples

    train_input_fn = squad_utils.input_fn_builder(
        input_file=per_worker_filenames,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        use_tpu=FLAGS.use_tpu,
        bsz=FLAGS.train_batch_size,
        is_v2=False)

    train_hooks = [habana_hooks.PerfLoggingHook(batch_size=train_batch_size, mode="train")]
    if horovod_enabled():
      train_hooks.append(hvd.BroadcastGlobalVariablesHook(0))

    if "range" == os.environ.get("HABANA_SYNAPSE_LOGGER", "False").lower():
      from habana_frameworks.tensorflow.synapse_logger_helpers import SynapseLoggerHook
      begin = 670
      end = begin + 10
      print("Begin: {}".format(begin))
      print("End: {}".format(end))
      train_hooks.append(SynapseLoggerHook(list(range(begin, end)), False))

    with dump_callback():
      estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=train_hooks)

  if FLAGS.do_predict:
    with tf.gfile.Open(FLAGS.predict_file) as predict_file:
      prediction_json = json.load(predict_file)["data"]

    eval_examples = squad_utils.read_squad_examples(
        input_file=FLAGS.predict_file, is_training=False)

    eval_writer = squad_utils.FeatureWriter(
        filename=os.path.join(model_dir, "eval.tf_record"), is_training=False)
    eval_features = []

    def append_feature(feature):
      eval_features.append(feature)
      eval_writer.process_feature(feature)

    squad_utils.convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=False,
        output_fn=append_feature,
        do_lower_case=FLAGS.do_lower_case)
    eval_writer.close()

    with tf.gfile.Open(os.path.join(model_dir, "eval_left.tf_record"), "wb") as fout:
      pickle.dump(eval_features, fout)

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_input_fn = squad_utils.input_fn_builder(
        input_file=os.path.join(model_dir, "eval.tf_record"),
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False,
        use_tpu=FLAGS.use_tpu,
        bsz=FLAGS.predict_batch_size,
        is_v2=False)

    eval_hooks = [habana_hooks.PerfLoggingHook(batch_size=FLAGS.predict_batch_size, mode="eval")]

    def get_result(checkpoint):
      """Evaluate the checkpoint on SQuAD 1.0."""
      # If running eval on the TPU, you will need to specify the number of
      # steps.
      reader = tf.train.NewCheckpointReader(checkpoint)
      global_step = reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
      all_results = []
      for result in estimator.predict(
          predict_input_fn, yield_single_examples=True,
          checkpoint_path=checkpoint, hooks=eval_hooks):
        if len(all_results) % 1000 == 0:
          tf.logging.info("Processing example: %d" % (len(all_results)))
        unique_id = int(result["unique_ids"])
        start_log_prob = [float(x) for x in result["start_log_prob"].flat]
        end_log_prob = [float(x) for x in result["end_log_prob"].flat]
        all_results.append(
            squad_utils.RawResult(
                unique_id=unique_id,
                start_log_prob=start_log_prob,
                end_log_prob=end_log_prob))

      output_prediction_file = os.path.join(
          model_dir, "predictions.json")
      output_nbest_file = os.path.join(
          model_dir, "nbest_predictions.json")

      result_dict = {}
      squad_utils.accumulate_predictions_v1(
          result_dict, eval_examples, eval_features,
          all_results, FLAGS.n_best_size, FLAGS.max_answer_length)
      predictions = squad_utils.write_predictions_v1(
          result_dict, eval_examples, eval_features, all_results,
          FLAGS.n_best_size, FLAGS.max_answer_length,
          output_prediction_file, output_nbest_file)

      return squad_utils.evaluate_v1(
          prediction_json, predictions), int(global_step)

    def _find_valid_cands(curr_step):
      filenames = tf.gfile.ListDirectory(model_dir)
      candidates = []
      for filename in filenames:
        if filename.endswith(".index"):
          ckpt_name = filename[:-6]
          idx = ckpt_name.split("-")[-1]
          if idx != "best" and int(idx) > curr_step:
            candidates.append(filename)
      return candidates

    output_eval_file = os.path.join(model_dir, "eval_results.txt")
    checkpoint_path = os.path.join(model_dir, "model.ckpt-best")
    key_name = "f1"
    writer = tf.gfile.GFile(output_eval_file, "w")
    if tf.gfile.Exists(checkpoint_path + ".index"):
      result = get_result(checkpoint_path)
      exact_match = result[0]["exact_match"]
      f1 = result[0]["f1"]
      with TBSummary(os.path.join(model_dir, 'eval')) as summary_writer:
          summary_writer.add_scalar('f1', f1, 0)
          summary_writer.add_scalar('exact_match', exact_match, 0)
      best_perf = result[0][key_name]
      global_step = result[1]
    else:
      global_step = -1
      best_perf = -1
      checkpoint_path = None
    while global_step < num_train_steps:
      steps_and_files = {}
      filenames = tf.gfile.ListDirectory(model_dir)
      for filename in filenames:
        if filename.endswith(".index"):
          ckpt_name = filename[:-6]
          cur_filename = os.path.join(model_dir, ckpt_name)
          if cur_filename.split("-")[-1] == "best":
            continue
          gstep = int(cur_filename.split("-")[-1])
          if gstep not in steps_and_files:
            tf.logging.info("Add {} to eval list.".format(cur_filename))
            steps_and_files[gstep] = cur_filename
      tf.logging.info("found {} files.".format(len(steps_and_files)))
      if not steps_and_files:
        tf.logging.info("found 0 file, global step: {}. Sleeping."
                        .format(global_step))
        time.sleep(60)
      else:
        for ele in sorted(steps_and_files.items()):
          step, checkpoint_path = ele
          if global_step >= step:
            if len(_find_valid_cands(step)) > 1:
              for ext in ["meta", "data-00000-of-00001", "index"]:
                src_ckpt = checkpoint_path + ".{}".format(ext)
                tf.logging.info("removing {}".format(src_ckpt))
                tf.gfile.Remove(src_ckpt)
            continue
          result, global_step = get_result(checkpoint_path)
          exact_match = result["exact_match"]
          f1 = result["f1"]
          with TBSummary(os.path.join(model_dir, 'eval')) as summary_writer:
            summary_writer.add_scalar('f1', f1, 0)
            summary_writer.add_scalar('exact_match', exact_match, 0)
          tf.logging.info("***** Eval results *****")
          for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
          if result[key_name] > best_perf:
            best_perf = result[key_name]
            for ext in ["meta", "data-00000-of-00001", "index"]:
              src_ckpt = checkpoint_path + ".{}".format(ext)
              tgt_ckpt = checkpoint_path.rsplit(
                  "-", 1)[0] + "-best.{}".format(ext)
              tf.logging.info("saving {} to {}".format(src_ckpt, tgt_ckpt))
              tf.gfile.Copy(src_ckpt, tgt_ckpt, overwrite=True)
              writer.write("saved {} to {}\n".format(src_ckpt, tgt_ckpt))
          writer.write("best {} = {}\n".format(key_name, best_perf))
          tf.logging.info("  best {} = {}\n".format(key_name, best_perf))

          if len(_find_valid_cands(global_step)) > 2:
            for ext in ["meta", "data-00000-of-00001", "index"]:
              src_ckpt = checkpoint_path + ".{}".format(ext)
              tf.logging.info("removing {}".format(src_ckpt))
              tf.gfile.Remove(src_ckpt)
          writer.write("=" * 50 + "\n")

    checkpoint_path = os.path.join(model_dir, "model.ckpt-best")
    result, global_step = get_result(checkpoint_path)
    tf.logging.info("***** Final Eval results *****")
    for key in sorted(result.keys()):
      tf.logging.info("  %s = %s", key, str(result[key]))
      writer.write("%s = %s\n" % (key, str(result[key])))
    writer.write("best perf happened at step: {}".format(global_step))

  if FLAGS.export_dir:
    tf.gfile.MakeDirs(FLAGS.export_dir)
    squad_serving_input_fn = (
        build_squad_serving_input_fn(FLAGS.max_seq_length))
    tf.logging.info("Starting to export model.")
    subfolder = estimator.export_saved_model(
        export_dir_base=os.path.join(FLAGS.export_dir, "saved_model"),
        serving_input_receiver_fn=squad_serving_input_fn)

    tf.logging.info("Starting to export TFLite.")
    converter = tf.lite.TFLiteConverter.from_saved_model(
        subfolder,
        input_arrays=["input_ids", "input_mask", "segment_ids"],
        output_arrays=["start_logits", "end_logits"])
    float_model = converter.convert()
    tflite_file = os.path.join(FLAGS.export_dir, "albert_model.tflite")
    with tf.gfile.GFile(tflite_file, "wb") as f:
      f.write(float_model)


if __name__ == "__main__":

  if FLAGS.deterministic_run:
    tensorflow.random.set_seed(1)
    tf.compat.v1.set_random_seed(1)

  if FLAGS.use_horovod:
    if hvd is None:
      raise RuntimeError(
          "Problem encountered during Horovod import. Please make sure that habana-horovod package is installed.")
    hvd.init()

  load_habana_module()


  flags.mark_flag_as_required("spm_model_file")
  flags.mark_flag_as_required("albert_config_file")
  flags.mark_flag_as_required("output_dir")

  tf.app.run()

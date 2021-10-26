import os
import time
import tensorflow as tf
from copy import deepcopy
from tensorboard.plugins.hparams import api as hp
from tensorflow.python.eager import context
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.summary import summary as tf_summary
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.compat.v1.keras.callbacks import TensorBoard, Callback


def _remove_prefix(s, prefix):
    if s.startswith(prefix):
        s = s[len(prefix):]
    return s


def _parse_precision():
    flag = os.environ.get('TF_BF16_CONVERSION')
    flag = flag or os.environ.get('TF_ENABLE_BF16_CONVERSION', '0')
    flag = flag.lower()
    try:
        value = int(flag)
    except:
        value = -1

    if flag == 'false' or value == 0:
        return 'fp32'
    elif flag == 'true' or value == 1:
        return 'bf16'
    return flag


def _set_precision_if_missing(hparams: dict):
    if 'precision' not in hparams:
        hparams['precision'] = _parse_precision()
    return hparams


def _copy_and_clean_hparams(hparams: dict):
    hparams_ = dict()
    for name, value in hparams.items():
        if isinstance(value, (str, bool, int, float)):
            hparams_[name] = value
            continue

        try:
            hparams_[name] = str(value)
            tf.compat.v1.logging.info(
                f'Type of parameter "{name}" is not one of (bool, int, float, str). '
                'It will be saved as a string.')
        except:
            tf.compat.v1.logging.info(
                f'Conversion of parameter "{name}" to string failed. '
                'Parameter will not be saved.')

    return hparams_


def write_hparams_v1(writer, hparams: dict):
    hparams = _copy_and_clean_hparams(hparams)
    hparams = _set_precision_if_missing(hparams)

    def _write(writer=writer):
        if isinstance(writer, str):
            writer = SummaryWriterCache.get(writer)
        summary = hp.hparams_pb(hparams).SerializeToString()
        writer.add_summary(summary)

    # If the Session was created already and we are in the scope, then
    # we can safely write summary.
    try:
        _write()
    except RuntimeError:
        # In case we are not in Session scope, we create Session here.
        with tf.compat.v1.Session():
            _write()


def write_hparams_v2(writer, hparams: dict):
    hparams = _copy_and_clean_hparams(hparams)
    hparams = _set_precision_if_missing(hparams)

    with writer.as_default():
        hp.hparams(hparams)


class ExamplesPerSecondEstimatorHook(tf.compat.v1.train.StepCounterHook):
    """Calculate and report global_step/sec and examples/sec during runtime."""
    # Copy-pasted from tensorflow_estimator/python/estimator/tpu/tpu_estimator.py

    def __init__(self,
                 batch_size=None,
                 every_n_steps=1,
                 every_n_secs=None,
                 output_dir=None,
                 summary_writer=None,
                 extra_metrics=None,
                 verbose=False):
        super().__init__(
            every_n_steps=every_n_steps,
            every_n_secs=every_n_secs,
            output_dir=output_dir,
            summary_writer=summary_writer)
        self._extra_metrics = extra_metrics or {}
        self._verbose = verbose
        if batch_size is not None:
            self._extra_metrics['examples/sec'] = batch_size

    def _add_summary(self, tag, value, step):
        Summary = tf.compat.v1.Summary
        global_step_summary = Summary(value=[
            Summary.Value(tag=tag, simple_value=value)
        ])
        self._summary_writer.add_summary(global_step_summary, step)
        if self._verbose:
            tf.compat.v1.logging.info(f'{tag}: {value}')

    def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
        global_step_per_sec = elapsed_steps / elapsed_time
        if self._summary_writer is not None:
            self._add_summary('global_step/sec',
                              global_step_per_sec, global_step)
            for name, factor in self._extra_metrics.items():
                value = factor * global_step_per_sec
                self._add_summary(name, value, global_step)

    def after_create_session(self, session, coord):
        self._timer.reset()


class ExamplesPerSecondKerasHook(Callback):
    def __init__(self,
                 every_n_steps=1,
                 every_n_secs=None,
                 output_dir=None,
                 summary_writer=None,
                 batch_size=None):
        self.writer = summary_writer or SummaryWriterCache.get(output_dir)
        self._timer = tf.compat.v1.train.SecondOrStepTimer(
            every_n_secs, every_n_steps)
        self._total_examples = 0
        self._should_trigger = True
        self._batch_size = batch_size

    def on_train_begin(self, logs=None):
        self._timer.reset()

    def on_train_batch_begin(self, batch, logs=None):
        self._should_trigger = self._timer.should_trigger_for_step(
            logs.get('batch', 0))

    def on_train_batch_end(self, batch, logs=None):
        step = logs.get('batch', 0)
        self._total_examples += logs.get('size', 0)
        if self._should_trigger:
            elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
                step)
            if elapsed_time is not None:
                total_examples = self._total_examples
                if self._batch_size is not None:
                    total_examples = self._batch_size * elapsed_steps
                self._log_and_record(
                    elapsed_steps, elapsed_time, step, total_examples)
                self._total_examples = 0

    def _log_and_record(self, elapsed_steps, elapsed_time,
                        global_step, total_examples=None):
        Summary = tf.compat.v1.Summary
        global_step_per_sec = elapsed_steps / elapsed_time
        if self.writer is not None:
            global_step_summary = Summary(value=[
                Summary.Value(
                    tag='global_step/sec', simple_value=global_step_per_sec)
            ])
            self.writer.add_summary(global_step_summary, global_step)
            if total_examples is not None:
                examples_per_sec = total_examples / elapsed_time
                example_summary = Summary(value=[
                    Summary.Value(tag='examples/sec',
                                  simple_value=examples_per_sec)
                ])
                self.writer.add_summary(example_summary, global_step)


class TBSummary(object):
    """
    Creates a proxy for FileWriter for TensorBoard.

    :param log_dir: - path where experiment is running (usually the same as
        model_dir in Estimator)
    """

    def __init__(self, log_dir: str):
        super().__init__()
        self._log_dir = log_dir
        self._session = None

    def __enter__(self):
        self._session = tf.compat.v1.Session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            self._session.close()
            self._session = None

    def add_scalar(self, tag, value, global_step=None):
        with self._session:
            writer = SummaryWriterCache.get(self._log_dir)
            summary = tf.compat.v1.Summary(
                value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
            event = tf.compat.v1.Event(summary=summary)
            event.wall_time = time.time()
            event.step = global_step
            writer.add_event(event)


class TensorBoardWithHParamsV1(TensorBoard):
    """
    Adds TensorBoard visualization to training process.

    Writes training tfevent file into default log directory, but
    stores evaluation in log_dir/eval subdirectory.
    """

    def __init__(self, hparams, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams = hparams
        self._train_summary = None
        self._eval_summary = None

    def _switch_writer(self, mode):
        self.writer = self._train_summary if mode == 'train' else self._eval_summary

    def _init_writer(self, model):
        """Sets file writer."""
        if context.executing_eagerly():
            raise NotImplementedError('hook does not support eager execution')

        self._train_summary = SummaryWriterCache.get(self.log_dir)
        self._eval_summary = SummaryWriterCache.get(
            os.path.join(self.log_dir, 'eval'))
        self._switch_writer('train')

        write_hparams_v1(self.writer, self.hparams)

    def _write_custom_summaries(self, step, logs=None):
        """
        This methods works on the assumption that metrics containing `val`
        in name are related to validation (that's the default in Keras).
        """

        logs = logs or {}
        train_logs = {}
        eval_logs = {}

        for name, value in logs.items():
            if 'val' in name:
                if name.startswith('batch_val_'):
                    name = 'batch_' + _remove_prefix(name, 'batch_val_')
                elif name.startswith('epoch_val_'):
                    name = _remove_prefix(name, 'epoch_val_')
                eval_logs[name] = value
            else:
                if name.startswith('batch_'):
                    name = _remove_prefix(name, 'batch_')
                train_logs[name] = value

        self._switch_writer('eval')
        super()._write_custom_summaries(step, eval_logs)
        self._switch_writer('train')
        super()._write_custom_summaries(step, train_logs)

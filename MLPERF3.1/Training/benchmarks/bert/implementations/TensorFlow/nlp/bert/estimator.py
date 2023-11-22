###############################################################################
# Copyright (C) 2021-2022 Habana Labs, Ltd. an Intel Company
###############################################################################

"""Estimator class based on original tensorflow implementation.
    It is extended to support additional training operations passed
    to it through EstimatorSpec as an ops_list attribute.
    Operations in ops_list are executed in order before EstimatorSpec.train_op
    is executed.

    This class is intended to work in tandem with training_ops.py which creates
    required operations for this class.

    This implementation will be used instead of original tensorflow estimator,
    if loop_unrolling_for_train_op is set to True in run_pretraining.py.
    """
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.summary import summary
from tensorflow.python.training import training
from tensorflow.python.training import training_util
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys

_estimator_api_gauge = monitoring.BoolGauge('/tensorflow/api/estimator_V1',
                                            'estimator api usage', 'method')


def _load_global_step_from_checkpoint_dir(checkpoint_dir):
  try:
    checkpoint_reader = tf.compat.v1.train.NewCheckpointReader(
        tf.train.latest_checkpoint(checkpoint_dir))
    return checkpoint_reader.get_tensor(tf.compat.v1.GraphKeys.GLOBAL_STEP)
  except:  # pylint: disable=bare-except
    return 0


def _check_hooks_type(hooks):
  """Returns hooks if all are `SessionRunHook`, raises TypeError otherwise."""
  hooks = list(hooks or [])
  for h in hooks:
    if not isinstance(h, tf.compat.v1.train.SessionRunHook):
      raise TypeError('Hooks must be a SessionRunHook, given: {}'.format(h))
  return hooks


def _check_listeners_type(saving_listeners):
  """Check listeners type."""
  listeners = list(saving_listeners or [])
  for l in listeners:
    if not isinstance(l, tf.compat.v1.train.CheckpointSaverListener):
      raise TypeError(
          'saving_listeners must be a list of CheckpointSaverListener, '
          'given: {}'.format(l))
  return listeners


class EstimatorSpec(tf.estimator.EstimatorSpec):

  def __new__(cls, ops_list, **kwargs):
    return super().__new__(cls, **kwargs)

  def __init__(self, ops_list, **kwargs):
    self.ops_list = ops_list
    super().__init__()


class Estimator(tf.estimator.Estimator):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  # Override to avoid assert
  def _assert_members_are_not_overridden(self):
    pass

  def train(self,
            input_fn,
            hooks=None,
            steps=None,
            max_steps=None,
            saving_listeners=None):
    """Trains a model given training data `input_fn`.

        Args:
        input_fn: A function that provides input data for training as minibatches.
            See [Premade Estimators](
            https://tensorflow.org/guide/premade_estimators#create_input_functions)
            for more information. The function should construct and return one of
            the following:
            * A `tf.data.Dataset` object: Outputs of `Dataset` object must be a
                tuple `(features, labels)` with same constraints as below.
            * A tuple `(features, labels)`: Where `features` is a `tf.Tensor` or a
                dictionary of string feature name to `Tensor` and `labels` is a
                `Tensor` or a dictionary of string label name to `Tensor`. Both
                `features` and `labels` are consumed by `model_fn`. They should
                satisfy the expectation of `model_fn` from inputs.
        hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
            callbacks inside the training loop.
        steps: Number of steps for which to train the model. If `None`, train
            forever or train until `input_fn` generates the `tf.errors.OutOfRange`
            error or `StopIteration` exception. `steps` works incrementally. If you
            call two times `train(steps=10)` then training occurs in total 20 steps.
            If `OutOfRange` or `StopIteration` occurs in the middle, training stops
            before 20 steps. If you don't want to have incremental behavior please
            set `max_steps` instead. If set, `max_steps` must be `None`.
        max_steps: Number of total steps for which to train model. If `None`,
            train forever or train until `input_fn` generates the
            `tf.errors.OutOfRange` error or `StopIteration` exception. If set,
            `steps` must be `None`. If `OutOfRange` or `StopIteration` occurs in the
            middle, training stops before `max_steps` steps. Two calls to
            `train(steps=100)` means 200 training iterations. On the other hand, two
            calls to `train(max_steps=100)` means that the second call will not do
            any iteration since first call did all 100 steps.
        saving_listeners: list of `CheckpointSaverListener` objects. Used for
            callbacks that run immediately before or after checkpoint savings.

        Returns:
        `self`, for chaining.

        Raises:
        ValueError: If both `steps` and `max_steps` are not `None`.
        ValueError: If either `steps` or `max_steps <= 0`.
        """
    _estimator_api_gauge.get_cell('train').set(True)
    if self.config.task_type in (run_config.TaskType.EVALUATOR,
                                 run_config.TaskType.PS):
      raise ValueError(
          'Train has been called wrong configuration. Please use '
          'tf.estimator.train_and_evaluate which calls proper API according '
          'to given configuration. Current configuration: {}.'.format(
              self.config))

    with context.graph_mode():
      if (steps is not None) and (max_steps is not None):
        raise ValueError('Can not provide both steps and max_steps.')
      if steps is not None and steps <= 0:
        raise ValueError('Must specify steps > 0, given: {}'.format(steps))
      if max_steps is not None and max_steps <= 0:
        raise ValueError(
            'Must specify max_steps > 0, given: {}'.format(max_steps))

      if max_steps is not None:
        start_step = _load_global_step_from_checkpoint_dir(self._model_dir)
        if max_steps <= start_step:
          logging.info('Skipping training since max_steps has already saved.')
          return self

      hooks = _check_hooks_type(hooks)
      hooks.extend(self._convert_train_steps_to_hooks(steps, max_steps))

      saving_listeners = _check_listeners_type(saving_listeners)
      loss = self._train_model(input_fn, hooks, saving_listeners)
      logging.info('Loss for final step: %s.', loss)
      return self

  def _train_model(self, input_fn, hooks, saving_listeners):
    if self._train_distribution:
      return self._train_model_distributed(input_fn, hooks, saving_listeners)
    else:
      return self._train_model_default(input_fn, hooks, saving_listeners)

  def _train_model_default(self, input_fn, hooks, saving_listeners):
    """Initiate training with `input_fn`, without `DistributionStrategies`.

        Args:
        input_fn: A function that provides input data for training as minibatches.
        hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
            callbacks inside the training loop.
        saving_listeners: list of `tf.train.CheckpointSaverListener` objects. Used
            for callbacks that run immediately before or after checkpoint savings.

        Returns:
        Loss from training
        """
    worker_hooks = []
    with tf.compat.v1.get_default_graph().as_default() as g, g.device(self._device_fn):
      global_step_tensor = self._create_and_assert_global_step(g)

    # Skip creating a read variable if _create_and_assert_global_step
    # returns None (e.g. tf.contrib.estimator.SavedModelEstimator).
    if global_step_tensor is not None:
      training_util._get_or_create_global_step_read(g)  # pylint: disable=protected-access
    features, labels, input_hooks = (
        self._get_features_and_labels_from_input_fn(input_fn, ModeKeys.TRAIN))
    worker_hooks.extend(input_hooks)
    estimator_spec = self._call_model_fn(features, labels, ModeKeys.TRAIN,
                                         self.config)
    return self._train_with_estimator_spec(estimator_spec, worker_hooks, hooks,
                                           global_step_tensor, saving_listeners)

  def _train_with_estimator_spec(self, estimator_spec, worker_hooks, hooks,
                                 global_step_tensor, saving_listeners):
    """Train a model with the given Estimator Spec."""
    if (self._warm_start_settings and
        not tf.train.latest_checkpoint(self._model_dir)):
      tf.compat.v1.logging.info('Warm-starting with WarmStartSettings: %s' %
                                (self._warm_start_settings,))
      tf.compat.v1.train.warm_start(*self._warm_start_settings)
    # Check if the user created a loss summary, and add one if they didn't.
    # We assume here that the summary is called 'loss'. If it is not, we will
    # make another one with the name 'loss' to ensure it shows up in the right
    # graph in TensorBoard.
    if not any([
        x.op.name == 'loss' for x in ops.get_collection(ops.GraphKeys.SUMMARIES)
    ]):
      summary.scalar('loss', estimator_spec.loss)
    ops.add_to_collection(ops.GraphKeys.LOSSES, estimator_spec.loss)
    worker_hooks.extend(hooks)
    worker_hooks.append(tf.compat.v1.train.NanTensorHook(estimator_spec.loss))
    if self._config.log_step_count_steps is not None:
      worker_hooks.append(
          tf.compat.v1.train.LoggingTensorHook(
              {
                  'loss': estimator_spec.loss,
                  'step': global_step_tensor
              },
              every_n_iter=self._config.log_step_count_steps))
    worker_hooks.extend(estimator_spec.training_hooks)

    if not (estimator_spec.scaffold.saver or
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SAVERS)):
      tf.compat.v1.add_to_collection(
          tf.compat.v1.GraphKeys.SAVERS,
          tf.compat.v1.train.Saver(
              sharded=True,
              max_to_keep=self._config.keep_checkpoint_max,
              keep_checkpoint_every_n_hours=(
                  self._config.keep_checkpoint_every_n_hours),
              defer_build=True,
              save_relative_paths=True))

    if (self._config.cluster_spec and type(self._train_distribution).__name__
        in ('CollectiveAllReduceStrategy', 'CollectiveAllReduceStrategyV1',
            'MultiWorkerMirroredStrategy')):
      return self._train_with_estimator_spec_distributed(
          estimator_spec, worker_hooks, saving_listeners)

    chief_hooks = []
    all_hooks = worker_hooks + list(estimator_spec.training_chief_hooks)
    saver_hooks = [
        h for h in all_hooks
        if isinstance(h, tf.compat.v1.train.CheckpointSaverHook)
    ]
    if (self._config.save_checkpoints_secs or
        self._config.save_checkpoints_steps):
      if not saver_hooks:
        chief_hooks = [
            tf.compat.v1.train.CheckpointSaverHook(
                self._model_dir,
                save_secs=self._config.save_checkpoints_secs,
                save_steps=self._config.save_checkpoints_steps,
                scaffold=estimator_spec.scaffold,
                save_graph_def=self._config.checkpoint_save_graph_def)
        ]
        saver_hooks = [chief_hooks[0]]
    if saving_listeners:
      if not saver_hooks:
        raise ValueError(
            'There should be a CheckpointSaverHook to use saving_listeners. '
            'Please set one of the RunConfig.save_checkpoints_steps or '
            'RunConfig.save_checkpoints_secs.')
    else:
      # It is expected to have one CheckpointSaverHook. If multiple, we pick
      # up the first one to add listener.
      for listener in saving_listeners:
        # pylint: disable=protected-access
        if listener not in saver_hooks[0]._listeners:
          saver_hooks[0]._listeners.append(listener)
        # pylint: disable=protected-access

    # Add summary hooks to worker 0 if we are running with a master, to ensure
    # that summaries are written at correct intervals even with long-running
    # evaluations.
    save_summary_steps = self._config.save_summary_steps
    log_step_count_steps = self._config.log_step_count_steps

    # Check existence of appropriate cluster spec fields, as well as master and
    # worker nodes. As master also performs evaluation, summary writing must
    # occur on a different node. The presence of a worker is also checked to
    # prevent reassigning hooks for single-replica jobs with just a master node.
    if (self._config.cluster_spec and self._config.cluster_spec.jobs and
        (run_config.TaskType.WORKER in self._config.cluster_spec.jobs) and
        (run_config.TaskType.MASTER in self._config.cluster_spec.jobs)):
      # Update config values to prevent the default hooks from being created on
      # the master or other workers.
      save_summary_steps = 0
      log_step_count_steps = None

    if (self._config.task_type == run_config.TaskType.WORKER and
        self._config.task_id == 0):
      if (self._config.save_summary_steps and
          self._config.save_summary_steps > 0):
        worker_hooks.append(
            tf.compat.v1.train.SummarySaverHook(
                save_steps=self._config.save_summary_steps,
                output_dir=self._config.model_dir,
                scaffold=estimator_spec.scaffold))

      if (self._config.log_step_count_steps and
          self._config.log_step_count_steps > 0):
        worker_hooks.append(
            tf.compat.v1.train.StepCounterHook(
                every_n_steps=self._config.log_step_count_steps,
                output_dir=self._config.model_dir))

    local_step_ph = tf.compat.v1.get_default_graph().get_tensor_by_name(
        'local_step:0')
    with training.MonitoredTrainingSession(
        master=self._config.master,
        is_chief=self._config.is_chief,
        checkpoint_dir=self._model_dir,
        scaffold=estimator_spec.scaffold,
        hooks=worker_hooks,
        chief_only_hooks=(tuple(chief_hooks) +
                          tuple(estimator_spec.training_chief_hooks)),
        save_checkpoint_secs=0,  # Saving is handled by a hook.
        save_summaries_steps=save_summary_steps,
        config=self._session_config,
        max_wait_secs=self._config.session_creation_timeout_secs,
        log_step_count_steps=log_step_count_steps,
        save_graph_def=self._config.checkpoint_save_graph_def) as mon_sess:
      loss = None
      current_step = 0
      while not mon_sess.should_stop():
        current_step += 1
        # just as keras(https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/engine/training.py#L1093),
        # trace should be enabled for every step
        with trace.Trace('train', step_num=current_step, _r=1):
          local_step = 1
          for subop in estimator_spec.ops_list:
            mon_sess.run([subop], feed_dict={local_step_ph: local_step})
            local_step += 1
          _, loss = mon_sess.run([estimator_spec.train_op, estimator_spec.loss],
                                 feed_dict={local_step_ph: local_step})
      if current_step == 0:
        tf.compat.v1.logging.warn('Training with estimator made no steps. '
                                  'Perhaps input is empty or misspecified.')
    return loss

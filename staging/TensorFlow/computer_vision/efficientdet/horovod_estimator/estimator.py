try:
    from tensorflow_estimator import estimator
except ImportError:
    from tensorflow import estimator

import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import training
from tensorflow.python.training import warm_starting_util
from tensorflow.python.training.monitored_session import USE_DEFAULT, Scaffold, MonitoredSession, ChiefSessionCreator

from .utis import hvd_info_rank0, is_rank0

estimator.Estimator._assert_members_are_not_overridden = lambda _: None


def MonitoredTrainingSession(master='',
                             is_chief=True,
                             checkpoint_dir=None,
                             scaffold=None,
                             hooks=None,
                             chief_only_hooks=None,
                             save_checkpoint_secs=USE_DEFAULT,
                             save_summaries_steps=USE_DEFAULT,
                             save_summaries_secs=USE_DEFAULT,
                             config=None,
                             stop_grace_period_secs=120,
                             log_step_count_steps=100,
                             save_checkpoint_steps=USE_DEFAULT,
                             summary_dir=None):
    if save_summaries_steps == USE_DEFAULT and save_summaries_secs == USE_DEFAULT:
        save_summaries_steps = 100
        save_summaries_secs = None
    elif save_summaries_secs == USE_DEFAULT:
        save_summaries_secs = None
    elif save_summaries_steps == USE_DEFAULT:
        save_summaries_steps = None

    if (save_checkpoint_steps == USE_DEFAULT and
            save_checkpoint_secs == USE_DEFAULT):
        save_checkpoint_steps = None
        save_checkpoint_secs = 600
    elif save_checkpoint_secs == USE_DEFAULT:
        save_checkpoint_secs = None
    elif save_checkpoint_steps == USE_DEFAULT:
        save_checkpoint_steps = None

    scaffold = scaffold or Scaffold()

    all_hooks = []
    if is_chief and chief_only_hooks:
        all_hooks.extend(chief_only_hooks)

    session_creator = ChiefSessionCreator(
        scaffold=scaffold,
        checkpoint_dir=checkpoint_dir,
        master=master,
        config=config)

    summary_dir = summary_dir or checkpoint_dir
    if summary_dir:
        if (save_summaries_steps and save_summaries_steps > 0) or (
                save_summaries_secs and save_summaries_secs > 0):
            all_hooks.append(
                basic_session_run_hooks.SummarySaverHook(
                    scaffold=scaffold,
                    save_steps=save_summaries_steps,
                    save_secs=save_summaries_secs,
                    output_dir=summary_dir))

    if checkpoint_dir:
        if (save_checkpoint_secs and save_checkpoint_secs > 0) or (
                save_checkpoint_steps and save_checkpoint_steps > 0):
            all_hooks.append(
                basic_session_run_hooks.CheckpointSaverHook(
                    checkpoint_dir,
                    save_steps=save_checkpoint_steps,
                    save_secs=save_checkpoint_secs,
                    scaffold=scaffold))

    if hooks:
        all_hooks.extend(hooks)

    hvd_info_rank0('all hooks {}'.format(all_hooks))
    return MonitoredSession(
        session_creator=session_creator,
        hooks=all_hooks,
        stop_grace_period_secs=stop_grace_period_secs)


class HorovodEstimator(tf.estimator.Estimator):
    def __init__(self, model_fn, model_dir, config=None, params=None, warm_start_from=None):
        super(HorovodEstimator, self).__init__(model_fn=model_fn, model_dir=model_dir, config=config, params=params,
                                               warm_start_from=warm_start_from)

    def _train_with_estimator_spec(self, estimator_spec, worker_hooks, hooks, global_step_tensor, saving_listeners):
        """Train a model with the given Estimator Spec."""
        if self._warm_start_settings:
            logging.info('Warm-starting with WarmStartSettings: %s' % (self._warm_start_settings,))
            warm_starting_util.warm_start(*self._warm_start_settings)
        # Check if the user created a loss summary, and add one if they didn't.
        # We assume here that the summary is called 'loss'. If it is not, we will
        # make another one with the name 'loss' to ensure it shows up in the right
        # graph in TensorBoard.
        # if not any([x.op.name == 'loss' for x in ops.get_collection(ops.GraphKeys.SUMMARIES)]):
        #     summary.scalar('loss', estimator_spec.loss)
        ops.add_to_collection(ops.GraphKeys.LOSSES, estimator_spec.loss)
        worker_hooks.extend(hooks)
        # worker_hooks.extend([
        #     training.NanTensorHook(estimator_spec.loss)
        # ])

        worker_hooks.extend(estimator_spec.training_hooks)

        if not (estimator_spec.scaffold.saver or
                ops.get_collection(ops.GraphKeys.SAVERS)):
            ops.add_to_collection(
                ops.GraphKeys.SAVERS,
                training.Saver(
                    sharded=True,
                    max_to_keep=self._config.keep_checkpoint_max,
                    keep_checkpoint_every_n_hours=(
                        self._config.keep_checkpoint_every_n_hours),
                    defer_build=True,
                    save_relative_paths=True))

        chief_hooks = []
        all_hooks = worker_hooks + list(estimator_spec.training_chief_hooks)
        saver_hooks = [h for h in all_hooks if isinstance(h, training.CheckpointSaverHook)]
        if (self._config.save_checkpoints_secs or
                self._config.save_checkpoints_steps):
            if not saver_hooks:
                chief_hooks = [
                    training.CheckpointSaverHook(
                        self._model_dir,
                        save_secs=self._config.save_checkpoints_secs,
                        save_steps=self._config.save_checkpoints_steps,
                        scaffold=estimator_spec.scaffold)
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
                saver_hooks[0]._listeners.extend(saving_listeners)  # pylint: disable=protected-access

        if is_rank0():
            log_step_count_steps = self._config.log_step_count_steps
            checkpoint_dir = self.model_dir
            chief_only_hooks = (tuple(chief_hooks) + tuple(estimator_spec.training_chief_hooks))
        else:
            log_step_count_steps = None
            checkpoint_dir = None
            chief_only_hooks = None

        with MonitoredTrainingSession(
                master=self._config.master,
                is_chief=is_rank0(),
                checkpoint_dir=checkpoint_dir,
                scaffold=estimator_spec.scaffold,
                hooks=worker_hooks,
                chief_only_hooks=chief_only_hooks,
                save_checkpoint_secs=0,  # Saving is handled by a hook.
                save_summaries_steps=self._config.save_summary_steps,
                config=self._session_config,
                log_step_count_steps=log_step_count_steps) as mon_sess:
            loss = None
            while not mon_sess.should_stop():
                _, loss = mon_sess.run([estimator_spec.train_op, estimator_spec.loss])
        return loss

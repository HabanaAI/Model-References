import os
from enum import Enum
import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.session_run_hook import SessionRunArgs

from demo.library_loader import load_habana_module

HLS1_MODULE_CNT = 8


def comm_size():
    return int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))


def comm_rank():
    return int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))


def comm_local_size():
    return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", 1))


def comm_local_rank():
    return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))


def is_hierarchical():
    return (bool(os.environ.get("HOROVOD_HIERARCHICAL_ALLREDUCE", 0)) or \
            bool(os.environ.get("HOROVOD_HCL_HIERARCHICAL_ALLREDUCE", 0))) and \
           comm_size() > comm_local_size()

def get_hw_module_id(rank):
    return rank % HLS1_MODULE_CNT

class HorovodHelpers:

    _hvd_init_done = False
    _hvd_init_failed = False
    _hvd = None
    _hvd_rank_prefix = ""

    @staticmethod
    def _try_horovod_init(horovod_required=False):
        if not HorovodHelpers._hvd_init_done:
            try:
                HorovodHelpers._horovod_init()
            except:
                HorovodHelpers._hvd = None
                logging.warning(
                    "Problem encountered when setting horovod. Multinode scenarios might not by available")
                if horovod_required:
                    raise
            finally:
                # It was either success or not, either way we do not want to try again
                HorovodHelpers._hvd_init_done = True

    @staticmethod
    def _set_env_prefix(env_name, prefix, leave_empty):
        old_prefix = os.environ.get(env_name, "")
        if leave_empty and not old_prefix:
            return
        new_prefix = f"{old_prefix}{prefix}"
        os.environ[env_name] = new_prefix

    @staticmethod
    def _horovod_init():
        size = comm_size()
        rank = comm_rank()

        if is_hierarchical():
            os.environ["HLS1_MODULE_ID"] = str(comm_local_rank())
            os.environ["ID"] = str(comm_local_rank())
        else:
            if comm_size() > 1:
                os.environ["HLS1_MODULE_ID"] = str(get_hw_module_id(rank))
                os.environ["ID"] = str(get_hw_module_id(rank))

        # Init synapse logger (if required)
        synapse_logger_init()
        # Init TF Module (for CPU Allocator)
        load_habana_module()
        import horovod.tensorflow as hvd
        hvd.init()
        assert rank == hvd.rank(), "There is possible rank mismatch between mpi and horovod"

        # Make sure every rank logging to different file
        # Only important on the same machine - so pretty much every scenarios
        if hvd.size() > 1:
            rank_prefix = "worker{}".format(rank)
            HorovodHelpers._set_env_prefix(
                "HBN_TF_GRAPH_PREFIX", rank_prefix, False)
            HorovodHelpers._set_env_prefix(
                "TF_DUMP_GRAPH_PREFIX", rank_prefix, True)
            HorovodHelpers._hvd_rank_prefix = rank_prefix
        HorovodHelpers._hvd = hvd

    @staticmethod
    def horovod(horovod_required=False):
        HorovodHelpers._try_horovod_init(horovod_required)
        if horovod_required:
            assert HorovodHelpers._hvd != None, "Horovod is required, but not ready."
        return HorovodHelpers._hvd


class Hvd(object):
    def __getattr__(self, attr):
        return getattr(HorovodHelpers.horovod(), attr)


hvd = Hvd()


class SynapseLoggerType(Enum):
    NONE = 0
    ALL = 1
    RANGE = 2


class SynapseLoggerHelpers:
    _syn_logger_init_done = False
    _syn_logger_setup_done = False
    _syn_logger = None

    @staticmethod
    def synapse_logger():
        if not SynapseLoggerHelpers._syn_logger_init_done and SynapseLoggerHelpers._synapse_logger_required():
            import sys
            new_path = os.path.join(os.environ["BUILD_ROOT_LATEST"])
            sys.path.insert(0, new_path)
            import py_synapse_logger as syn_log
            SynapseLoggerHelpers._syn_logger = syn_log
            SynapseLoggerHelpers._syn_logger_init_done = True
        return SynapseLoggerHelpers._syn_logger

    @staticmethod
    def _get_synapse_logger_env():
        return os.environ.get("HABANA_SYNAPSE_LOGGER", "False").lower()

    @staticmethod
    def _synapse_logger_required():
        return SynapseLoggerHelpers._get_synapse_logger_env() in ["true", "t", "1", "on", "all", "range"]

    @staticmethod
    def _synapse_logger_type():
        if SynapseLoggerHelpers._synapse_logger_required():
            if SynapseLoggerHelpers._get_synapse_logger_env() in ["range"]:
                return SynapseLoggerType.RANGE
            else:
                return SynapseLoggerType.ALL
        return SynapseLoggerType.NONE

    @staticmethod
    def _setup_synapse_logger(log_name_prefix=""):
        syn_log = SynapseLoggerHelpers.synapse_logger()
        if not syn_log:
            logger_required = SynapseLoggerHelpers._synapse_logger_required()
            assert not logger_required, "SynapseLoggerHelpers.synapse_logger() returns None when logger is enabled"
            return
        syn_log.command("stop_data_capture")
        if log_name_prefix:
            synapse_logger_set_file_name = "file_name={}".format(
                "{}.local.synapse_log".format(log_name_prefix))
            syn_log.command(synapse_logger_set_file_name)
            syn_log.command("category_mask=0x3")
            if SynapseLoggerHelpers._synapse_logger_type() == SynapseLoggerType.ALL:
                syn_log.command("restart")


def synapse_logger_init():
    SynapseLoggerHelpers.synapse_logger()
    rank_no = comm_rank()
    SynapseLoggerHelpers._setup_synapse_logger(f"worker{rank_no}")


def horovod_enabled():
    return HorovodHelpers._hvd != None and HorovodHelpers._hvd.size() > 1


def hvd_rank_prefix():
    return HorovodHelpers._hvd_rank_prefix

# Functions below will bahave exactly like their hvd.* method counterparts,
# but will crash if horovod is not available.


def hvd_init():
    return HorovodHelpers.horovod(horovod_required=True)


def hvd_size():
    return HorovodHelpers.horovod(horovod_required=True).size()


def hvd_rank():
    return HorovodHelpers.horovod(horovod_required=True).rank()

def hvd_local_size():
    return HorovodHelpers.horovod(horovod_required=True).local_size()

def synapse_logger_start(step_idx, profile_hw=False):
    syn_log = SynapseLoggerHelpers.synapse_logger()
    if syn_log:
        log_name_prefix = "worker_{}_step_{}".format(hvd.rank(), step_idx)
        synapse_logger_set_file_name = "file_name={}".format(
            "{}.local.synapse_log".format(log_name_prefix))
        syn_log.command(synapse_logger_set_file_name)
        syn_log.command("category_mask=0x3")
        syn_log.command("restart")
        if profile_hw:
            if hvd.size() == 1 or hvd.rank() == 0:
                syn_log.start_hw_profile()
    else:
        logging.warning(
            "Synapse logger has not been enabled. Unable to start logging.")


def synapse_logger_stop(profile_hw=False):
    syn_log = SynapseLoggerHelpers.synapse_logger()
    if syn_log:
        if profile_hw:
            if hvd.size() == 1 or hvd.rank() == 0:
                syn_log.stop_hw_profile()
        syn_log.command("disable")
    else:
        logging.warning(
            "Synapse logger has not been enabled. Unable to start logging.")


def synapse_logger_log(log_msg):
    syn_log = SynapseLoggerHelpers.synapse_logger()
    if syn_log:
        syn_log.put_log(log_msg)


class SynapseLoggerHook(tf.estimator.SessionRunHook):

    def __init__(self, steps_to_log, profile_hw=False):
        tf.estimator.SessionRunHook.__init__(self)
        self._step_cnt = 0
        self._profile_hw = profile_hw
        if not isinstance(steps_to_log, list):
            self._steps_to_log = [steps_to_log]
        else:
            self._steps_to_log = steps_to_log
        self._syn_logger_running = False

    def before_run(self, run_context):
        if not self._syn_logger_running:
            if self._step_cnt in self._steps_to_log:
                synapse_logger_start(
                    self._step_cnt, profile_hw=self._profile_hw)
                self._syn_logger_running = True
        if self._syn_logger_running:
            synapse_logger_log(
                f'"name":"call:step", "ph":"B", "cname":"vsync_highlight_color", "func":"void step(int it)", "args":{{"it":{self._step_cnt}}}'
            )

    def after_run(self, run_context, run_values):
        self._step_cnt += 1

        if self._syn_logger_running:
            synapse_logger_log(f'"name":"call:step", "ph":"E"')
            if self._step_cnt not in self._steps_to_log:
                synapse_logger_stop(profile_hw=self._profile_hw)
                self._syn_logger_running = False

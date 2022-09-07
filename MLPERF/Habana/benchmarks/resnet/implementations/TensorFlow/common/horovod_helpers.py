###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
import os
from enum import Enum
import json
import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.session_run_hook import SessionRunArgs

from habana_frameworks.tensorflow import load_habana_module

HLS1_MODULE_CNT = 8
HCL_CONFIG_PATH_VAR = "HCL_CONFIG_PATH"

class Framework(Enum):
    TENSORFLOW = 0
    KERAS = 1

def comm_size():
    return int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))


def comm_rank():
    return int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))


def comm_local_size():
    return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", 1))


def comm_local_rank():
    return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))


def is_hierarchical():
    return (bool(os.environ.get("HOROVOD_HIERARCHICAL_ALLREDUCE", 0)) or
            bool(os.environ.get("HOROVOD_HCL_HIERARCHICAL_ALLREDUCE", 0))) and \
        comm_size() > comm_local_size()


def get_hw_module_id(rank):
    return rank % HLS1_MODULE_CNT


def get_hcl_config():
    hcl_config_path = os.environ.get(HCL_CONFIG_PATH_VAR)
    assert hcl_config_path != None, "{} is not set, but required by Horovod".format(HCL_CONFIG_PATH_VAR)
    assert os.path.isfile(hcl_config_path), "{} points to not accessible file: {}".format(HCL_CONFIG_PATH_VAR, hcl_config_path)

    with open(hcl_config_path, "r") as hcl_config_file:
        try:
            return json.load(hcl_config_file)
        except json.JSONDecodeError:
            logging.error("{} indicated by {} is not valid json file".format(hcl_config_path, HCL_CONFIG_PATH_VAR))
            raise


def get_hcl_type(hcl_config):
    return hcl_config["HCL_TYPE"]


class HorovodHelpers:

    _hvd_init_done = False
    _hvd_init_failed = False
    _hvd = None
    _hvd_rank_prefix = ""

    @staticmethod
    def _try_horovod_init(framework, horovod_required=False):
        if not HorovodHelpers._hvd_init_done:
            try:
                HorovodHelpers._horovod_init(framework)
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
    def _horovod_init(framework):
        size = comm_size()
        rank = comm_rank()

        hcl_config = get_hcl_config()
        hcl_type = get_hcl_type(hcl_config)

        if hcl_type != "HLS1-H":
            # All env variables should be set before loading_habana_modules
            if is_hierarchical():
                os.environ["HLS1_MODULE_ID"] = str(comm_local_rank())
                os.environ["ID"] = str(comm_local_rank())
            else:
                if size > 1:
                    os.environ["HLS1_MODULE_ID"] = str(get_hw_module_id(rank))
                    os.environ["ID"] = str(get_hw_module_id(rank))

        # Make sure every rank logging to different file
        # Only important on the same machine - so pretty much every scenarios
        if size > 1:
            rank_prefix = "rank_{}_".format(rank)
            HorovodHelpers._set_env_prefix(
                "TF_RANK_PREFIX", rank_prefix, False)
            HorovodHelpers._set_env_prefix(
                "HBN_TF_GRAPH_PREFIX", rank_prefix, False)
            HorovodHelpers._set_env_prefix(
                "TF_DUMP_GRAPH_PREFIX", rank_prefix, True)
            HorovodHelpers._hvd_rank_prefix = rank_prefix

        # Init synapse logger (if required)
        synapse_logger_init()
        # Init TF Module (for CPU Allocator)
        load_habana_module()
        # Temporary WA to support both paths: with and without habana_frameworks package installed
        try:
            from habana_frameworks.tensorflow.lib_utils import libraries_location
            tf.load_library(os.path.join(libraries_location, "libsynapse_helpers.so." + tf.__version__))
        except:
            logging.warning(
                "Can't import habana_frameworks, trying to run anyway")
        if framework == Framework.TENSORFLOW:
            import horovod.tensorflow as hvd
        elif framework == Framework.KERAS:
            import horovod.tensorflow.keras as hvd
        else:
            raise Exception("Specified framework: {} is not supported by horovod_helpers".format(framework))

        hvd.init()
        assert rank == hvd.rank(), "There is possible rank mismatch between mpi and horovod"
        HorovodHelpers._hvd = hvd

    @staticmethod
    def horovod(framework=Framework.TENSORFLOW, horovod_required=False):
        HorovodHelpers._try_horovod_init(framework, horovod_required)
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
            import habana_frameworks.tensorflow as htf
            sys.path.insert(0, htf.sysconfig.get_lib_dir())
            import habana_frameworks.tensorflow.py_synapse_logger as syn_log
            SynapseLoggerHelpers._syn_logger = syn_log
            SynapseLoggerHelpers._syn_logger_init_done = True
            syn_log.command("disable")
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


def hvd_init(framework=Framework.TENSORFLOW):
    return HorovodHelpers.horovod(framework, horovod_required=True)


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
        syn_log.put_log(log_msg, 1)


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

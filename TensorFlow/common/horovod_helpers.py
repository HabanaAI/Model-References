###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
import os
from enum import Enum
import json
import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging


class Framework(Enum):
    TENSORFLOW = 0
    KERAS = 1


class HorovodHelpers:

    _hvd_init_done = False
    _hvd_init_failed = False
    _hvd = None
    _hvd_rank_prefix = ""

    @staticmethod
    def _try_horovod_init(framework, horovod_required=False):
        if not HorovodHelpers._hvd_init_done:
            try:
                if framework == Framework.TENSORFLOW:
                    import horovod.tensorflow as hvd
                elif framework == Framework.KERAS:
                    import horovod.tensorflow.keras as hvd
                hvd.init()
                HorovodHelpers._hvd = hvd
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
    def horovod(framework=Framework.TENSORFLOW, horovod_required=False):
        HorovodHelpers._try_horovod_init(framework, horovod_required)
        if horovod_required:
            assert HorovodHelpers._hvd != None, "Horovod is required, but not ready."
        return HorovodHelpers._hvd


class Hvd(object):
    def __getattr__(self, attr):
        return getattr(HorovodHelpers.horovod(), attr)


hvd = Hvd()


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

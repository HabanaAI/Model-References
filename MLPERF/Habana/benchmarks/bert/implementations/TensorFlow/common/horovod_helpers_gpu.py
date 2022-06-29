import os
import horovod.tensorflow as hvd

def hvd_init():
    hvd.init()

def hvd_size():
    return hvd.size()

def hvd_rank():
    return hvd.rank()

def horovod_enabled():
    try:
        return hvd.size() > 1
    except ValueError:
        return False

def comm_local_rank():
    return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
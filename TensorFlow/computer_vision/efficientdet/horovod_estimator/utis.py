import socket

from TensorFlow.common.horovod_helpers import hvd, hvd_init, horovod_enabled
import tensorflow.compat.v1 as tf

def is_rank0():
    if horovod_enabled():
        return hvd.rank() == 0
    else:
        return True


global IS_HVD_INIT
IS_HVD_INIT = False


def hvd_try_init():
    global IS_HVD_INIT
    if not IS_HVD_INIT and horovod_enabled():
        hvd_init()
        IS_HVD_INIT = True

        tf.get_logger().propagate = False
        if hvd.rank() == 0:
            tf.logging.set_verbosity('INFO')
        else:
            tf.logging.set_verbosity('WARN')


def hvd_info(msg):
    hvd_try_init()
    if horovod_enabled():
        head = 'hvd rank{}/{} in {}'.format(hvd.rank(), hvd.size(), socket.gethostname())
    else:
        head = '{}'.format(socket.gethostname())
    tf.logging.info('{}: {}'.format(head, msg))


def hvd_info_rank0(msg, with_head=True):
    hvd_try_init()
    if is_rank0():
        if with_head:
            if horovod_enabled():
                head = 'hvd only rank{}/{} in {}'.format(hvd.rank(), hvd.size(), socket.gethostname())
            else:
                head = '{}'.format(socket.gethostname())
            tf.logging.info('{}: {}'.format(head, msg))
        else:
            tf.logging.info(msg)


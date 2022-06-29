import socket
try:
    import horovod.tensorflow as hvd
except ImportError:
    hvd = None
import tensorflow.compat.v1 as tf

def is_rank0():
    if hvd is not None and hvd.is_initialized():
        return hvd.rank() == 0
    else:
        return True


global IS_HVD_INIT
IS_HVD_INIT = False


def hvd_try_init():
    global IS_HVD_INIT
    if not IS_HVD_INIT and hvd is not None and hvd.is_initialized():
        hvd.init()
        IS_HVD_INIT = True

        tf.get_logger().propagate = False
        if hvd.rank() == 0:
            tf.logging.set_verbosity('INFO')
        else:
            tf.logging.set_verbosity('WARN')


def hvd_info(msg):
    hvd_try_init()
    if hvd is not None and hvd.is_initialized():
        head = 'hvd rank{}/{} in {}'.format(hvd.rank(), hvd.size(), socket.gethostname())
    else:
        head = '{}'.format(socket.gethostname())
    tf.logging.info('{}: {}'.format(head, msg))


def hvd_info_rank0(msg, with_head=True):
    hvd_try_init()
    if is_rank0():
        if with_head:
            if hvd is not None and hvd.is_initialized():
                head = 'hvd only rank{}/{} in {}'.format(hvd.rank(), hvd.size(), socket.gethostname())
            else:
                head = '{}'.format(socket.gethostname())
            tf.logging.info('{}: {}'.format(head, msg))
        else:
            tf.logging.info(msg)


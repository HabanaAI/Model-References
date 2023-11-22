#!/bin/bash

# Clear caches
PROC_FS=${PROC_FS:-"/proc"}
sync && echo 3 > $PROC_FS/sys/vm/drop_caches

#Add timestamp per worker for init
python3 -c "
import mlperf_logging.mllog.constants as mllog_constants
from mlperf_logging_utils import mllogger
mllogger.event(key=mllog_constants.CACHE_CLEAR, value=True)
mllogger.start(key=mllog_constants.INIT_START)
"

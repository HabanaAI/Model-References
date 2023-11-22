###############################################################################
# Copyright (c) 2022, Habana Labs Ltd.  All rights reserved.
###############################################################################

from synapse_profiler_api import SynapseProfilerApi, TraceType

class SynapseProfilerAdatper(object):
    def __init__(self, steps, node_rank):

        print(type(steps))
        print(steps)

        self._start_profiler = lambda : None
        self._stop_profiler = lambda : None
        self._dump_data = lambda : None
        self._training_step = 0

        self.start_step, self.stop_step = steps

        try:
            profiler = SynapseProfilerApi()
            self._start_profiler = lambda: profiler.profiler_start(TraceType.TraceAll, node_rank)
            self._stop_profiler = lambda: profiler.profiler_stop(TraceType.TraceAll, node_rank)
            self._dump_data = lambda: profiler.profiler_get_trace_json(TraceType.TraceAll, node_rank)
            print(f"Setup synapse profiler for steps: {(self.start_step, self.stop_step)}")
        except:
            print("Failed to setup synapse profiler")

    def start(self):
        assert self._training_step == 0
        if self.start_step == self._training_step:
            print(f"Profiler start at step: {self._training_step}")
            self._start_profiler()

    def stop(self):
        print("Dumping profiling data...")
        self._dump_data()

    def step(self):
        if self._training_step == self.start_step - 1:
            print(f"Profiler start after step: { self._training_step}")
            self._start_profiler()
        if self._training_step == self.stop_step:
            print(f"Profiler stop after step: {self._training_step}")
            self._stop_profiler()
        self._training_step += 1

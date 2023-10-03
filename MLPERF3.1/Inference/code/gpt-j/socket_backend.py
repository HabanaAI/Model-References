###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import array
import subprocess
import threading
import os
import time
import itertools

import socket_utils
import mlperf_loadgen as lg
from dataset import Dataset
from backlog import Backlog


SOCKET_PATH = "/tmp/inference.socket"
LOAD_TIME = time.time()
LONGEST_EVENT = 0
MAX_WORKER_BACKLOG = 2

# Buckets need to be arranged so that N-th worker is able to handle all jobs from workers > N
# This way the same bucket list works for any number of workers
BUCKETS = {
    'bfloat16_Offline': [
        [(1919, 12), (511, 12)],
        [(1663, 12), (511, 12)],
        [(1407, 12), (511, 12)],
        [(1151, 12), (511, 12)],
        [(1023, 12), (511, 12)],
        [(895, 12), (511, 12)],
        [(767, 12), (511, 12)],
        [(639, 24)]
    ],
    'bfloat16_Server': [
        [(1919, 12), (511, 12)],
        [(1663, 12), (511, 12)],
        [(1407, 12), (511, 12)],
        [(1151, 12), (511, 12)],
        [(1023, 12), (511, 12)],
        [(895, 12), (511, 12)],
        [(767, 12), (511, 12)],
        [(639, 12), (511, 12)]
    ],
    'float8_Offline': [
        [(1919, 32), (511, 32)],
        [(1663, 32), (511, 32)],
        [(1407, 32), (511, 32)],
        [(1151, 32), (511, 32)],
        [(1023, 32), (511, 32)],
        [(895, 64)],
        [(767, 64)],
        [(639, 64)]
    ],
    'float8_Server': [
        [(1919, 32)],
        [(1663, 32)],
        [(1407, 32)],
        [(1151, 32), (511, 32)],
        [(1023, 32), (511, 32)],
        [(895, 32), (511, 32)],
        [(767, 32), (511, 32)],
        [(639, 32), (511, 32)]
    ]
}


def get_backlog_buckets(scenario):
    return list(sorted(set(itertools.chain(*BUCKETS[scenario]))))


def start_workers(args):
    cmd = []
    if args.num_workers > 1:
        cores_per_worker = os.cpu_count() // args.num_workers // 2
        cmd += ['mpirun',
                '--allow-run-as-root',
                '--bind-to core',
                f'--map-by socket:PE={cores_per_worker}',
                f'--np {args.num_workers}']

    cmd += ['python3',
            './socket_worker.py',
            f'--socket={SOCKET_PATH}',
            f'--model-path={args.model_path}',
            f'--dtype={args.dtype}',
            f'--dataset-path={args.dataset_path}',
            f'--max_examples={args.max_examples}',
            f'--options={args.options}']
    if args.fake_device:
        cmd.append(f'--fake_device')
    if args.fake_dataset:
        cmd.append('--fake_dataset')
    if args.quantization_file:
        cmd.append(f'--quantization_file={args.quantization_file}')
    log('START', cmd)
    return subprocess.Popen(" ".join(cmd), shell=True)


def log(event, *args, log_file=None):
    global LONGEST_EVENT
    LONGEST_EVENT = max(LONGEST_EVENT, len(event))
    t = time.time() - LOAD_TIME
    print(f'{t:10.3f} | {event.ljust(LONGEST_EVENT)} |', *args, flush=True, file=log_file)


class Worker():
    def __init__(self, socket, worker_id, scenario, log_file):
        self.log_file = log_file
        self.worker_id = worker_id
        self.dst, client_address = socket.accept()
        self.buckets = BUCKETS[scenario][worker_id]
        self.idle_time = 0

    def start(self, clbk):
        def thread_main():
            try:
                while True:
                    self.wait(clbk)
            except Exception as e:
                log("EXCEPTION:", e, log_file=self.log_file)
                os._exit(1)
        self.thread = threading.Thread(target=thread_main, daemon=True)
        self.thread.start()

    def wait(self, clbk=None, send_response=True):
        output = socket_utils.receive(self.dst)
        if send_response:
            self.send_response(output)
        if clbk is not None:
            clbk(self.worker_id)

    def log(self, event, *args, log_file=None):
        log(f'{event}({self.worker_id})', *args, log_file=log_file)

    def enqueue(self, batch, max_input_length, batch_size):
        self.log('ENQUEUE', f'bs:{len(batch)}', f'target_bs:{batch_size}',
                 f'max_input_length:{max_input_length}',
                 f'ids:{[elem[0][0] for elem in batch]}',
                 f'input_lengths:{[elem[0][2] for elem in batch if len(elem[0]) > 2]}',
                 log_file=self.log_file)
        socket_utils.send(
            self.dst, (batch, {'max_input_length': max_input_length}, batch_size))
        self.idle_time = time.time() + 0.002 * max_input_length # used only in Server

    def send_response(self, output):
        self.log('DONE', f'ids:{[elem[0] for elem in output]}', log_file=self.log_file)
        buffers = [(elem[0], array.array("B", elem[1])) for elem in output]
        lg.QuerySamplesComplete([lg.QuerySampleResponse(
            elem[0], *elem[1].buffer_info()) for elem in buffers])


class SUT_base():
    def __init__(self, args):
        self.log_file = open("backend_logs.txt", 'w') if not args.stdout else None
        self.dataset_path = args.dataset_path
        self.model_path = args.model_path
        self.num_workers = args.num_workers
        buckets_scenario = args.dtype + "_" + args.scenario
        self.backlog = Backlog(get_backlog_buckets(buckets_scenario), self.query_len)
        self.lock = threading.Lock()

        self.socket = socket_utils.listen(SOCKET_PATH)
        self.process = start_workers(args)
        self.num_scheduled = [0] * self.num_workers
        self.disable_multiple_buckets = os.environ.get('DISABLE_MULTIPLE_BUCKETS', False)

        self.data_object = Dataset(
            self.model_path, self.dataset_path, total_count_override=args.max_examples, add_padding=False, fake_data=args.fake_dataset)

        assert self.process.poll() is None, "Some of the processes exited early!"

        log('BUCKETS', BUCKETS[buckets_scenario], log_file=self.log_file)

        self.workers = [Worker(self.socket, worker_id, buckets_scenario, self.log_file)
                        for worker_id in range(self.num_workers)]
        self.warmup()
        self.start()

        self.qsl = lg.ConstructQSL(self.data_object.count, self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)

    def close_log_file(self):
        if self.log_file:
            self.log_file.close()

    def query_len(self, query):
        return query[2]

    def start(self):
        for worker in self.workers:
            worker.start(self.on_device_rdy)

    def warmup(self):
        log('WARMUP', 'START', log_file=self.log_file)
        # compilation takes more than 1 iter for fp8
        for _ in range(3):
            for worker in self.workers:
                for bucket in worker.buckets:
                    batch = [((-1, -1), time.time())] * bucket[1]
                    worker.enqueue(batch, bucket[0], bucket[1])
            for worker in self.workers:
                for bucket in worker.buckets:
                    worker.wait(send_response=False)
        log('WARMUP', 'DONE', log_file=self.log_file)

    def enqueue(self, worker_id, batch, max_input_length, batch_size):
        self.num_scheduled[worker_id] += 1
        self.workers[worker_id].enqueue(batch, max_input_length, batch_size)

    def _get_smallest_possible_bucket(self, max_bucket_size, batch, worker):
        bucket_size = max_bucket_size
        # set the bucket length to the smallest possible based on the max length in batch
        if not self.disable_multiple_buckets:
            max_length_in_batch = max([elem[0][2] for elem in batch])
            for bucket in self.workers[worker].buckets:
                if bucket[0] > max_length_in_batch:
                    bucket_size = bucket[0]
        return bucket_size

    def reschedule(self):
        log('RESCHEDULE', f'worker_load:{self.num_scheduled}',
            f'backlog_size:{len(self.backlog)}', f'backlog_load:{self.backlog.get_load()}',
            log_file=self.log_file)

        idle_workers = self._get_idle_workers()

        for w in idle_workers:
            batch, batch_size, max_bucket_size = self._get_data(w)

            if len(batch) == 0:
                continue

            bucket_size = self._get_smallest_possible_bucket(max_bucket_size, batch, w)

            self.enqueue(w, batch, bucket_size, batch_size)

    def issue_queries(self, query_samples):
        new_queries = [(sample.id, sample.index, self.data_object.source_encoded_input_ids[sample.index].size(-1))
                       for sample in query_samples]
        with self.lock:
            self.backlog.add(new_queries)
            self.reschedule()

    def flush_queries(self):
        pass

    def on_device_rdy(self, worker_id):
        with self.lock:
            self.num_scheduled[worker_id] -= 1
            self.reschedule()


class SUT_Offline(SUT_base):
    def __init__(self, args):
        SUT_base.__init__(self, args)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def _get_data(self, worker):
        max_bucket = self.workers[worker].buckets[0]
        max_bucket_size = max_bucket[0]
        batch_size = max_bucket[1]
        return self.backlog.next_n(max_bucket_size, batch_size), batch_size, max_bucket_size

    def issue_queries(self, query_samples):
        log('QUERIES', f'ids:{[s.id for s in query_samples]}', log_file=self.log_file)
        SUT_base.issue_queries(self, query_samples)

    def _get_idle_workers(self):
        idle_workers = [[worker_id] * (MAX_WORKER_BACKLOG - worker_backlog)
                        for worker_id, worker_backlog in enumerate(self.num_scheduled) if worker_backlog < MAX_WORKER_BACKLOG]
        # We want to schedule on last worker first as it has the smallest bucket
        idle_workers.reverse()
        idle_workers = itertools.chain(*itertools.zip_longest(*idle_workers))
        return [w for w in idle_workers if w is not None]

class SUT_Server(SUT_base):
    def __init__(self, args):
        SUT_base.__init__(self, args)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.is_first_query = True

    def _get_idle_workers(self):
        if self.is_first_query:
            return []
        t = time.time()
        idle_workers = [[worker_id] * (MAX_WORKER_BACKLOG - worker_backlog)
                        for worker_id, worker_backlog in enumerate(self.num_scheduled) if worker_backlog < MAX_WORKER_BACKLOG and t > self.workers[worker_id].idle_time]
        # We want to schedule on last worker first as it has the smallest bucket
        idle_workers.reverse()
        idle_workers = itertools.chain(*itertools.zip_longest(*idle_workers))
        return [w for w in idle_workers if w is not None]

    def _get_data(self, worker):
        max_bucket = self.workers[worker].buckets[0]
        min_bucket = self.workers[worker].buckets[-1]

        max_bucket_size = max_bucket[0]
        min_bucket_size = min_bucket[0]

        batch_size_min_bucket = min_bucket[1]
        batch_size_max_bucket = max_bucket[1]

        available_queries_in_min_bucket = self.backlog.get_load()[0][1]

        wait_time_max_bucket = self.backlog.get_max_wait_time_from_bucket(max_bucket_size)
        wait_time_min_bucket = self.backlog.get_max_wait_time_from_bucket(min_bucket_size)
        # TODO: come up with algo that handles the case where some bucket waits for more than 20s
        small_bucket_has_priority = wait_time_min_bucket > wait_time_max_bucket
        is_full_batch_ready_in_small_bucket = available_queries_in_min_bucket >= batch_size_min_bucket

        if is_full_batch_ready_in_small_bucket and small_bucket_has_priority:
            batch_size = batch_size_min_bucket
            batch = self.backlog.next_n(min_bucket_size, batch_size)
        else:
            batch_size = batch_size_max_bucket
            batch = self.backlog.next_n(max_bucket_size, batch_size)

        return batch, batch_size, max_bucket_size

    def issue_queries(self, query_samples):
        if self.is_first_query:
            self.is_first_query = False
            t = time.time()
            for worker in range(self.num_workers):
                length, bs = self.workers[worker].buckets[0]
                delay = length / 640 # start delay
                self.workers[worker].idle_time = t + delay

        log('QUERIES', f'ids:[{query_samples[0].id}] lengths:[{self.data_object.source_encoded_input_ids[query_samples[0].index].size(-1)}]', log_file=self.log_file)
        SUT_base.issue_queries(self, query_samples)

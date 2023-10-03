###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import bisect
import itertools
import time


class Backlog:
    def __init__(self, buckets, key_fn):
        self.buckets = buckets
        self.todo = [[] for b in buckets]
        self.key_fn = key_fn

    def find_bucket(self, key):
        key_tuple = (key,0)
        return bisect.bisect_left(self.buckets, key_tuple)

    def add(self, queries):
        for q in sorted(queries, key=self.key_fn, reverse=True):
            self.todo[self.find_bucket(self.key_fn(q))].append((q, time.time()))

    def next(self, max_size):
        starting_bucket = self.find_bucket(max_size)
        for bidx in range(starting_bucket, -1, -1):
            while len(self.todo[bidx]) > 0:
                yield self.todo[bidx].pop(0)

    def next_n(self, max_size, n):
        return list(itertools.islice(self.next(max_size), n))

    def __len__(self):
        return sum(len(b) for b in self.todo)

    def get_load(self):
        return [(b[0], len(t)) for b, t in zip(self.buckets, self.todo)]

    def get_max_wait_time_from_bucket(self, bucket_size):
        bucket_idx = self.find_bucket(bucket_size)
        if len(self.todo[bucket_idx]) == 0:
            return 0.0
        return time.time() - self.todo[bucket_idx][0][-1]

if __name__ == '__main__':
    import random
    buckets = [256, 512, 768]
    queries = [(random.choice(['A', 'B', 'C']), random.randrange(buckets[-1])) for _ in range(16)]

    backlog = Backlog(buckets, lambda q: q[1])

    backlog.add(queries)
    print(backlog.todo)
    print(768, backlog.next_n(768, 3))
    print(256, backlog.next_n(256, 16))
    print(backlog.todo)

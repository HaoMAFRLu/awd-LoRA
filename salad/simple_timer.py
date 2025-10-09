import time
import torch
import torch.distributed as dist

class SimpleTimer():

    def __init__(self, name, sync_cuda=True):
        self.name = name
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self.total = 0.0
        self.count = 0
        self._t0 = None

    def __enter__(self):
        if self.sync_cuda: torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.sync_cuda: torch.cuda.synchronize()
        dt = time.perf_counter() - self._t0
        self.total += dt
        self.count += 1

    def avg(self):
        return self.total / max(1, self.count)

    def reset(self):
        self.total = 0.0
        self.count = 0
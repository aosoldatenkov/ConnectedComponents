"""Connected components of the sign regions {F >= 0} and {F < 0} on the grid."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from conncomp import _connected


def _as_numpy(vals):
    if isinstance(vals, torch.Tensor):
        vals = vals.detach().cpu().numpy()
    return np.ascontiguousarray(vals, dtype=np.float64)


def component_sizes(vals, pat, num_threads=-1):
    """Pixel counts of the sign components for each grid in the batch `vals` (shape (N, W, W))."""
    return _connected.components_batch(_as_numpy(vals), pat, num_threads)


def component_labels(vals, pat):
    """Labels of the sign components of a single grid `vals` (shape (W, W)).

    Returns (labels, sizes): labels[i, j] is the index of the component of pixel (i, j) in `sizes`.
    """
    return _connected.component_labels(_as_numpy(vals), pat)


def sign_array(vals):
    """uint8 sign classes (1 where vals >= 0) as a contiguous numpy array; computed on the device of `vals`."""
    if isinstance(vals, torch.Tensor):
        return (vals >= 0).to(torch.uint8).cpu().numpy()
    return np.ascontiguousarray(np.asarray(vals) >= 0, dtype=np.uint8)


def count_components(vals, pat, min_size=1, num_threads=-1):
    """Number of sign components with at least `min_size` pixels, for each grid in the batch.

    `vals` are values (float, torch or numpy) or uint8 sign classes (0/1) of shape (N, W, W). Small
    components are discarded as discretization noise. For a curve of even degree in RP^2, the
    complement of k ovals has k + 1 components.
    """
    signs = vals if isinstance(vals, np.ndarray) and vals.dtype == np.uint8 else sign_array(vals)
    return _connected.count_components_batch(signs, pat, min_size, num_threads).tolist()


class CountPipeline:
    """Overlap the GPU work for the next batch with the CPU component count of the current one.

    `submit(signs, payload)` takes uint8 sign classes on the GPU (see Experiment.signs), starts a
    non-blocking copy into pinned host memory, and queues the count on a worker thread (the C++ code
    releases the GIL). `results(pending=1)` yields (counts, payload) for all but the last `pending`
    submitted batches, in order. With pending=1 the main thread prepares batch k + 1 while batch k is
    being counted. Use `results(0)` at the end.

        pipe = CountPipeline(exp.pat, min_size)
        for it in range(niter):
            coefs = sample(...)
            pipe.submit(exp.signs(coefs), coefs)
            for counts, coefs_done in pipe.results():
                ...
        for counts, coefs_done in pipe.results(0):
            ...
    """

    def __init__(self, pat, min_size=1, num_threads=-1):
        self.pat, self.min_size, self.num_threads = pat, min_size, num_threads
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = deque()

    def _count(self, host, event):
        if event is not None:
            event.synchronize()
        return _connected.count_components_batch(host.numpy(), self.pat, self.min_size, self.num_threads).tolist()

    def submit(self, signs, payload=None):
        if signs.is_cuda:
            host = torch.empty(signs.shape, dtype=torch.uint8, pin_memory=True)
            host.copy_(signs, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        else:
            host, event = signs.contiguous(), None
        self.pending.append((self.executor.submit(self._count, host, event), payload))

    def results(self, pending=1):
        while len(self.pending) > pending:
            fut, payload = self.pending.popleft()
            yield fut.result(), payload

    def close(self):
        self.executor.shutdown(wait=True)

"""
Multi-processing DataLoader with staging queue and batch queue.
Generic version — works with any dataset, not just Parquet.
Each worker creates its own dataset instance from a factory function.
"""
import multiprocessing as mp
import time
import json
import os

import torch

from minimal_dataset.sampler import LockFreeSampler


def _default_collate_fn(samples):
    images = torch.stack([s[0] for s in samples])
    labels = [s[1] for s in samples]
    if isinstance(labels[0], str):
        labels = torch.tensor([int(l.replace('n', '')) for l in labels])
    else:
        labels = torch.tensor(labels)
    return images, labels


def _worker_fn(worker_id, dataset_factory, indices, staging_queue, stop_event, metric_queue):
    """Worker process: creates its own dataset and loads samples."""
    dataset = dataset_factory()
    stage_times = {"io": [], "decode": [], "preprocess": [], "staging_put": [], "total_sample": []}

    for idx in indices:
        if stop_event.is_set():
            break

        t0 = time.perf_counter()
        sample = dataset[idx]
        t1 = time.perf_counter()

        io_time = getattr(dataset, '_last_io_time', 0)
        decode_time = getattr(dataset, '_last_decode_time', 0)
        preprocess_time = getattr(dataset, '_last_preprocess_time', 0)

        staging_queue.put(sample)
        t2 = time.perf_counter()

        stage_times["io"].append(io_time)
        stage_times["decode"].append(decode_time)
        stage_times["preprocess"].append(preprocess_time)
        stage_times["staging_put"].append(t2 - t1)
        stage_times["total_sample"].append(t1 - t0)

    metric_queue.put(stage_times)


def _orchestrator_fn(staging_queue, batch_queue, batch_size, stop_event, result_queue):
    buffer = []
    batches_produced = 0
    empty_events = 0
    wait_time = 0.0
    collate_times = []
    batch_put_times = []

    while not stop_event.is_set():
        try:
            t0 = time.perf_counter()
            sample = staging_queue.get(timeout=0.1)
            wait_time += time.perf_counter() - t0
            buffer.append(sample)
            if len(buffer) >= batch_size:
                batch_samples = buffer[:batch_size]
                buffer = buffer[batch_size:]

                t1 = time.perf_counter()
                batch = _default_collate_fn(batch_samples)
                t2 = time.perf_counter()
                batch_queue.put(batch)
                t3 = time.perf_counter()

                collate_times.append(t2 - t1)
                batch_put_times.append(t3 - t2)
                batches_produced += 1
        except:
            empty_events += 1
            continue

    while len(buffer) >= batch_size:
        batch = _default_collate_fn(buffer[:batch_size])
        buffer = buffer[batch_size:]
        batch_queue.put(batch)
        batches_produced += 1

    result_queue.put({
        "batches_produced": batches_produced,
        "empty_events": empty_events,
        "wait_time": wait_time,
        "collate_times": collate_times,
        "batch_put_times": batch_put_times,
    })


class DataLoaderMP:
    def __init__(self, dataset_factory, total_samples, batch_size,
                 num_workers=1, max_staging_size=256, max_batch_queue_size=8,
                 metrics_dir=None):
        self.dataset_factory = dataset_factory
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metrics_dir = metrics_dir

        self._ctx = mp.get_context('fork')
        self.staging_queue = self._ctx.Queue(maxsize=max_staging_size)
        self.batch_queue = self._ctx.Queue(maxsize=max_batch_queue_size)
        self.result_queue = self._ctx.Queue()
        self.metric_queue = self._ctx.Queue()

        self.sampler = LockFreeSampler(total_samples, num_workers, shuffle=True)
        self._stop_event = self._ctx.Event()
        self._workers = []
        self._orchestrator = None

    def __iter__(self):
        self._stop_event.clear()
        self._workers = []

        for w in range(self.num_workers):
            indices = self.sampler.get_partition(w)
            p = self._ctx.Process(
                target=_worker_fn,
                args=(w, self.dataset_factory, indices,
                      self.staging_queue, self._stop_event, self.metric_queue)
            )
            p.start()
            self._workers.append(p)

        self._orchestrator = self._ctx.Process(
            target=_orchestrator_fn,
            args=(self.staging_queue, self.batch_queue, self.batch_size,
                  self._stop_event, self.result_queue)
        )
        self._orchestrator.start()
        return self

    def __next__(self):
        procs = self._workers + ([self._orchestrator] if self._orchestrator else [])
        all_done = all(not p.is_alive() for p in procs)
        if self.batch_queue.empty() and all_done:
            self._cleanup()
            raise StopIteration
        try:
            return self.batch_queue.get(timeout=1.0)
        except:
            if self.batch_queue.empty() and all(not p.is_alive() for p in procs):
                self._cleanup()
                raise StopIteration
            return self.batch_queue.get()

    def _cleanup(self):
        self._stop_event.set()
        for p in self._workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)
        if self._orchestrator and self._orchestrator.is_alive():
            self._orchestrator.terminate()
            self._orchestrator.join(timeout=2.0)

        # Collect worker metrics
        worker_stage_times = {"io": [], "decode": [], "preprocess": [], "staging_put": [], "total_sample": []}
        while not self.metric_queue.empty():
            try:
                stage_times = self.metric_queue.get_nowait()
                for key in worker_stage_times:
                    worker_stage_times[key].extend(stage_times.get(key, []))
            except:
                pass

        # Collect orchestrator stats
        try:
            stats = self.result_queue.get_nowait()
        except:
            stats = {"batches_produced": 0, "empty_events": 0, "wait_time": 0,
                     "collate_times": [], "batch_put_times": []}

        # Build summary
        import statistics
        summary = {
            "num_workers": self.num_workers,
            "batches_produced": stats.get("batches_produced", 0),
            "stage_times": {}
        }
        for key in ["io", "decode", "preprocess", "staging_put", "total_sample"]:
            times = worker_stage_times[key]
            if times:
                summary["stage_times"][key] = {
                    "count": len(times),
                    "mean_ms": round(statistics.mean(times) * 1000, 3),
                    "std_ms": round(statistics.stdev(times) * 1000, 3) if len(times) > 1 else 0,
                }
            else:
                summary["stage_times"][key] = {"count": 0, "mean_ms": 0, "std_ms": 0}

        collate_times = stats.get("collate_times", [])
        batch_put_times = stats.get("batch_put_times", [])
        summary["stage_times"]["collate"] = {
            "count": len(collate_times),
            "mean_ms": round(statistics.mean(collate_times) * 1000, 3) if collate_times else 0,
            "std_ms": round(statistics.stdev(collate_times) * 1000, 3) if len(collate_times) > 1 else 0,
        }
        summary["stage_times"]["batch_put"] = {
            "count": len(batch_put_times),
            "mean_ms": round(statistics.mean(batch_put_times) * 1000, 3) if batch_put_times else 0,
            "std_ms": round(statistics.stdev(batch_put_times) * 1000, 3) if len(batch_put_times) > 1 else 0,
        }

        if self.metrics_dir:
            os.makedirs(self.metrics_dir, exist_ok=True)
            path = os.path.join(self.metrics_dir,
                                f"metrics_mp_w{self.num_workers}_bs{self.batch_size}.json")
            with open(path, 'w') as f:
                json.dump(summary, f, indent=2)

        return summary

    def set_epoch(self, seed=None):
        self.sampler.reshuffle(seed)

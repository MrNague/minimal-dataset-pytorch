"""
Multi-threaded DataLoader with staging queue and batch queue.
"""
import threading
import time
import json
import os
import statistics
from typing import Optional, Callable

import torch

from .sampler import LockFreeSampler
from .monitored_queue import MonitoredQueue
from .metrics import MetricsTracker


class DataLoader:
    def __init__(self, dataset, batch_size: int,
                 num_workers: int = 1,
                 max_staging_size: int = 256,
                 max_batch_queue_size: int = 8,
                 collate_fn: Optional[Callable] = None,
                 metrics_dir: Optional[str] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metrics_dir = metrics_dir
        self.staging_queue = MonitoredQueue(maxsize=max_staging_size, name="staging")
        self.batch_queue = MonitoredQueue(maxsize=max_batch_queue_size, name="batch")
        self.collate_fn = collate_fn or self._default_collate
        self.sampler = LockFreeSampler(len(dataset), num_workers, shuffle=True)
        self._stop_event = threading.Event()
        self._threads = []
        self._workers_done = 0
        self._tracker = MetricsTracker(num_workers)

        self._stage_times = {
            "io": [],
            "decode": [],
            "preprocess": [],
            "staging_put": [],
            "collate": [],
            "batch_put": [],
            "total_sample": [],
        }
        self._stage_lock = threading.Lock()

    def _worker(self, worker_id: int):
        indices = self.sampler.get_partition(worker_id)
        wm = self._tracker.get_worker(worker_id)
        for idx in indices:
            if self._stop_event.is_set():
                break
            wm.start_sample()
            t0 = time.perf_counter()
            sample = self.dataset[idx]
            t1 = time.perf_counter()
            wm.end_sample()
            io_time = getattr(self.dataset, '_last_io_time', 0)
            decode_time = getattr(self.dataset, '_last_decode_time', 0)
            preprocess_time = getattr(self.dataset, '_last_preprocess_time', 0)
            t2 = time.perf_counter()
            self.staging_queue.put(sample)
            t3 = time.perf_counter()
            with self._stage_lock:
                self._stage_times["io"].append(io_time)
                self._stage_times["decode"].append(decode_time)
                self._stage_times["preprocess"].append(preprocess_time)
                self._stage_times["staging_put"].append(t3 - t2)
                self._stage_times["total_sample"].append(t1 - t0)
        self._workers_done += 1

    def _orchestrator(self):
        buffer = []
        while not self._stop_event.is_set():
            try:
                sample = self.staging_queue.get(timeout=0.1)
                buffer.append(sample)
                if len(buffer) >= self.batch_size:
                    batch_samples = buffer[:self.batch_size]
                    buffer = buffer[self.batch_size:]
                    t0 = time.perf_counter()
                    batch = self.collate_fn(batch_samples)
                    t1 = time.perf_counter()
                    self.batch_queue.put(batch)
                    t2 = time.perf_counter()
                    self._tracker.record_batch()
                    with self._stage_lock:
                        self._stage_times["collate"].append(t1 - t0)
                        self._stage_times["batch_put"].append(t2 - t1)
            except Exception:
                if self._workers_done >= self.num_workers and self.staging_queue.qsize() == 0:
                    break
        while len(buffer) >= self.batch_size:
            batch_samples = buffer[:self.batch_size]
            buffer = buffer[self.batch_size:]
            batch = self.collate_fn(batch_samples)
            self.batch_queue.put(batch)
            self._tracker.record_batch()

    def _default_collate(self, samples):
        images = torch.stack([s[0] for s in samples])
        labels = [s[1] for s in samples]
        if isinstance(labels[0], str):
            labels = torch.tensor([int(l.replace('n', '')) for l in labels])
        else:
            labels = torch.tensor(labels)
        return images, labels

    def __iter__(self):
        self._stop_event.clear()
        self._threads = []
        self._workers_done = 0
        self._tracker = MetricsTracker(self.num_workers)
        self._stage_times = {
            "io": [], "decode": [], "preprocess": [],
            "staging_put": [], "collate": [], "batch_put": [], "total_sample": []
        }
        for w in range(self.num_workers):
            t = threading.Thread(target=self._worker, args=(w,))
            t.start()
            self._threads.append(t)
        orch = threading.Thread(target=self._orchestrator)
        orch.start()
        self._threads.append(orch)
        return self

    def __next__(self):
        all_done = all(not t.is_alive() for t in self._threads)
        if self.batch_queue.empty() and all_done:
            self._cleanup()
            raise StopIteration
        try:
            return self.batch_queue.get(timeout=1.0)
        except Exception:
            if self.batch_queue.empty() and all(not t.is_alive() for t in self._threads):
                self._cleanup()
                raise StopIteration
            return self.batch_queue.get()

    def _cleanup(self):
        self._stop_event.set()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=2.0)
        summary = self._tracker.summary(
            self.staging_queue.stats(),
            self.batch_queue.stats()
        )
        with self._stage_lock:
            stage_stats = {}
            for stage, times in self._stage_times.items():
                if times:
                    stage_stats[stage] = {
                        "count": len(times),
                        "mean_ms": round(statistics.mean(times) * 1000, 3),
                        "std_ms": round(statistics.stdev(times) * 1000, 3) if len(times) > 1 else 0,
                        "total_s": round(sum(times), 3),
                    }
                else:
                    stage_stats[stage] = {"count": 0, "mean_ms": 0, "std_ms": 0, "total_s": 0}
        summary["stage_times"] = stage_stats
        if self.metrics_dir:
            os.makedirs(self.metrics_dir, exist_ok=True)
            path = os.path.join(
                self.metrics_dir,
                f"metrics_w{self.num_workers}_bs{self.batch_size}.json"
            )
            with open(path, 'w') as f:
                json.dump(summary, f, indent=2)
        return summary

    def set_epoch(self, seed: int = None):
        self.sampler.reshuffle(seed)
        self.staging_queue.reset()
        self.batch_queue.reset()
        self._tracker.reset()

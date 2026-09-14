#!/usr/bin/env python3

import sys
import time
import math
import gc
import queue
import threading
import resource
import statistics

import torch
import torch.nn.functional as F
import torchvision.io as tvio

sys.path.insert(0, "/home/nague/bachelor-project")

from minimal_dataset import ParquetDataset, DataLoader


PARQUET = "/fscratch/nague/storage_benchmarks/images.parquet"

SAMPLES = 9984
BATCH_SIZE = 256

WORKERS = [16, 32]
CHUNKS = [16, 32]
CAPACITIES = [256, 512, 1024]
BACKENDS = ["monitored", "plain"]

REPEATS = 2


torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def cpu_time():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


# ============================================================
# Plain queue wrapper
# ============================================================

class PlainQueue:

    def __init__(self, maxsize=0):
        self._queue = queue.Queue(maxsize=maxsize)
        self.maxsize = maxsize

    def put(self, item, timeout=None):
        if timeout is None:
            self._queue.put(item)
        else:
            self._queue.put(
                item,
                timeout=timeout
            )

    def get(self, timeout=None):
        if timeout is None:
            return self._queue.get()

        return self._queue.get(
            timeout=timeout
        )

    def qsize(self):
        return self._queue.qsize()

    def empty(self):
        return self._queue.empty()

    def full(self):
        return self._queue.full()

    def stats(self):
        return {
            "name": "plain",
            "maxsize": self.maxsize,
            "current_size": self._queue.qsize(),
        }

    def reset(self):
        pass


# ============================================================
# Dataset
# ============================================================

print("Loading Parquet table once...", flush=True)

base = ParquetDataset(
    PARQUET,
    max_samples=SAMPLES
)

table = base._table

print("Table ready.", flush=True)


class UInt8Dataset:

    def __len__(self):
        return SAMPLES

    def __getitem__(self, index):

        row = table.slice(
            index,
            1
        ).to_pylist()[0]

        encoded = torch.frombuffer(
            bytearray(row["image"]),
            dtype=torch.uint8
        )

        img = tvio.decode_jpeg(
            encoded,
            mode=tvio.ImageReadMode.RGB
        )

        if img.shape[-2:] != (64, 64):

            img = F.interpolate(
                img.unsqueeze(0),
                size=(64, 64),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

        return img, row["label"]


dataset = UInt8Dataset()


def collate(samples):

    images = torch.stack([
        x[0]
        for x in samples
    ])

    labels = [
        x[1]
        for x in samples
    ]

    if isinstance(labels[0], str):
        labels = torch.tensor([
            int(x.replace("n", ""))
            for x in labels
        ])
    else:
        labels = torch.tensor(labels)

    return images, labels


# ============================================================
# Chunked loader
# ============================================================

class ChunkedLoader(DataLoader):

    def __init__(
        self,
        *args,
        chunk_size=16,
        queue_backend="monitored",
        **kwargs
    ):

        self.chunk_size = chunk_size
        self.queue_backend = queue_backend

        super().__init__(
            *args,
            **kwargs
        )

        # Replace ONLY staging queue for plain test
        if queue_backend == "plain":

            maxsize = self.staging_queue.maxsize

            self.staging_queue = PlainQueue(
                maxsize=maxsize
            )

        self.worker_put_times = [
            []
            for _ in range(self.num_workers)
        ]


    def _worker(self, worker_id):

        indices = self.sampler.get_partition(
            worker_id
        )

        wm = self._tracker.get_worker(
            worker_id
        )

        chunk = []


        def flush():

            if not chunk:
                return

            payload = list(chunk)

            t0 = time.perf_counter()

            self.staging_queue.put(
                payload
            )

            elapsed = (
                time.perf_counter()
                - t0
            )

            self.worker_put_times[
                worker_id
            ].append(
                (
                    elapsed,
                    len(payload)
                )
            )

            chunk.clear()


        for idx in indices:

            if self._stop_event.is_set():
                break

            wm.start_sample()

            sample = self.dataset[idx]

            wm.end_sample()

            chunk.append(sample)

            if len(chunk) >= self.chunk_size:
                flush()


        flush()


        lock = getattr(
            self,
            "_workers_done_lock",
            None
        )

        if lock is not None:

            with lock:
                self._workers_done += 1

        else:

            self._workers_done += 1


    def _orchestrator(self):

        buffer = []

        while not self._stop_event.is_set():

            try:

                chunk = self.staging_queue.get(
                    timeout=0.1
                )

                buffer.extend(chunk)


                while len(buffer) >= self.batch_size:

                    samples = buffer[
                        :self.batch_size
                    ]

                    del buffer[
                        :self.batch_size
                    ]

                    batch = self.collate_fn(
                        samples
                    )

                    self.batch_queue.put(
                        batch
                    )

                    self._tracker.record_batch()


            except queue.Empty:

                lock = getattr(
                    self,
                    "_workers_done_lock",
                    None
                )

                if lock is not None:

                    with lock:
                        done = self._workers_done

                else:
                    done = self._workers_done


                if (
                    done >= self.num_workers
                    and self.staging_queue.qsize() == 0
                ):
                    break


            except Exception:

                lock = getattr(
                    self,
                    "_workers_done_lock",
                    None
                )

                if lock is not None:

                    with lock:
                        done = self._workers_done

                else:
                    done = self._workers_done


                if (
                    done >= self.num_workers
                    and self.staging_queue.qsize() == 0
                ):
                    break


# ============================================================
# Benchmark
# ============================================================

def benchmark(
    workers,
    chunk,
    sample_capacity,
    backend
):

    queue_capacity_chunks = max(
        1,
        math.ceil(
            sample_capacity
            / chunk
        )
    )


    loader = ChunkedLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=workers,
        max_staging_size=queue_capacity_chunks,
        collate_fn=collate,
        chunk_size=chunk,
        queue_backend=backend,
    )


    cpu_before = cpu_time()
    start = time.perf_counter()

    total = 0


    for images, labels in loader:

        total += images.shape[0]


    elapsed = (
        time.perf_counter()
        - start
    )

    cpu_after = cpu_time()


    throughput = (
        total / elapsed
    )

    cores = (
        cpu_after - cpu_before
    ) / elapsed


    records = []

    for worker_values in (
        loader.worker_put_times
    ):
        records.extend(
            worker_values
        )


    waits_ms = [
        duration * 1000
        for duration, _
        in records
    ]


    sample_wait_us = [
        duration
        / count
        * 1_000_000
        for duration, count
        in records
    ]


    mean_put_ms = (
        statistics.mean(waits_ms)
        if waits_ms
        else 0
    )


    mean_sample_us = (
        statistics.mean(
            sample_wait_us
        )
        if sample_wait_us
        else 0
    )


    num_puts = len(records)


    del loader
    gc.collect()


    return (
        throughput,
        cores,
        queue_capacity_chunks,
        num_puts,
        mean_put_ms,
        mean_sample_us,
    )


print()
print("=" * 120)
print("QUEUE BACKEND + CAPACITY TEST")
print("=" * 120)

print(
    "backend,workers,chunk,"
    "sample_capacity,repeat,"
    "throughput,cpu,"
    "queue_capacity_chunks,"
    "num_puts,"
    "mean_put_ms,"
    "put_us_per_sample",
    flush=True
)


for backend in BACKENDS:

    for workers in WORKERS:

        for chunk in CHUNKS:

            for capacity in CAPACITIES:

                results = []

                for repeat in range(
                    1,
                    REPEATS + 1
                ):

                    (
                        tp,
                        cores,
                        qcap,
                        num_puts,
                        mean_put,
                        sample_put
                    ) = benchmark(
                        workers,
                        chunk,
                        capacity,
                        backend
                    )

                    results.append(tp)

                    print(
                        f"{backend},"
                        f"{workers},"
                        f"{chunk},"
                        f"{capacity},"
                        f"{repeat},"
                        f"{tp:.1f},"
                        f"{cores:.2f},"
                        f"{qcap},"
                        f"{num_puts},"
                        f"{mean_put:.3f},"
                        f"{sample_put:.2f}",
                        flush=True
                    )


                print(
                    f"MEAN,"
                    f"backend={backend},"
                    f"workers={workers},"
                    f"chunk={chunk},"
                    f"capacity={capacity},"
                    f"throughput="
                    f"{statistics.mean(results):.1f}",
                    flush=True
                )


print("\nDONE", flush=True)

#!/usr/bin/env python3

import sys
import time
import math
import gc
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
CHUNK_SIZES = [1, 4, 8, 16]
REPEATS = 3

# Keep approximately the same maximum number of samples
# in the staging queue for every chunk size.
STAGING_SAMPLE_CAPACITY = 256


torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def cpu_time():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)

    idx = int(
        round(
            (len(values) - 1) * p
        )
    )

    return values[idx]


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


# ============================================================
# Collate
# ============================================================

def uint8_collate(samples):

    images = torch.stack([
        sample[0]
        for sample in samples
    ])

    labels = [
        sample[1]
        for sample in samples
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
# Chunked DataLoader
#
# Reuses the current DataLoader infrastructure, but changes:
#
# worker:
#     sample -> queue
#
# into:
#
# worker:
#     [sample, sample, ...] -> queue
# ============================================================

class ChunkedDataLoader(DataLoader):

    def __init__(
        self,
        *args,
        staging_chunk_size=1,
        **kwargs
    ):

        self.staging_chunk_size = (
            staging_chunk_size
        )

        super().__init__(
            *args,
            **kwargs
        )

        self._put_times_by_worker = [
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


        def send_chunk():

            if not chunk:
                return

            # Copy reference list because chunk is reused
            payload = list(chunk)

            t0 = time.perf_counter()

            self.staging_queue.put(
                payload
            )

            t1 = time.perf_counter()

            self._put_times_by_worker[
                worker_id
            ].append(
                (
                    t1 - t0,
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

            if (
                len(chunk)
                >= self.staging_chunk_size
            ):
                send_chunk()


        # Flush final incomplete chunk
        send_chunk()


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

                # Main difference:
                # one queue item can contain several samples.
                buffer.extend(chunk)


                while len(buffer) >= self.batch_size:

                    batch_samples = buffer[
                        :self.batch_size
                    ]

                    del buffer[
                        :self.batch_size
                    ]


                    t0 = time.perf_counter()

                    batch = self.collate_fn(
                        batch_samples
                    )

                    t1 = time.perf_counter()


                    self.batch_queue.put(
                        batch
                    )

                    t2 = time.perf_counter()


                    self._tracker.record_batch()


                    with self._stage_lock:

                        self._stage_times[
                            "collate"
                        ].append(
                            t1 - t0
                        )

                        self._stage_times[
                            "batch_put"
                        ].append(
                            t2 - t1
                        )


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
    chunk_size
):

    # IMPORTANT:
    # queue maxsize counts CHUNKS, not samples.
    #
    # Thus:
    # chunk=1  -> maxsize 256
    # chunk=4  -> maxsize 64
    # chunk=8  -> maxsize 32
    # chunk=16 -> maxsize 16
    #
    # Effective capacity remains about 256 samples.

    staging_capacity = max(
        1,
        math.ceil(
            STAGING_SAMPLE_CAPACITY
            / chunk_size
        )
    )


    loader = ChunkedDataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=workers,
        max_staging_size=staging_capacity,
        collate_fn=uint8_collate,
        staging_chunk_size=chunk_size,
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


    # Aggregate queue put timings
    records = []

    for worker_records in (
        loader._put_times_by_worker
    ):
        records.extend(
            worker_records
        )


    chunk_put_ms = [
        duration * 1000
        for duration, size
        in records
    ]


    # Amortized queue waiting cost per sample
    per_sample_us = [
        duration
        / size
        * 1_000_000
        for duration, size
        in records
    ]


    mean_chunk_put = (
        statistics.mean(
            chunk_put_ms
        )
        if chunk_put_ms
        else 0
    )

    p95_chunk_put = percentile(
        chunk_put_ms,
        0.95
    )

    mean_sample_put = (
        statistics.mean(
            per_sample_us
        )
        if per_sample_us
        else 0
    )


    collate_values = (
        loader._stage_times.get(
            "collate",
            []
        )
    )

    collate_ms = (
        statistics.mean(
            collate_values
        ) * 1000
        if collate_values
        else 0
    )


    num_puts = len(records)


    try:
        queue_stats = (
            loader.staging_queue.stats()
        )
    except Exception:
        queue_stats = {}


    del loader
    gc.collect()


    return {
        "throughput": throughput,
        "cpu": cores,
        "capacity": staging_capacity,
        "num_puts": num_puts,
        "mean_chunk_put_ms": mean_chunk_put,
        "p95_chunk_put_ms": p95_chunk_put,
        "mean_sample_put_us": mean_sample_put,
        "collate_ms": collate_ms,
        "queue_stats": queue_stats,
    }


# ============================================================
# Run
# ============================================================

print()
print("=" * 110)
print("STAGING QUEUE CHUNKING")
print("=" * 110)

print(
    "workers,chunk,repeat,"
    "throughput,cpu,"
    "queue_capacity_chunks,"
    "num_queue_puts,"
    "mean_chunk_put_ms,"
    "p95_chunk_put_ms,"
    "amortized_put_us_per_sample,"
    "collate_ms",
    flush=True
)


for workers in WORKERS:

    for chunk_size in CHUNK_SIZES:

        throughputs = []

        for repeat in range(
            1,
            REPEATS + 1
        ):

            r = benchmark(
                workers,
                chunk_size
            )

            throughputs.append(
                r["throughput"]
            )

            print(
                f"{workers},"
                f"{chunk_size},"
                f"{repeat},"
                f"{r['throughput']:.1f},"
                f"{r['cpu']:.2f},"
                f"{r['capacity']},"
                f"{r['num_puts']},"
                f"{r['mean_chunk_put_ms']:.3f},"
                f"{r['p95_chunk_put_ms']:.3f},"
                f"{r['mean_sample_put_us']:.2f},"
                f"{r['collate_ms']:.3f}",
                flush=True
            )


        print(
            f"MEAN,"
            f"workers={workers},"
            f"chunk={chunk_size},"
            f"throughput="
            f"{statistics.mean(throughputs):.1f}",
            flush=True
        )


print()
print("DONE", flush=True)

"""
Multi-threaded DataLoader with chunked staging.

Design:
- Worker threads load samples from the dataset.
- Each worker groups samples into small chunks before sending them
  to the staging queue.
- One orchestrator assembles complete batches.
- A batch queue exposes ready batches to the consumer.

The chunked staging design reduces queue contention at high worker counts.
"""

import math
import os
import json
import queue
import statistics
import threading
import time
from typing import Optional, Callable

import torch

from .sampler import LockFreeSampler
from .monitored_queue import MonitoredQueue
from .metrics import MetricsTracker


class DataLoader:

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_workers: int = 1,
        num_orchestrators: int = 1,
        max_staging_size: int = 512,
        max_batch_queue_size: int = 8,
        collate_fn: Optional[Callable] = None,
        metrics_dir: Optional[str] = None,
        staging_chunk_size: int = 32,
    ):

        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if num_workers <= 0:
            raise ValueError("num_workers must be > 0")

        if staging_chunk_size <= 0:
            raise ValueError(
                "staging_chunk_size must be > 0"
            )

        # Experiments showed that multiple orchestrators hurt
        # throughput and can split samples across independent buffers.
        if num_orchestrators != 1:
            raise ValueError(
                "This DataLoader supports exactly one orchestrator. "
                "Benchmarks showed multiple orchestrators reduce throughput."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_orchestrators = 1

        self.metrics_dir = metrics_dir

        self.staging_chunk_size = staging_chunk_size

        # max_staging_size is expressed in approximate number
        # of SAMPLES, not number of chunks.
        self.max_staging_size = max_staging_size

        staging_queue_chunks = max(
            1,
            math.ceil(
                max_staging_size
                / staging_chunk_size
            )
        )

        self.staging_queue = MonitoredQueue(
            maxsize=staging_queue_chunks,
            name="staging",
        )

        self.batch_queue = MonitoredQueue(
            maxsize=max_batch_queue_size,
            name="batch",
        )

        self.collate_fn = (
            collate_fn
            or self._default_collate
        )

        self.sampler = LockFreeSampler(
            len(dataset),
            num_workers,
            shuffle=True,
        )

        self._stop_event = threading.Event()

        self._threads = []

        self._workers_done = 0
        self._workers_done_lock = threading.Lock()

        self._tracker = MetricsTracker(
            num_workers
        )

        self._stage_lock = threading.Lock()

        self._stage_times = self._new_stage_times()


    # ============================================================
    # Metrics
    # ============================================================

    @staticmethod
    def _new_stage_times():

        return {
            "total_sample": [],
            "staging_put": [],
            "staging_put_chunk": [],
            "collate": [],
            "batch_put": [],
        }


    # ============================================================
    # Worker
    # ============================================================

    def _worker(
        self,
        worker_id: int
    ):

        indices = self.sampler.get_partition(
            worker_id
        )

        worker_metrics = (
            self._tracker.get_worker(
                worker_id
            )
        )

        chunk = []

        # Keep metrics local to this worker.
        #
        # This avoids acquiring _stage_lock for every sample.
        local_sample_times = []
        local_staging_sample_times = []
        local_staging_chunk_times = []


        def flush_chunk():

            nonlocal chunk

            if not chunk:
                return

            payload = chunk

            # Important:
            # create a NEW list for subsequent samples instead
            # of clearing payload after it has entered the queue.
            chunk = []

            start_put = time.perf_counter()

            self.staging_queue.put(
                payload
            )

            put_time = (
                time.perf_counter()
                - start_put
            )

            local_staging_chunk_times.append(
                put_time
            )

            # Amortized queue wait per sample.
            local_staging_sample_times.extend(
                [put_time / len(payload)]
                * len(payload)
            )


        for idx in indices:

            if self._stop_event.is_set():
                break

            worker_metrics.start_sample()

            sample_start = time.perf_counter()

            sample = self.dataset[idx]

            sample_time = (
                time.perf_counter()
                - sample_start
            )

            worker_metrics.end_sample()

            local_sample_times.append(
                sample_time
            )

            chunk.append(sample)

            if (
                len(chunk)
                >= self.staging_chunk_size
            ):
                flush_chunk()


        # Last incomplete chunk
        flush_chunk()


        # Merge local metrics only once per worker.
        with self._stage_lock:

            self._stage_times[
                "total_sample"
            ].extend(
                local_sample_times
            )

            self._stage_times[
                "staging_put"
            ].extend(
                local_staging_sample_times
            )

            self._stage_times[
                "staging_put_chunk"
            ].extend(
                local_staging_chunk_times
            )


        with self._workers_done_lock:
            self._workers_done += 1


    # ============================================================
    # Orchestrator
    # ============================================================

    def _orchestrator(self):

        buffer = []


        while not self._stop_event.is_set():

            try:

                chunk = self.staging_queue.get(
                    timeout=0.1
                )

            except queue.Empty:

                with self._workers_done_lock:
                    done = self._workers_done

                if (
                    done >= self.num_workers
                    and self.staging_queue.empty()
                ):
                    break

                continue


            # One queue item now contains several samples.
            buffer.extend(chunk)


            while (
                len(buffer)
                >= self.batch_size
            ):

                batch_samples = buffer[
                    :self.batch_size
                ]

                del buffer[
                    :self.batch_size
                ]


                collate_start = (
                    time.perf_counter()
                )

                batch = self.collate_fn(
                    batch_samples
                )

                collate_time = (
                    time.perf_counter()
                    - collate_start
                )


                batch_put_start = (
                    time.perf_counter()
                )

                self.batch_queue.put(
                    batch
                )

                batch_put_time = (
                    time.perf_counter()
                    - batch_put_start
                )


                self._tracker.record_batch()


                with self._stage_lock:

                    self._stage_times[
                        "collate"
                    ].append(
                        collate_time
                    )

                    self._stage_times[
                        "batch_put"
                    ].append(
                        batch_put_time
                    )


        # Preserve the behavior of the previous implementation:
        # incomplete final batches are dropped.
        while (
            len(buffer)
            >= self.batch_size
        ):

            batch_samples = buffer[
                :self.batch_size
            ]

            del buffer[
                :self.batch_size
            ]

            collate_start = (
                time.perf_counter()
            )

            batch = self.collate_fn(
                batch_samples
            )

            collate_time = (
                time.perf_counter()
                - collate_start
            )


            batch_put_start = (
                time.perf_counter()
            )

            self.batch_queue.put(
                batch
            )

            batch_put_time = (
                time.perf_counter()
                - batch_put_start
            )


            self._tracker.record_batch()


            with self._stage_lock:

                self._stage_times[
                    "collate"
                ].append(
                    collate_time
                )

                self._stage_times[
                    "batch_put"
                ].append(
                    batch_put_time
                )


    # ============================================================
    # Collate
    # ============================================================

    def _default_collate(
        self,
        samples
    ):

        images = torch.stack([
            sample[0]
            for sample in samples
        ])

        # Optimized ParquetDataset returns resized uint8 tensors.
        # Convert/normalize once per batch instead of once per sample.
        if images.dtype == torch.uint8:
            images = images.float()
            images.div_(255.0)

        labels = [
            sample[1]
            for sample in samples
        ]

        if isinstance(
            labels[0],
            str
        ):

            labels = torch.tensor([
                int(
                    label.replace(
                        "n",
                        ""
                    )
                )
                for label in labels
            ])

        else:

            labels = torch.tensor(
                labels
            )

        return images, labels


    # ============================================================
    # Iterator
    # ============================================================

    def __iter__(self):

        self._stop_event.clear()

        self._threads = []

        self._workers_done = 0

        self._tracker = MetricsTracker(
            self.num_workers
        )

        self._stage_times = (
            self._new_stage_times()
        )


        # Worker threads
        for worker_id in range(
            self.num_workers
        ):

            thread = threading.Thread(
                target=self._worker,
                args=(worker_id,),
                name=f"dataloader-worker-{worker_id}",
            )

            thread.start()

            self._threads.append(
                thread
            )


        # Exactly one orchestrator.
        orchestrator = threading.Thread(
            target=self._orchestrator,
            name="dataloader-orchestrator",
        )

        orchestrator.start()

        self._threads.append(
            orchestrator
        )

        return self


    def __next__(self):

        all_done = all(
            not thread.is_alive()
            for thread in self._threads
        )

        if (
            self.batch_queue.empty()
            and all_done
        ):

            self._cleanup()

            raise StopIteration


        try:

            return self.batch_queue.get(
                timeout=1.0
            )

        except queue.Empty:

            all_done = all(
                not thread.is_alive()
                for thread in self._threads
            )

            if (
                self.batch_queue.empty()
                and all_done
            ):

                self._cleanup()

                raise StopIteration

            return self.batch_queue.get()


    # ============================================================
    # Cleanup / summary
    # ============================================================

    def _cleanup(self):

        self._stop_event.set()


        for thread in self._threads:

            if thread.is_alive():

                thread.join(
                    timeout=2.0
                )


        summary = self._tracker.summary(
            self.staging_queue.stats(),
            self.batch_queue.stats(),
        )


        with self._stage_lock:

            stage_stats = {}

            for (
                stage,
                times
            ) in self._stage_times.items():

                if times:

                    stage_stats[stage] = {
                        "count": len(times),

                        "mean_ms": round(
                            statistics.mean(
                                times
                            ) * 1000,
                            3,
                        ),

                        "std_ms": round(
                            statistics.stdev(
                                times
                            ) * 1000,
                            3,
                        )
                        if len(times) > 1
                        else 0,

                        "total_s": round(
                            sum(times),
                            3,
                        ),
                    }

                else:

                    stage_stats[stage] = {
                        "count": 0,
                        "mean_ms": 0,
                        "std_ms": 0,
                        "total_s": 0,
                    }


        summary[
            "stage_times"
        ] = stage_stats

        summary[
            "staging_chunk_size"
        ] = self.staging_chunk_size

        summary[
            "staging_sample_capacity"
        ] = self.max_staging_size

        summary[
            "staging_queue_chunk_capacity"
        ] = self.staging_queue.maxsize


        if self.metrics_dir:

            os.makedirs(
                self.metrics_dir,
                exist_ok=True
            )

            path = os.path.join(
                self.metrics_dir,
                (
                    f"metrics_"
                    f"w{self.num_workers}_"
                    f"bs{self.batch_size}_"
                    f"chunk{self.staging_chunk_size}.json"
                ),
            )

            with open(
                path,
                "w"
            ) as file:

                json.dump(
                    summary,
                    file,
                    indent=2,
                )


        return summary


    # ============================================================
    # Epoch reset
    # ============================================================

    def set_epoch(
        self,
        seed: int = None
    ):

        self.sampler.reshuffle(
            seed
        )

        self.staging_queue.reset()

        self.batch_queue.reset()

        self._tracker.reset()

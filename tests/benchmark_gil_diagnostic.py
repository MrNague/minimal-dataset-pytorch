#!/usr/bin/env python3

import os
import sys
import gc
import csv
import time
import inspect
import argparse
import resource

import torch

sys.path.insert(0, "/home/nague/bachelor-project")

from minimal_dataset import DataLoader, ParquetDataset


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BATCH_SIZE = 256
WORKERS = [1, 2, 4, 8, 16, 32]


# Prevent PyTorch itself from using many CPU threads internally.
torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ------------------------------------------------------------
# Synthetic datasets
# ------------------------------------------------------------

class NoopDataset:
    """
    Almost no work in __getitem__.
    This stresses DataLoader/queues/collation rather than the dataset.
    """
    def __init__(self, length):
        self.length = length
        self.image = torch.zeros((3, 64, 64), dtype=torch.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.image, index % 200


class SleepDataset:
    """
    Simulates work that does not keep the Python interpreter busy.
    Useful as a non-GIL-bound control.
    """
    def __init__(self, length, sleep_s=0.001):
        self.length = length
        self.sleep_s = sleep_s
        self.image = torch.zeros((3, 64, 64), dtype=torch.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        time.sleep(self.sleep_s)
        return self.image, index % 200


class PythonDataset:
    """
    Deliberately performs pure Python CPU work.
    This is our positive GIL-contention control.
    """
    def __init__(self, length, iterations):
        self.length = length
        self.iterations = iterations
        self.image = torch.zeros((3, 64, 64), dtype=torch.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = index + 1

        for _ in range(self.iterations):
            x = ((x * 1664525) + 1013904223) & 0xFFFFFFFF

        return self.image, index % 200


# ------------------------------------------------------------
# Calibration
# ------------------------------------------------------------

def calibrate_python_iterations(target_s=0.001):
    """
    Estimate how many pure-Python iterations take ~target_s
    with one thread.
    """
    n = 5000
    x = 1

    start = time.perf_counter()

    for _ in range(n):
        x = ((x * 1664525) + 1013904223) & 0xFFFFFFFF

    elapsed = time.perf_counter() - start

    if elapsed <= 0:
        return n

    estimate = int(n * target_s / elapsed)

    return max(100, estimate)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def cpu_time():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def create_loader(dataset, workers):
    kwargs = dict(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=workers,
    )

    # Compatible with both the older and newer DataLoader.
    signature = inspect.signature(DataLoader.__init__)

    if "num_orchestrators" in signature.parameters:
        kwargs["num_orchestrators"] = 1

    return DataLoader(**kwargs)


def get_stage_metric(summary, stage, metric="mean_ms"):
    try:
        return summary["stage_times"][stage][metric]
    except Exception:
        return float("nan")


def get_queue_metric(summary, queue_name, metric):
    try:
        return summary[queue_name][metric]
    except Exception:
        return ""


def run_once(mode, dataset, workers, phase):
    gc.collect()

    loader = create_loader(dataset, workers)

    cpu_before = cpu_time()
    start = time.perf_counter()

    samples = 0

    for batch in loader:
        images, labels = batch
        samples += images.shape[0]

    elapsed = time.perf_counter() - start
    cpu_after = cpu_time()

    # _cleanup() only creates the summary; calling it here lets us inspect it.
    summary = loader._cleanup()

    cpu_seconds = cpu_after - cpu_before

    # Equivalent number of fully busy CPU cores.
    cpu_core_equiv = cpu_seconds / elapsed if elapsed > 0 else 0

    throughput = samples / elapsed if elapsed > 0 else 0

    result = {
        "phase": phase,
        "mode": mode,
        "workers": workers,
        "switch_ms": round(sys.getswitchinterval() * 1000, 3),
        "samples": samples,
        "elapsed_s": round(elapsed, 4),
        "throughput_sps": round(throughput, 2),
        "cpu_core_equiv": round(cpu_core_equiv, 3),
        "collate_ms": get_stage_metric(summary, "collate"),
        "staging_put_ms": get_stage_metric(summary, "staging_put"),
        "batch_put_ms": get_stage_metric(summary, "batch_put"),
        "staging_full": get_queue_metric(
            summary, "staging_queue", "full_events"
        ),
        "batch_full": get_queue_metric(
            summary, "batch_queue", "full_events"
        ),
    }

    print(
        f"{phase:12s} "
        f"{mode:10s} "
        f"w={workers:2d} "
        f"switch={result['switch_ms']:6.2f}ms "
        f"throughput={result['throughput_sps']:9.1f} "
        f"cpu_cores={result['cpu_core_equiv']:6.2f} "
        f"collate={result['collate_ms']} "
        f"staging_put={result['staging_put_ms']} "
        f"staging_full={result['staging_full']}",
        flush=True,
    )

    return result


# ------------------------------------------------------------
# Main benchmark
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--parquet",
        default="/fscratch/nague/storage_benchmarks/images.parquet",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=9984,
        help="9984 = 39 full batches of 256",
    )

    parser.add_argument(
        "--output",
        default="/netscratch/nague/gil_diagnostic.csv",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("GIL / CONTENTION DIAGNOSTIC")
    print("=" * 80)

    print(f"Python: {sys.version}")
    print(f"PID: {os.getpid()}")
    print(f"Samples: {args.samples}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Workers: {WORKERS}")
    print(f"Default switch interval: {sys.getswitchinterval()*1000:.3f} ms")

    allocated_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "unknown")
    print(f"SLURM_CPUS_PER_TASK: {allocated_cpus}")

    python_iterations = calibrate_python_iterations(0.001)

    print(
        f"Pure-Python calibration: "
        f"{python_iterations} iterations ~= 1 ms single-thread"
    )

    print("=" * 80)

    results = []

    # --------------------------------------------------------
    # Normal scaling curves
    # --------------------------------------------------------

    datasets = {
        "noop": NoopDataset(args.samples),

        "sleep": SleepDataset(
            args.samples,
            sleep_s=0.001,
        ),

        "python": PythonDataset(
            args.samples,
            iterations=python_iterations,
        ),

        "real": ParquetDataset(
            args.parquet,
            max_samples=args.samples,
        ),
    }

    default_switch = sys.getswitchinterval()

    for mode in ["noop", "sleep", "python", "real"]:

        print()
        print("-" * 80)
        print(f"MODE: {mode}")
        print("-" * 80)

        dataset = datasets[mode]

        for workers in WORKERS:
            result = run_once(
                mode=mode,
                dataset=dataset,
                workers=workers,
                phase="scaling",
            )

            results.append(result)

    # --------------------------------------------------------
    # GIL switch-interval sensitivity
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("GIL SWITCH-INTERVAL TEST @ 32 WORKERS")
    print("=" * 80)

    switch_intervals = [
        0.001,   # 1 ms
        0.005,   # 5 ms
        0.020,   # 20 ms
    ]

    for interval in switch_intervals:

        sys.setswitchinterval(interval)

        print()
        print(
            f"Testing switch interval = "
            f"{interval * 1000:.1f} ms"
        )

        # Positive GIL control
        results.append(
            run_once(
                mode="python",
                dataset=datasets["python"],
                workers=32,
                phase="switch_test",
            )
        )

        # Real pipeline
        results.append(
            run_once(
                mode="real",
                dataset=datasets["real"],
                workers=32,
                phase="switch_test",
            )
        )

    sys.setswitchinterval(default_switch)

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(args.output),
        exist_ok=True,
    )

    fields = [
        "phase",
        "mode",
        "workers",
        "switch_ms",
        "samples",
        "elapsed_s",
        "throughput_sps",
        "cpu_core_equiv",
        "collate_ms",
        "staging_put_ms",
        "batch_put_ms",
        "staging_full",
        "batch_full",
    ]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 80)
    print(f"CSV saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import sys
import time
import resource

import torch
import torch.nn.functional as F
import torchvision.io as tvio

sys.path.insert(0, "/home/nague/bachelor-project")

from minimal_dataset import ParquetDataset, DataLoader


PARQUET = "/fscratch/nague/storage_benchmarks/images.parquet"

SAMPLES = 9984
BATCH_SIZE = 256
WORKERS = [8, 16, 32]


torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def cpu_time():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


# ============================================================
# Load table ONCE
# ============================================================

print("Loading Parquet table once...", flush=True)

base = ParquetDataset(
    PARQUET,
    max_samples=SAMPLES
)

table = base._table

print("Table ready.", flush=True)


# ============================================================
# Dataset
# ============================================================

class TVDataset:

    def __init__(self, mode):
        self.mode = mode

    def __len__(self):
        return SAMPLES

    def __getitem__(self, index):

        row = table.slice(index, 1).to_pylist()[0]

        img_bytes = row["image"]
        label = row["label"]

        encoded = torch.frombuffer(
            bytearray(img_bytes),
            dtype=torch.uint8
        )

        img = tvio.decode_jpeg(
            encoded,
            mode=tvio.ImageReadMode.RGB
        )

        # Resize BEFORE float conversion
        if img.shape[-2:] != (64, 64):

            img = F.interpolate(
                img.unsqueeze(0),
                size=(64, 64),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

        if self.mode == "two_alloc":

            # Allocation for float()
            # + allocation for division
            img = img.float() / 255.0

        elif self.mode == "one_alloc":

            # float allocation only;
            # division is in-place
            img = img.float()
            img.div_(255.0)

        elif self.mode == "batch":

            # Keep uint8.
            # Conversion will happen after stacking.
            pass

        return img, label


# ============================================================
# Batch normalization collate
# ============================================================

def batch_float_collate(samples):

    # uint8 batch: [B, C, H, W]
    images = torch.stack(
        [sample[0] for sample in samples]
    )

    # ONE float allocation for the whole batch
    images = images.float()

    # no second allocation
    images.div_(255.0)

    labels = [
        sample[1]
        for sample in samples
    ]

    if isinstance(labels[0], str):

        labels = torch.tensor([
            int(label.replace("n", ""))
            for label in labels
        ])

    else:

        labels = torch.tensor(labels)

    return images, labels


# ============================================================
# Benchmark
# ============================================================

def benchmark(mode, workers):

    dataset = TVDataset(mode)

    kwargs = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "num_workers": workers,
    }

    if mode == "batch":
        kwargs["collate_fn"] = batch_float_collate

    loader = DataLoader(**kwargs)

    cpu_before = cpu_time()

    start = time.perf_counter()

    total = 0

    for images, labels in loader:

        total += images.shape[0]

        # Ensure final interface is identical
        assert images.dtype == torch.float32
        assert images.shape[1:] == (3, 64, 64)

    elapsed = time.perf_counter() - start

    cpu_after = cpu_time()

    throughput = total / elapsed

    cores = (
        cpu_after - cpu_before
    ) / elapsed

    return throughput, cores


# ============================================================
# Execute
# ============================================================

MODES = [
    "two_alloc",
    "one_alloc",
    "batch",
]


print()
print("=" * 90)
print("PER-SAMPLE vs BATCH NORMALIZATION")
print("=" * 90)

print(
    "mode,workers,throughput,cpu_cores",
    flush=True
)


for mode in MODES:

    print()
    print(
        f"--- {mode.upper()} ---",
        flush=True
    )

    for workers in WORKERS:

        tp, cores = benchmark(
            mode,
            workers
        )

        print(
            f"{mode},"
            f"{workers},"
            f"{tp:.1f},"
            f"{cores:.2f}",
            flush=True
        )


print()
print("DONE", flush=True)

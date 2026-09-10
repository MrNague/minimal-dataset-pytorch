#!/usr/bin/env python3

import sys
import time
import resource
import gc

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


print("Loading Parquet table once...", flush=True)

base = ParquetDataset(
    PARQUET,
    max_samples=SAMPLES
)

table = base._table

print("Table ready.", flush=True)


# ============================================================
# Dataset:
# workers do ONLY
#
# parquet -> JPEG decode -> resize uint8
# ============================================================

class UInt8Dataset:

    def __len__(self):
        return SAMPLES

    def __getitem__(self, index):

        row = table.slice(index, 1).to_pylist()[0]

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


def make_labels(samples):

    labels = [
        sample[1]
        for sample in samples
    ]

    if isinstance(labels[0], str):

        return torch.tensor([
            int(x.replace("n", ""))
            for x in labels
        ])

    return torch.tensor(labels)


# ============================================================
# Collates
# ============================================================

def uint8_collate(samples):

    images = torch.stack([
        sample[0]
        for sample in samples
    ])

    return images, make_labels(samples)


def collate_float_inplace(samples):

    images = torch.stack([
        sample[0]
        for sample in samples
    ])

    images = images.float()
    images.div_(255.0)

    return images, make_labels(samples)


def collate_direct_div(samples):

    images = torch.stack([
        sample[0]
        for sample in samples
    ])

    # uint8 -> normalized float in one expression
    images = images / 255.0

    return images, make_labels(samples)


# ============================================================
# Benchmark
# ============================================================

def benchmark(mode, workers):

    if mode == "uint8_only":
        collate = uint8_collate

    elif mode == "collate_float":
        collate = collate_float_inplace

    elif mode == "collate_direct_div":
        collate = collate_direct_div

    elif mode == "consumer_float":
        collate = uint8_collate

    else:
        raise ValueError(mode)


    loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=workers,
        collate_fn=collate,
    )


    cpu_before = cpu_time()
    start = time.perf_counter()

    total = 0


    for images, labels in loader:

        if mode == "consumer_float":

            # Conversion AFTER the batch leaves the orchestrator.
            # Workers can meanwhile work on following batches.
            images = images.float()
            images.div_(255.0)


        if mode == "uint8_only":
            assert images.dtype == torch.uint8

        else:
            assert images.dtype == torch.float32


        total += images.shape[0]


    elapsed = time.perf_counter() - start

    cpu_after = cpu_time()


    throughput = total / elapsed

    cores = (
        cpu_after - cpu_before
    ) / elapsed


    del loader
    gc.collect()

    return throughput, cores


MODES = [
    "uint8_only",
    "collate_float",
    "collate_direct_div",
    "consumer_float",
]


print()
print("=" * 90)
print("CONVERSION PLACEMENT DIAGNOSTIC")
print("=" * 90)

print(
    "mode,workers,throughput,cpu_cores",
    flush=True
)


for mode in MODES:

    print(
        f"\n--- {mode.upper()} ---",
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


print("\nDONE", flush=True)

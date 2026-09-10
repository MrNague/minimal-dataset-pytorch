#!/usr/bin/env python3

import sys
import io
import time
import threading
import resource

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, "/home/nague/bachelor-project")

from minimal_dataset import ParquetDataset


PARQUET = "/fscratch/nague/storage_benchmarks/images.parquet"

CACHE_SIZE = 2048
OPERATIONS = 100000
WORKERS = [1, 2, 4, 8, 16, 32]

to_tensor = T.ToTensor()

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def cpu_time():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


# ============================================================
# Load data once
# ============================================================

print("Loading Parquet dataset once...", flush=True)

dataset = ParquetDataset(
    PARQUET,
    max_samples=CACHE_SIZE
)

table = dataset._table

print("Dataset loaded.", flush=True)


# ============================================================
# Prepare EXACT input that ToTensor receives in real pipeline
#
# JPEG
#   -> PIL decode
#   -> RGB
#   -> resize 64x64
#   -> [THIS is the input to ToTensor]
# ============================================================

print(
    f"Preparing {CACHE_SIZE} decoded + resized 64x64 PIL images...",
    flush=True
)

pil64 = []

for i in range(CACHE_SIZE):

    row = table.slice(i, 1).to_pylist()[0]

    img = Image.open(
        io.BytesIO(row["image"])
    )

    # Force JPEG decode explicitly
    img.load()

    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((64, 64))

    pil64.append(img)


print("64x64 PIL cache ready.", flush=True)


# ============================================================
# Precompute intermediate representations
# ============================================================

print("Preparing intermediate representations...", flush=True)

numpy_hwc = [
    np.array(
        img,
        dtype=np.uint8,
        copy=True
    )
    for img in pil64
]


torch_hwc = [
    torch.from_numpy(arr)
    for arr in numpy_hwc
]


torch_chw_view = [
    t.permute(2, 0, 1)
    for t in torch_hwc
]


torch_chw_uint8 = [
    t.contiguous()
    for t in torch_chw_view
]


torch_chw_float = [
    t.float()
    for t in torch_chw_uint8
]


print("Intermediate caches ready.", flush=True)


# ============================================================
# Isolated operations
# ============================================================

def current_totensor(i):
    """
    torchvision.transforms.ToTensor on the SAME 64x64 PIL
    input as the real pipeline.
    """
    return to_tensor(
        pil64[i % CACHE_SIZE]
    )


def pil_to_numpy(i):
    """
    PIL -> HWC uint8 NumPy copy.
    """
    return np.array(
        pil64[i % CACHE_SIZE],
        dtype=np.uint8,
        copy=True
    )


def numpy_to_torch(i):
    """
    NumPy -> torch tensor.
    torch.from_numpy is normally zero-copy.
    """
    return torch.from_numpy(
        numpy_hwc[i % CACHE_SIZE]
    )


def permute_view(i):
    """
    HWC -> CHW view.
    Normally no data copy yet.
    """
    return torch_hwc[
        i % CACHE_SIZE
    ].permute(2, 0, 1)


def contiguous_copy(i):
    """
    Non-contiguous CHW view -> contiguous CHW tensor.
    Real memory copy.
    """
    return torch_chw_view[
        i % CACHE_SIZE
    ].contiguous()


def float_conversion(i):
    """
    uint8 -> float32.
    Allocates a new tensor.
    """
    return torch_chw_uint8[
        i % CACHE_SIZE
    ].float()


def normalization(i):
    """
    float32 / 255.
    Allocates output tensor.
    """
    return torch_chw_float[
        i % CACHE_SIZE
    ] / 255.0


def manual_totensor(i):
    """
    Approximate manual decomposition of ToTensor:
      PIL
      -> numpy copy
      -> torch
      -> HWC/CHW
      -> contiguous
      -> float
      -> /255
    """

    arr = np.array(
        pil64[i % CACHE_SIZE],
        dtype=np.uint8,
        copy=True
    )

    tensor = torch.from_numpy(arr)

    tensor = tensor.permute(
        2, 0, 1
    ).contiguous()

    tensor = tensor.float()

    tensor = tensor / 255.0

    return tensor


MODES = {
    "totensor_64": current_totensor,
    "pil_to_numpy": pil_to_numpy,
    "numpy_to_torch": numpy_to_torch,
    "permute": permute_view,
    "contiguous": contiguous_copy,
    "float": float_conversion,
    "divide_255": normalization,
    "manual_totensor": manual_totensor,
}


# ============================================================
# Benchmark
# ============================================================

def benchmark(fn, workers):

    start_event = threading.Event()

    counts = [0] * workers


    def worker(worker_id):

        start_event.wait()

        count = 0

        for i in range(
            worker_id,
            OPERATIONS,
            workers
        ):

            result = fn(i)

            # Prevent accidental optimization / early release assumptions.
            _ = result

            count += 1

        counts[worker_id] = count


    threads = []

    for worker_id in range(workers):

        t = threading.Thread(
            target=worker,
            args=(worker_id,)
        )

        t.start()

        threads.append(t)


    cpu_before = cpu_time()

    start = time.perf_counter()

    start_event.set()


    for t in threads:
        t.join()


    wall = time.perf_counter() - start

    cpu_after = cpu_time()

    total = sum(counts)

    throughput = total / wall

    cpu_cores = (
        cpu_after - cpu_before
    ) / wall

    return throughput, cpu_cores


# ============================================================
# Run
# ============================================================

print()
print("=" * 90)
print("TOTENSOR 64x64 COMPONENT SCALING")
print("=" * 90)

print(
    "mode,workers,throughput,cpu_cores",
    flush=True
)


for mode_name, fn in MODES.items():

    print()
    print(
        f"--- {mode_name.upper()} ---",
        flush=True
    )

    for workers in WORKERS:

        tp, cores = benchmark(
            fn,
            workers
        )

        print(
            f"{mode_name},"
            f"{workers},"
            f"{tp:.1f},"
            f"{cores:.2f}",
            flush=True
        )


print()
print("DONE", flush=True)

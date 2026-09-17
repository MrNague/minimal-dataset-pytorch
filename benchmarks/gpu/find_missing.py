#!/usr/bin/env python3
"""Find which GPU/model/batch_size/optimizer combos are missing. """
import os

SUMMARY_DIR = "/netscratch/nague/gpu_benchmarks"

GPU_CONFIGS = {
    "A100-40GB": [32, 64, 128, 256, 384, 512, 768, 1024, 1280],
    "A100-80GB": [32, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536],
    "A100-PCI":  [32, 64, 128, 256, 384, 512, 768, 1024, 1280],
    "B200":      [32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 2560, 3072],
    "H100":      [32, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536],
    "H100-PCI":  [32, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536],
    "H200":      [32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 2560],
    "H200-PCI":  [32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 2560],
    "L40S":      [32, 64, 128, 192, 256, 384, 512, 640, 768],
    "RTX3090":   [32, 64, 128, 192, 256, 320],
    "RTXA6000":  [32, 64, 128, 192, 256, 384, 512, 640, 768],
    "RTXB6000":  [32, 64, 128, 192, 256, 320],
}

MODELS = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14",
    "swin_t", "swin_s", "swin_b",
    "swin_v2_t", "swin_v2_s", "swin_v2_b",
]

OPTIMIZERS = ["adam", "sgd"]

# Get existing summaries
existing = set()
for fname in os.listdir(SUMMARY_DIR):
    if fname.startswith("summary_") and fname.endswith(".csv"):
        # Parse filename: summary_{model}_bs{bs}_{opt}_{gpu}_{ts}.csv
        parts = fname.replace("summary_", "").replace(".csv", "").split("_")
        # Reconstruct from parts
        try:
            bs = int([p for p in parts if p.startswith("bs")][0].replace("bs", ""))
            opt = [p for p in parts if p in ("adam", "sgd")][0]
            # Model and GPU are harder — just store the tuple from CSV content
            with open(os.path.join(SUMMARY_DIR, fname)) as f:
                line = f.readlines()[-1]  # last line = data
                fields = line.strip().split(",")
                model = fields[0]
                gpu_raw = fields[3]
                bs_from_csv = int(fields[1])
                opt_from_csv = fields[2]
                existing.add((gpu_raw, model, bs_from_csv, opt_from_csv))
        except:
            pass

# Map real GPU names to partitions
GPU_MAP_REVERSE = {
    "NVIDIA RTX A6000": "RTXA6000",
    "NVIDIA GeForce RTX 3090": "RTX3090",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": "RTXB6000",
    "NVIDIA L40S": "L40S",
    "NVIDIA A40": "A40",
    "NVIDIA A100-PCIE-40GB": "A100-40GB",
    "NVIDIA A100-SXM4-40GB": "A100-40GB",
    "NVIDIA H100 NVL": "H100",
    "NVIDIA H200 NVL": "H200",
    "NVIDIA B200": "B200",
}

missing = []
for partition, batch_sizes in GPU_CONFIGS.items():
    for model in MODELS:
        for bs in batch_sizes:
            for opt in OPTIMIZERS:
                # Check if exists under any real GPU name for this partition
                found = False
                for real_name, part_name in GPU_MAP_REVERSE.items():
                    if part_name == partition and (real_name, model, bs, opt) in existing:
                        found = True
                        break
                if not found:
                    missing.append((partition, model, bs, opt))

print(f"Total missing: {len(missing)}")
for partition in GPU_CONFIGS:
    count = sum(1 for m in missing if m[0] == partition)
    print(f"  {partition}: {count} missing")

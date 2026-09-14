#!/usr/bin/env python3
"""Deep instrumentation benchmark for MP DataLoader."""
import sys, os, time, argparse

sys.path.insert(0, '/home/nague/bachelor-project')
from minimal_dataset import ParquetDataset
from dataloader_mp import DataLoaderMP

def parquet_factory():
    return ParquetDataset("/fscratch/nague/storage_benchmarks/images.parquet", max_samples=10000)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, required=True)
    args = parser.parse_args()

    temp_ds = parquet_factory()
    total = len(temp_ds)
    print(f"Dataset: {total} samples", flush=True)

    loader = DataLoaderMP(parquet_factory, total, batch_size=256,
                          num_workers=args.num_workers,
                          metrics_dir="/netscratch/nague/deep_mp_metrics")

    batch_count = 0
    start = time.time()
    for batch in loader:
        batch_count += 1
        if batch_count % 50 == 0:
            print(f"  [MP {args.num_workers}] batch {batch_count}", flush=True)

    elapsed = time.time() - start
    summary = loader._cleanup()

    if args.num_workers == 1:
        print("workers,batches,time_s,throughput,io_ms,decode_ms,preprocess_ms,staging_put_ms,collate_ms,batch_put_ms,total_sample_ms")

    st = summary["stage_times"]
    print(f"{args.num_workers},{batch_count},{elapsed:.2f},{batch_count*256/elapsed:.1f},"
          f"{st['io']['mean_ms']},{st['decode']['mean_ms']},{st['preprocess']['mean_ms']},"
          f"{st['staging_put']['mean_ms']},{st['collate']['mean_ms']},{st['batch_put']['mean_ms']},"
          f"{st['total_sample']['mean_ms']}", flush=True)

if __name__ == '__main__':
    main()

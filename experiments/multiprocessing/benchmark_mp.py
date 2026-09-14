#!/usr/bin/env python3
"""Benchmark multiprocessing DataLoader."""
import sys, os, time, argparse

def main():
    sys.path.insert(0, '/home/nague/bachelor-project')
    from minimal_dataset import ParquetDataset
    from dataloader_mp import DataLoaderMP

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    args = parser.parse_args()

    def parquet_factory():
        return ParquetDataset(
            "/fscratch/nague/storage_benchmarks/images.parquet"
        )

    dataset = parquet_factory()
    total = len(dataset)

    loader = DataLoaderMP(
        parquet_factory,
        total,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    batch_count = 0
    start = time.time()
    for batch in loader:
        batch_count += 1
        if batch_count % 500 == 0:
            print(f"  [MP {args.batch_size}/{args.num_workers}] batch {batch_count}", flush=True)

    elapsed = time.time() - start
    samples = batch_count * args.batch_size
    print(f"MP,{args.batch_size},{args.num_workers},{batch_count},{elapsed:.2f},{samples/elapsed:.1f}", flush=True)

if __name__ == '__main__':
    main()

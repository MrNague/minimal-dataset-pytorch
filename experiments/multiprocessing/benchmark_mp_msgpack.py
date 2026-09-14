#!/usr/bin/env python3
"""Benchmark multiprocessing DataLoader with MessagePack dataset."""
import sys, os, time, argparse

sys.path.insert(0, '/home/nague/bachelor-project')

# Import only the classes we need from benchmark_msgpack
# The file imports file_io which is in the same directory
sys.path.insert(0, '/home/nague/bachelor-project/benchmarks/storage')

import msgpack
import numpy as np
import io
from PIL import Image
from torch.utils.data import Dataset
from file_io import BinaryReader

class ImageDataset(Dataset):
    def __init__(self, dataset_dir: str, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.shard_paths = []
        self.shard_readers = {}
        self.shard_lengths = []
        self.total_length = 0
        self._discover_shards()
        self._calculate_offsets()

    def _discover_shards(self):
        shard_indices = []
        for item in os.listdir(self.dataset_dir):
            item_path = os.path.join(self.dataset_dir, item)
            if os.path.isdir(item_path) and item.isdigit():
                shard_indices.append(int(item))
        shard_indices.sort()
        for shard_idx in shard_indices:
            shard_path = os.path.join(self.dataset_dir, str(shard_idx))
            self.shard_paths.append(shard_path)
            reader = BinaryReader(shard_path)
            self.shard_readers[shard_idx] = reader
            self.shard_lengths.append(len(reader))
            self.total_length += len(reader)

    def _calculate_offsets(self):
        self.shard_offsets = [0]
        for length in self.shard_lengths:
            self.shard_offsets.append(self.shard_offsets[-1] + length)

    def _find_shard_and_index(self, global_index):
        for i, offset in enumerate(self.shard_offsets[1:]):
            if global_index < offset:
                return i, global_index - self.shard_offsets[i]
        raise IndexError

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        shard_idx, local_idx = self._find_shard_and_index(index)
        reader = self.shard_readers[shard_idx]
        raw_bytes = reader[local_idx]
        data_dict = msgpack.unpackb(raw_bytes, raw=False)
        label = data_dict['label']
        img_bytes = data_dict['image']
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


from dataloader_mp import DataLoaderMP


def msgpack_factory():
    return ImageDataset("/fscratch/nague/storage_benchmarks/msgpack")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    args = parser.parse_args()

    temp_ds = msgpack_factory()
    total = len(temp_ds)
    print(f"Dataset: {total} samples", flush=True)

    loader = DataLoaderMP(msgpack_factory, total, batch_size=args.batch_size, num_workers=args.num_workers)

    batch_count = 0
    start = time.time()
    for batch in loader:
        batch_count += 1
        if batch_count % 500 == 0:
            print(f"  [MP-MsgPack {args.batch_size}/{args.num_workers}] batch {batch_count}", flush=True)

    elapsed = time.time() - start
    samples = batch_count * args.batch_size
    print(f"MP_MSGPACK,{args.batch_size},{args.num_workers},{batch_count},{elapsed:.2f},{samples/elapsed:.1f}", flush=True)

if __name__ == '__main__':
    main()

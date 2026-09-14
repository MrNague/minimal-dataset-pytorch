#!/usr/bin/env python3
"""Test if JPEG decoding slows down with more threads (without DataLoader)."""
import sys, io, time, threading
from PIL import Image
import pyarrow.parquet as pq

table = pq.read_table("/fscratch/nague/storage_benchmarks/images.parquet")

test_images = []
for i in range(1000):
    row = table.slice(i, 1).to_pylist()[0]
    test_images.append(row['image'])

print(f"Test images: {len(test_images)}", flush=True)

def decode_worker(worker_id, results):
    local_images = test_images[worker_id*100:(worker_id+1)*100]
    start = time.perf_counter()
    for img_bytes in local_images:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    elapsed = time.perf_counter() - start
    results[worker_id] = elapsed / len(local_images)

for nw in [1, 2, 4, 8, 16, 32]:
    results = [0] * nw
    threads = []
    for w in range(nw):
        t = threading.Thread(target=decode_worker, args=(w, results))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    
    avg_time = sum(results) / len(results)
    print(f"Workers: {nw}, Avg decode time per image: {avg_time*1000:.2f} ms", flush=True)

print("DONE")

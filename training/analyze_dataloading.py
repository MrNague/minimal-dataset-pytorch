"""
Script analysis for performances of data loading.
Mesure : chargement time vs calculus time , throughput, GPU utilization.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import time

# ============================================================
# CONFIGURATION 
# ============================================================

DATA_DIR = "/ds-sds/images/imagenet"
BATCH_SIZE = 256
NUM_EPOCHS = 3  # Just 3 epochs for analysis
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8

print("=" * 60)
print("DATA LOADING PERFORMANCE ANALYSIS")
print("=" * 60)
print(f"Data source: {DATA_DIR}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Num workers: {NUM_WORKERS}")
print(f"Epochs to run: {NUM_EPOCHS}")
print("=" * 60)

# Data loading (same transformations as the original job )
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("\nLoading dataset...")
train_dataset = torchvision.datasets.ImageFolder(
    root=f"{DATA_DIR}/train",
    transform=train_transforms
)
print(f"Dataset: {len(train_dataset):,} images")

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True
)

# Model
print("Creating ResNet-50...")
model = models.resnet50(weights=None)
model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# ============================================================
# PROFILING LOOP
# ============================================================

print("\n" + "=" * 60)
print("STARTING PROFILING RUN (3 epochs)")
print("=" * 60)
print(f"{'Epoch':<8} {'Batch':<10} {'Data(s)':<10} {'Compute(s)':<12} {'Total(s)':<10} {'GPU_Wait%':<12} {'Img/s':<10}")
print("-" * 70)

# Warmup: 50 batches for stabilize
print("Warmup (50 batches)...")
model.train()
warmup_iter = iter(train_loader)
for _ in range(50):
    images, labels = next(warmup_iter)
    images, labels = images.cuda(), labels.cuda()
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# Profiling
model.train()
epoch_stats = []

for epoch in range(NUM_EPOCHS):
    epoch_data_time = 0.0
    epoch_compute_time = 0.0
    epoch_samples = 0
    
    data_start = time.time()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        data_time = time.time() - data_start
        
        compute_start = time.time()
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        compute_time = time.time() - compute_start
        
        epoch_data_time += data_time
        epoch_compute_time += compute_time
        epoch_samples += images.size(0)
        
        if (batch_idx + 1) % 500 == 0:
            total_time = data_time + compute_time
            gpu_wait_pct = 100.0 * data_time / total_time if total_time > 0 else 0
            img_per_sec = BATCH_SIZE / total_time if total_time > 0 else 0
            print(f"{epoch+1:<8} {batch_idx+1:<10} {data_time:<10.3f} {compute_time:<12.3f} {total_time:<10.3f} {gpu_wait_pct:<12.1f} {img_per_sec:<10.1f}")
        
        data_start = time.time()
    
    # Stats for the epoch
    avg_data = epoch_data_time / len(train_loader)
    avg_compute = epoch_compute_time / len(train_loader)
    avg_total = avg_data + avg_compute
    gpu_wait = 100.0 * avg_data / avg_total if avg_total > 0 else 0
    throughput = BATCH_SIZE / avg_total if avg_total > 0 else 0
    
    epoch_stats.append({
        'epoch': epoch + 1,
        'avg_data_time': avg_data,
        'avg_compute_time': avg_compute,
        'avg_total_time': avg_total,
        'gpu_wait_pct': gpu_wait,
        'throughput_img_s': throughput
    })
    
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Avg data time/batch:   {avg_data:.3f}s ({gpu_wait:.1f}% GPU wait)")
    print(f"  Avg compute time/batch: {avg_compute:.3f}s")
    print(f"  Avg total time/batch:   {avg_total:.3f}s")
    print(f"  Throughput:             {throughput:.1f} images/s")
    print()

# ============================================================
# FINAL REPORT
# ============================================================
print("=" * 60)
print("FINAL PROFILING SUMMARY")
print("=" * 60)
print(f"{'Epoch':<8} {'Data(s)':<10} {'Compute(s)':<12} {'Total(s)':<10} {'GPU_Wait%':<12} {'Img/s':<10}")
print("-" * 65)
for s in epoch_stats:
    print(f"{s['epoch']:<8} {s['avg_data_time']:<10.3f} {s['avg_compute_time']:<12.3f} {s['avg_total_time']:<10.3f} {s['gpu_wait_pct']:<12.1f} {s['throughput_img_s']:<10.1f}")

overall_data = sum(s['avg_data_time'] for s in epoch_stats) / len(epoch_stats)
overall_compute = sum(s['avg_compute_time'] for s in epoch_stats) / len(epoch_stats)
overall_total = sum(s['avg_total_time'] for s in epoch_stats) / len(epoch_stats)
overall_wait = 100.0 * overall_data / overall_total
overall_throughput = BATCH_SIZE / overall_total

print(f"\n{'OVERALL':<8} {overall_data:<10.3f} {overall_compute:<12.3f} {overall_total:<10.3f} {overall_wait:<12.1f} {overall_throughput:<10.1f}")
print("\nKey Finding:")
print(f"  GPU spends {overall_wait:.1f}% of time waiting for data!")
print(f"  Throughput: {overall_throughput:.1f} images/s")
print(f"  For 1.28M images: ~{1281167 / overall_throughput / 3600:.1f} hours per epoch")
print(f"  For 90 epochs: ~{90 * 1281167 / overall_throughput / 3600:.1f} hours total")
print(f"  GPU utilization ceiling: {100 - overall_wait:.1f}%")

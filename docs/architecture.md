# Project Documentation

This directory contains architecture diagrams and benchmark figures produced during the development of the Minimal Dependency Dataset Library for PyTorch.

## Architecture

Architecture and project-overview diagrams are stored in:

`docs/images/architecture/`

Main figures include:

- `00_MainOverview1.png`
- `01_MainOverview2.png`
- `02_dataLoader1.png`
- `03_dataLoader2.png`
- `04_Storage Format Evaluation.png`
- `05_GPU Compute Delay.png`
- `06_DataLoaderBenchmark.png`
- `minimal-dataset-design.png`

## Final Performance Comparison

Final and before/after performance figures are stored in:

`docs/images/comparison/`

- `final_scaling.png`
- `plot_decode_before_after.png`
- `plot_throughput_before_after.png`

## DataLoader / Parquet Benchmarks

Benchmark figures produced during the DataLoader experiments are stored in:

`docs/images/parquet/`

- `plot_comparison_bs16.png`
- `plot_local_vs_remote.png`
- `plot_ours_throughput.png`
- `plot_pytorch_throughput.png`
- `plot_pytorch_vs_ours_current.png`
- `plot_speedup.png`
- `plot_speedup_comparison.png`
- `plot_speedup_linear.png`
- `plot_throughput_linear.png`

## Pipeline Profiling

Per-stage profiling figures are stored in:

`docs/images/stages/`

- `plot_stages.png`
- `plot_stages_combined.png`

## Storage Experiments

MessagePack/storage-format comparison figures are stored in:

`docs/images/msgpack/`

- `plot_format_comparison.png`

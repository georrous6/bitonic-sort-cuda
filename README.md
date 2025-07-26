# Bitonic Sort - CUDA Benchmark

This project implements **Bitonic Sort** in CUDA using three progressively optimized kernel versions, each benchmarked for performance and correctness. It supports sorting in both **ascending** and **descending** order, and includes profiling with `nsys`.

## Project Structure

- **`.vscode`**:
- **`benchmarks`**:
- **`docs`**:
- **`include`**:
- **`src`**:
- **`tests`**:

## Kernel Versions (Homework 3)

- V0: Each thread performs one compare-exchange. No inner loop.  
  --> Simple but results in many kernel launches and global synchronizations.

- V1: Inner loop (k) is moved into the kernel.  
  --> Fewer launches and syncs. Requires custom intra-block synchronization.

- V2: Like V1, but uses local/shared memory for better performance.  
  --> Significantly reduced global memory access; highest performance.

## Benchmark Usage

Compile with:

```
make
```

Run benchmark with:

```
./build/benchmark <n> [--kernel v0|v1|v2] [--desc]
```

- <n>: Total number of elements to sort (must be power of two).
- --kernel: Choose kernel version (v0, v1, or v2). Omit for serial sort.
- --desc: Optional flag for descending sort. Default is ascending.

Example:

```
./build/benchmark 1048576 --kernel v1
```

## Profiling with nsys

A sample Slurm script for profiling with NVIDIA NSight Systems is provided in `scripts/run_benchmark.sh`.

To run:

```
sbatch scripts/run_benchmark.sh /absolute/path/to/project
```

It checks if the GPU is CUDA-compatible and then launches `nsys` on the executable.

## Portability Notes

Paths to CUDA include and lib are configurable:

```
CUDA_INC ?= /usr/include
CUDA_LIB ?= /usr/lib/x86_64-linux-gnu
```

You can override these by setting environment variables:

```
export CUDA_INC=/path/to/cuda/include
export CUDA_LIB=/path/to/cuda/lib64
```

The Makefile uses `?=` so it's flexible across different machines and clusters.

## Clean Build

```
make clean
```

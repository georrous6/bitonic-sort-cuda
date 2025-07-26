#!/bin/bash
#SBATCH --job-name=bitonic_benchmark
#SBATCH --partition=ampere
#SBATCH --output=logs/slurm.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

set -e  # Exit immediately on error

export UCX_WARN_UNUSED_ENV_VARS=n

# --- Check and read input argument ---
PROJECT_DIR="$1"
if [ -z "$PROJECT_DIR" ]; then
    echo "Usage: $0 <project_dir>"
    exit 1
fi

# --- Move to the project directory ---
cd "$PROJECT_DIR" || { echo "Cannot cd to $PROJECT_DIR"; exit 1; }

# --- Check if log directory exists, if not create it ---
if [ ! -d "logs" ]; then
    mkdir logs
fi

# --- Load required modules ---
module purge
module load gcc cuda

# --- Build the project ---
make

# --- Define executable path ---
EXECUTABLE="$PROJECT_DIR/build/benchmark"

# --- Check if executable exists ---
if [ ! -x "$EXECUTABLE" ]; then
    echo "Error: Executable not found or not executable: $EXECUTABLE"
    exit 1
fi

# --- Set parameters ---
SIZE=$((2 ** 20))
KERNEL="v0"

# --- Run profiling with Nsight Systems ---
nsys profile -t cuda --stats=true "$EXECUTABLE" "$SIZE" --kernel "$KERNEL" --desc

echo "All benchmarks completed successfully."

#!/bin/bash
#SBATCH --job-name=bitonic_benchmark
#SBATCH --partition=gpu
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

# --- Specify report and log directories
LOGS_DIR="logs"
REPORTS_DIR="reports"
mkdir -p "$LOGS_DIR"
mkdir -p "$REPORTS_DIR"

# --- Load required modules ---
module purge
module load gcc cuda

# --- Check for CUDA-compatible GPU ---
echo -e "\n=== Checking for CUDA-compatible GPU ==="
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
if [ -z "$GPU_NAME" ]; then
    echo "No CUDA-compatible GPU detected."
    exit 1
fi
echo "CUDA-compatible GPU detected: $GPU_NAME"

# --- Check CUDA environment variables ---
echo -e "\n=== Checking CUDA environment variables ==="
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_PATH=$CUDA_PATH"

# --- Check if nvcc is available ---
echo -e "\n=== Checking nvcc compiler ==="
echo "which nvcc:"
which nvcc
echo "nvcc --version:"
nvcc --version

# --- Build the project ---
echo -e "\n=== Building the project ==="
make

# --- Define executable path ---
EXECUTABLE="$PROJECT_DIR/build/benchmark"

# --- Check if executable exists ---
if [ ! -x "$EXECUTABLE" ]; then
    echo "Error: Executable not found or not executable: $EXECUTABLE"
    exit 1
fi

# --- Set parameters ---

# --- Run profiling with Nsight Systems ---
echo -e "\n=== Profiling with Nsight Systems ==="

KERNELS=("v0" "none")

for KERNEL in "${KERNELS[@]}"; do
    
    # Define sizes to test
    for q in {10..20}; do

        SIZE=$((2 ** q))
        echo -e "\n --- Running benchmark for kernel $KERNEL with size $SIZE ---"
        
        # Run the benchmark and profile it
        nsys profile -t cuda --stats=true -o "$REPORTS_DIR/report_${KERNEL}_${q}" \ 
        "$EXECUTABLE" "$SIZE" --kernel "$KERNEL" --desc
    done
done

echo -e "\nAll benchmarks completed successfully."

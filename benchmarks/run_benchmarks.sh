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
PLOTS_DIR="docs/figures"
TIMING_FILE="total_times.log"
rm -rf "$REPORTS_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "$REPORTS_DIR"
mkdir -p "$PLOTS_DIR"

# --- Load required modules ---
module purge
module load gcc/13.2.0-nbog6z2 cuda/12.4.0-obe7ebz python/3.11.7-47zltq2
module list

# --- Create python environment ---
if [ ! -d ~/grousenv ]; then
    python3 -m venv ~/grousenv
    source ~/grousenv/bin/activate
    pip install numpy pandas matplotlib
else
    source ~/grousenv/bin/activate
fi

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
make clean
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

KERNELS=("v0")
Q_MIN=10
Q_MAX=20

for KERNEL in "${KERNELS[@]}"; do
    
    # Define sizes to test
    for q in $(seq $Q_MIN $Q_MAX); do

        echo -e "\n --- Running benchmark for kernel $KERNEL with q=$q ---"

        # Run the benchmark and profile it
        nsys profile -t cuda --stats=true -o "$REPORTS_DIR/report_${KERNEL}_${q}" \
        "$EXECUTABLE" "$q" --kernel "$KERNEL" --timing-file "$REPORTS_DIR/$TIMING_FILE"
    done
done

# --- Run benchmarks for CPU-only version ---
for q in $(seq $Q_MIN $Q_MAX); do
    echo -e "\n --- Running CPU-only benchmark with q=$q ---"
    "$EXECUTABLE" "$q" --kernel none --timing-file "$REPORTS_DIR/$TIMING_FILE"
done

# --- Remove nsys-rep files
rm "$REPORTS_DIR"/*.nsys-rep

# --- Export results ---
python3 benchmarks/export_benchmark_results.py "$REPORTS_DIR" "$PLOTS_DIR"

echo -e "\nAll benchmarks completed successfully."

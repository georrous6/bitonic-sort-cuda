#!/bin/bash
#SBATCH --job-name=bitonic_benchmark
#SBATCH --partition=gpu
#SBATCH --output=slurm.out
#SBATCH --time=00:20:00
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
LOGS_DIR="benchmarks/logs"
PLOTS_DIR="docs/figures"
DATA_DIR="docs/data"
TIMING_FILE="$LOGS_DIR/execution_times.log"
rm -rf "$LOGS_DIR" "$PLOTS_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "$PLOTS_DIR"

# --- Load required modules ---
module purge
module load gcc/12.2.0 cuda/12.2.1-bxtxsod python/3.10.8-cidwh6y
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
nvidia-smi

# --- Check CUDA environment variables ---
echo -e "\n=== Checking CUDA environment variables ==="
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_PATH=$CUDA_PATH"
export CUDA_HOME=$CUDA_HOME

# --- Check if nvcc is available ---
echo -e "\n=== Checking nvcc compiler ==="
echo "which nvcc:"
which nvcc
echo "nvcc --version:"
nvcc --version

# --- Build the project ---
echo -e "\n=== Building the project ==="
make clean
make BUILD_TYPE=release -j$(nproc)

# --- Define executable path ---
EXECUTABLE="$PROJECT_DIR/build/release/benchmarks"

# --- Check if executable exists ---
if [ ! -x "$EXECUTABLE" ]; then
    echo "Error: Executable not found or not executable: $EXECUTABLE"
    exit 1
fi

KERNELS=("none" "v0" "v1" "v2")
Q_MIN=10
Q_MAX=27

for KERNEL in "${KERNELS[@]}"; do
    
    # Define sizes to test
    for q in $(seq $Q_MIN $Q_MAX); do

        echo "--- Running benchmark for kernel $KERNEL with q=$q ---"

        "$EXECUTABLE" "$q" --kernel "$KERNEL" --timing-file "$TIMING_FILE"
    done
done

# --- Export results ---
python3 benchmarks/export_benchmark_results.py "$TIMING_FILE" "$PLOTS_DIR" "$DATA_DIR"

echo -e "\nAll benchmarks completed successfully."

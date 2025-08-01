#!/bin/bash
#SBATCH --job-name=bitonic_profile
#SBATCH --partition=gpu
#SBATCH --output=logs/slurm-%j.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

start=$(date +%s)

export UCX_WARN_UNUSED_ENV_VARS=n

# --- Check and read input argument ---
PROJECT_DIR="$1"
VERSION="$2"
q="$3"
if [ -z "$PROJECT_DIR" ] || [ -z "$VERSION" ] || [ -z "$q" ]; then
    echo "Usage: $0 <project_dir> <version> <q>"
    exit 1
fi

# --- Print job information ---
echo -e "\n*** Running Nsight Systems Profiler on partition: $SLURM_JOB_PARTITION ***"
echo "Date: $(date)"

# --- Move to the project directory ---
cd "$PROJECT_DIR" || { echo "Cannot cd to $PROJECT_DIR"; exit 1; }

# --- Load required modules ---
module purge
module load gcc/12.2.0 cuda/12.2.1-bxtxsod
module list

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
make BUILD_TYPE=debug -j$(nproc)

# --- Define executable path ---
EXECUTABLE="$PROJECT_DIR/build/debug/benchmarks"

# --- Check if executable exists ---
if [ ! -x "$EXECUTABLE" ]; then
    echo "Error: Executable not found or not executable: $EXECUTABLE"
    exit 1
fi

# --- Run Nsight Systems Profiler ---
echo -e "\n=== Running Nsight Systems Profiler for $VERSION version with q=$q ==="
OUTPUT_FILE="$PROJECT_DIR/profile/reports/report_${VERSION}_${q}"
nsys profile --trace=cuda,nvtx --force-overwrite true -o "$OUTPUT_FILE" \
    "$EXECUTABLE" "$q" --version "$VERSION"
if [ $? -ne 0 ]; then
    echo "Error: Nsight Systems profiling failed"
    exit 1
fi

end=$(date +%s)
ellapsed=$((end - start))

minutes=$((ellapsed / 60))
seconds=$((ellapsed % 60))

echo -e "\nNsight Systems profiling completed successfully after $minutes minutes and $seconds seconds."

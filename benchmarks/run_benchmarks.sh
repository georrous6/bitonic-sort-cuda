#!/bin/bash
#SBATCH --job-name=bitonic_benchmark
#SBATCH --partition=gpu
#SBATCH --output=slurm.out
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

start=$(date +%s)

export UCX_WARN_UNUSED_ENV_VARS=n

# --- Check and read input argument ---
PROJECT_DIR="$1"
if [ -z "$PROJECT_DIR" ]; then
    echo "Usage: $0 <project_dir>"
    exit 1
fi

# --- Print job information ---
echo "Date: $(date)"
echo "Running on partition: $SLURM_JOB_PARTITION"

# --- Move to the project directory ---
cd "$PROJECT_DIR" || { echo "Cannot cd to $PROJECT_DIR"; exit 1; }

# --- Specify report and log directories
RESULTS_DIR="benchmarks/results"
TIMING_FILE="$RESULTS_DIR/execution_times_$SLURM_JOB_PARTITION.log"
mkdir -p "$RESULTS_DIR"

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

VERSIONS=("serial" "v0" "v1" "v2" "v3")
Q_MIN=10
Q_MAX=27

for VERSION in "${VERSIONS[@]}"; do

    # Define sizes to test
    for q in $(seq $Q_MIN $Q_MAX); do

        echo "--- Running benchmark for version $VERSION with q=$q ---"

        "$EXECUTABLE" "$q" --version "$VERSION" --timing-file "$TIMING_FILE"
        if [ $? -ne 0 ]; then
            echo "Error running benchmark for version $VERSION with q=$q"
            exit 1
        fi
    done
done

# --- Export results ---
python3 benchmarks/export_benchmark_results.py "$TIMING_FILE" "$RESULTS_DIR" "$SLURM_JOB_PARTITION"
if [ $? -ne 0 ]; then
    echo "Error exporting benchmark results"
    exit 1
fi

end=$(date +%s)
ellapsed=$((end - start))

minutes=$((ellapsed / 60))
seconds=$((ellapsed % 60))

echo -e "\nBenchmarks completed successfully after $minutes minutes and $seconds seconds."

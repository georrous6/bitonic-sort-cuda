#!/bin/bash

PROJECT_ROOT="$PWD"/..
PARTITION="$1"
if [ -z "$PARTITION" ]; then
    echo "Usage: $0 <partition>. Available partitions: gpu, ampere"
    exit 1
fi

rm -f *.log *.out
sbatch --partition "$PARTITION" run_nsys_profiler.sh "$PROJECT_ROOT" "v4" "20"

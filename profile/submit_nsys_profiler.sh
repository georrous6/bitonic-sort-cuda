#!/bin/bash

PROJECT_ROOT="$PWD"/..
PARTITION="$1"
if [ -z "$PARTITION" ]; then
    echo "Usage: $0 <partition>. Available partitions: gpu, ampere"
    exit 1
fi

rm -rf reports/
rm -rf logs/
mkdir -p logs/
mkdir -p reports/
sbatch --partition "$PARTITION" run_nsys_profile.sh "$PROJECT_ROOT" "v4" "20"

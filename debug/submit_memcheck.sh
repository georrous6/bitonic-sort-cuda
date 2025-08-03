#!/bin/bash

PROJECT_ROOT="$PWD"/..
PARTITION="$1"
if [ -z "$PARTITION" ]; then
    echo "Usage: $0 <partition>. Available partitions: gpu, ampere"
    exit 1
fi

rm -rf results/
mkdir -p results/
rm -f *.out
sbatch --partition "$PARTITION" run_memcheck.sh "$PROJECT_ROOT" "v4" "20"

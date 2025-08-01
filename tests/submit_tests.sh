#!/bin/bash

PROJECT_ROOT="$PWD"/..
PARTITION="$1"
if [ -z "$PARTITION" ]; then
    echo "Usage: $0 <partition>. Available partitions: gpu, ampere"
    exit 1
fi

rm -f *.out
sbatch --partition "$PARTITION" run_tests.sh "$PROJECT_ROOT"

#!/bin/bash

PROJECT_ROOT="$PWD"/../

rm *.log
sbatch --partition gpu run_memcheck.sh "$PROJECT_ROOT" "v4" "27"
sbatch --partition ampere run_memcheck.sh "$PROJECT_ROOT" "v4" "27"

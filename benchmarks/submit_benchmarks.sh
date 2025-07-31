#!/bin/bash

PROJECT_ROOT="$PWD"/../

sbatch --partition=gpu run_benchmarks.sh "$PROJECT_ROOT"
sbatch --partition=ampere run_benchmarks.sh "$PROJECT_ROOT"
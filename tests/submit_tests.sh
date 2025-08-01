#!/bin/bash

PROJECT_ROOT="$PWD"/../

rm *.out
sbatch --partition gpu run_tests.sh "$PROJECT_ROOT"
sbatch --partition ampere run_tests.sh "$PROJECT_ROOT"

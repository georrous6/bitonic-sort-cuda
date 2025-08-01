#!/bin/bash

PROJECT_ROOT="$PWD"/..

rm -rf results/
rm *.out
sbatch --partition ampere run_memcheck.sh "$PROJECT_ROOT" "v4" "20"


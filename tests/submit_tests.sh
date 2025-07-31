#!/bin/bash

PROJECT_ROOT="$PWD"/../

rm *.out
sbatch run_tests.sh "$PROJECT_ROOT"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include "bitonic_sort_cuda.cuh"


void fill_random(int *arr, int n) {
    srand(0);
    for (int i = 0; i < n; ++i)
        arr[i] = rand();
}


bitonic_version_t parse_kernel_version(const char *arg) {
    if (strcmp(arg, "v0") == 0) return VERSION_V0;
    if (strcmp(arg, "v1") == 0) return VERSION_V1;
    if (strcmp(arg, "v2") == 0) return VERSION_V2;
    return VERSION_SERIAL;
}


int parse_arguments(int argc, char **argv, int **data, int *q, const char **timing_filename, bitonic_version_t *version, int *descending, int *validate) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <q> [--kernel v0|v1|v2|none] [--desc] [--timing-file <file>] [--no-validate]\n", argv[0]);
        return EXIT_FAILURE;
    }

    *q = atoi(argv[1]);
    if (*q < 1 || *q > 31) {
        fprintf(stderr, "Invalid value for q: %d. Must be between 1 and 31.\n", *q);
        return EXIT_FAILURE;
    }
    const int n = (1 << *q);

    *data = (int *)malloc(n * sizeof(int));
    if (!(*data)) {
        fprintf(stderr, "Memory allocation failed for data array.\n");
        return EXIT_FAILURE;
    }

    *version = VERSION_SERIAL;
    *descending = 0;
    *validate = 1;  // default: validate

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            *version = parse_kernel_version(argv[i + 1]);
            if (*version == VERSION_SERIAL && strcmp(argv[i + 1], "none") != 0) {
                fprintf(stderr, "Invalid kernel version: %s. Falling back to serial.\n", argv[i + 1]);
            }
            i++;
        } else if (strcmp(argv[i], "--timing-file") == 0 && i + 1 < argc) {
            *timing_filename = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--desc") == 0) {
            *descending = 1;
        } else if (strcmp(argv[i], "--no-validate") == 0) {
            *validate = 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}


int save_timing_data(const char *filename, int q, bitonic_version_t version, double time_ms) {
    FILE *file = fopen(filename, "a");
    if (!file) {
        fprintf(stderr, "Could not open file %s for writing.\n", filename);
        return EXIT_FAILURE;
    }

    // Write the header if the file is empty
    if (ftell(file) == 0) {
        fprintf(file, "q,kernel,time_ms\n");
    }

    // Write the timing data
    const char *version_str[] = { "none", "v0", "v1", "v2" };
    fprintf(file, "%d,%s,%lf\n", q, version_str[version], time_ms);
    fclose(file);
    return EXIT_SUCCESS;
}


int validate_sort(int *arr, int n, int ascending) {
    for (int i = 0; i < n - 1; ++i) {
        if ((ascending && arr[i] > arr[i + 1]) || (!ascending && arr[i] < arr[i + 1])) {
            fprintf(stderr, "Validation failed at index %d: %d vs %d\n", i, arr[i], arr[i + 1]);
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}


int main(int argc, char **argv) {
    int q, descending, validate;
    bitonic_version_t version;
    int *data = NULL;
    const char *timing_filename = NULL;

    // Warmup CUDA
    if (wakeup_cuda()) return EXIT_FAILURE;

    if (parse_arguments(argc, argv, &data, &q, &timing_filename, &version, &descending, &validate) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    const int n = (1 << q);
    fill_random(data, n);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    int status = bitonic_sort_cuda(data, n, !descending, version);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_ms = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_nsec - start.tv_nsec) / 1e6;

    if (timing_filename) {
        if (save_timing_data(timing_filename, q, version, elapsed_ms)) {
            free(data);
            return EXIT_FAILURE;
        }
    }

    if (validate) {
        if (validate_sort(data, n, !descending)) {
            free(data);
            return EXIT_FAILURE;
        }
    }

    free(data);
    return status;
}

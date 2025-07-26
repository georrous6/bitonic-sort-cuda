#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bitonic_sort.cuh"


void fill_random(int *arr, int n) {
    srand(0);
    for (int i = 0; i < n; ++i)
        arr[i] = rand();
}


kernel_version_t parse_kernel_version(const char *arg) {
    if (strcmp(arg, "v0") == 0) return KERNEL_V0;
    if (strcmp(arg, "v1") == 0) return KERNEL_V1;
    if (strcmp(arg, "v2") == 0) return KERNEL_V2;
    return KERNEL_NONE;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <n> [--kernel v0|v1|v2] [--desc]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    if ((n & (n - 1)) != 0 || n <= 1) {
        fprintf(stderr, "Error: n must be a power of 2 and > 1.\n");
        return EXIT_FAILURE;
    }

    kernel_version_t version = KERNEL_NONE;
    int descending = 0;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            version = parse_kernel_version(argv[i + 1]);
            if (version == KERNEL_NONE && strcmp(argv[i + 1], "none") != 0) {
                fprintf(stderr, "Invalid kernel version: %s. Falling back to serial.\n", argv[i + 1]);
            }
            i++; // skip next arg
        } 
        else if (strcmp(argv[i], "--desc") == 0) {
            descending = 1;
        } 
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
    }

    int *data = (int *)malloc(n * sizeof(int));
    if (!data) {
        perror("malloc");
        return EXIT_FAILURE;
    }

    fill_random(data, n);

    int status = bitonic_sort(data, n, !descending, version);

    free(data);
    return status;
}

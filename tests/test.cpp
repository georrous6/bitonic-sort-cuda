#include <stdio.h>
#include <stdlib.h>
#include "bitonic_sort_cuda.cuh"

#define Q_MIN 1
#define Q_MAX 23

#define BOLD_BLUE "\033[1;34m"
#define BOLD_GREEN "\033[1;32m"
#define BOLD_RED "\033[1;31m"
#define RESET "\033[0m"


int cmp_asc(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}


int cmp_desc(const void *a, const void *b) {
    return (*(int *)b - *(int *)a);
}


int test_case(int q, int descending, bitonic_version_t version) {

    printf(BOLD_BLUE "Running test case with q = %d, descending = %d, kernel version = %d ...\n" RESET, q, descending, version - 1);
    fflush(stdout);

    int n = 1 << q;
    int *data = (int *)malloc(n * sizeof(int));
    int *sorted_data = (int *)malloc(n * sizeof(int));
    if (!data || !sorted_data) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize the array with random values
    for (int i = 0; i < n; i++) {
        data[i] = rand();
        sorted_data[i] = data[i];
    }

    qsort(sorted_data, n, sizeof(int), descending ? cmp_desc : cmp_asc);
    if (bitonic_sort_cuda(data, n, descending, version)) {
        free(data);
        free(sorted_data);
        return EXIT_FAILURE;
    }

    int status = EXIT_SUCCESS;

    // Print sorted array
    for (int i = 0; i < n; i++) {
        if (sorted_data[i] != data[i]) {
            printf(BOLD_RED "Mismatch at index %d: expected %d, got %d\n" RESET, i, sorted_data[i], data[i]);
            fflush(stdout);
            status = EXIT_FAILURE;
            break;
        }
    }

    free(data);
    free(sorted_data);
    return status;
}


int main(void) {

    int tests_passed = 0;
    int n_tests = 0;

    srand(0); // Seed for reproducibility

    if (print_cuda_device_info()) return EXIT_FAILURE;

    for (int v = VERSION_V0; v <= VERSION_V5; v++) {
        
        for (int q = Q_MIN; q <= Q_MAX; q++) {
            int descending = q % 2;
            n_tests++;

            if (test_case(q, descending, (bitonic_version_t)v) != EXIT_SUCCESS) {
                printf(BOLD_RED "Failed\n" RESET);
                fflush(stdout);
            }
            else {
                printf(BOLD_GREEN "Passed\n" RESET); 
                fflush(stdout);
                tests_passed++;
            }
        }
    }

    if (tests_passed < n_tests) {
        printf(BOLD_RED "\n====================\n");
        printf("%d out of %d tests failed.\n" RESET, n_tests - tests_passed, n_tests);
        fflush(stdout);
        return EXIT_FAILURE;
    }

    printf(BOLD_GREEN "\n====================\n");
    printf("All tests passed (%d/%d)\n" RESET, tests_passed, n_tests);
    fflush(stdout);
    return EXIT_SUCCESS;
}

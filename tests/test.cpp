#include <stdio.h>
#include <stdlib.h>
#include "bitonic_sort_cuda.cuh"

#define Q_MAX 20

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


int test_case(int q, int ascending, kernel_version_t version) {

    printf(BOLD_BLUE "Running test case with q = %d, ascending = %d, kernel version = %d ...\n" RESET, q, ascending, version - 1);
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

    qsort(sorted_data, n, sizeof(int), ascending ? cmp_asc : cmp_desc);
    if (bitonic_sort_cuda(data, n, ascending, version)) {
        free(data);
        free(sorted_data);
        return EXIT_FAILURE;
    }

    int status = EXIT_SUCCESS;

    // Print sorted array
    for (int i = 0; i < n; i++) {
        if (sorted_data[i] != data[i]) {
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

    for (int v = KERNEL_V0; v <= KERNEL_V2; v++) {
        
        for (int q = 1; q <= Q_MAX; q++) {
            int ascending = q % 2;
            n_tests++;

            if (test_case(q, ascending, (kernel_version_t)v) != EXIT_SUCCESS) {
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

#include <stdio.h>
#include <stdlib.h>
#include "bitonic_sort_cuda.cuh"

#define NTESTS 12

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


int test_case(int n, int ascending, kernel_version_t version) {

    printf(BOLD_BLUE "Running test case with n = %d, ascending = %d, version = %d ...\n" RESET, n, ascending, version);
    fflush(stdout);

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
    bitonic_sort_cuda(data, n, ascending, version);

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

    srand(0); // Seed for reproducibility
    
    for (int i = 10; i < NTESTS + 10; i++) {
        int n = 1 << i;
        int ascending = i % 2;
        kernel_version_t version = (kernel_version_t)(i % 3);

        if (test_case(n, ascending, version) != EXIT_SUCCESS) {
            printf(BOLD_RED "Failed\n" RESET); 
            fflush(stdout);
        }
        else {
            printf(BOLD_GREEN "Passed\n" RESET); 
            fflush(stdout);
            tests_passed++;
        }
    }

    if (tests_passed < NTESTS) {
        printf(BOLD_RED "%d out of %d tests failed.\n" RESET, NTESTS - tests_passed, NTESTS);
        fflush(stdout);
        return EXIT_FAILURE;
    }

    printf(BOLD_GREEN "All tests passed (%d/%d)\n" RESET, tests_passed, NTESTS);
    fflush(stdout);
    return EXIT_SUCCESS;
}

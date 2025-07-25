#include <stdio.h>
#include <stdlib.h>
#include "bitonic_sort.hpp"

#define NTESTS 12


#define BOLD_BLUE "\033[1;34m"
#define BOLD_GREEN "\033[1;32m"
#define BOLD_RED "\033[1;31m"
#define RESET "\033[0m"


int test_case(int n, int ascending, kernel_version_t version) {

    printf(BOLD_BLUE "Running test case with n = %d, ascending = %d, version = %d ...\n" RESET, n, ascending, version);
    fflush(stdout);

    int *data = (int *)malloc(n * sizeof(int));
    if (!data) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize the array with random values
    for (int i = 0; i < n; i++) {
        data[i] = rand();
    }
    
    if (bitonic_sort(data, n, ascending, version)) {
        fprintf(stderr, "Bitonic sort failed\n");
        free(data);
        return EXIT_FAILURE;
    }

    int status = EXIT_SUCCESS;

    // Print sorted array
    for (int i = 1; i < n; i++) {
        if ((ascending && data[i] < data[i - 1]) || (!ascending && data[i] > data[i - 1])) {
            status = EXIT_FAILURE;
            break;
        }
    }

    free(data);
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

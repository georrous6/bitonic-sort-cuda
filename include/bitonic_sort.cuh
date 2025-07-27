#ifndef BITONIC_SORT_H
#define BITONIC_SORT_H

#include <cuda_runtime.h>

typedef enum {
    KERNEL_NONE = 0,
    KERNEL_V0 = 1,
    KERNEL_V1 = 2,
    KERNEL_V2 = 3
} kernel_version_t;

__host__ int bitonic_sort(int *data, int n, int ascending, kernel_version_t version);


__host__ int wakeup_cuda(void);


#endif
#ifndef BITONIC_SORT_CUDA_H
#define BITONIC_SORT_CUDA_H

#include <cuda_runtime.h>

typedef enum {
    VERSION_SERIAL = 0,
    VERSION_V0 = 1,
    VERSION_V1 = 2,
    VERSION_V2 = 3,
    VERSION_V3 = 4,
    VERSION_V4 = 5,
    VERSION_V5 = 6
} bitonic_version_t;


__host__ int bitonic_sort_cuda(int *data, int n, int descending, bitonic_version_t version);


__host__ int wakeup_cuda(void);


__host__ int print_cuda_device_info(void);
#endif

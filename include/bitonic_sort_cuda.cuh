#ifndef BITONIC_SORT_CUDA_H
#define BITONIC_SORT_CUDA_H

#include <cuda_runtime.h>

typedef enum {
    VERSION_SERIAL = 0,
    VERSION_V0 = 1,
    VERSION_V1 = 2,
    VERSION_V2 = 3,
    VERSION_V3 = 4
} bitonic_version_t;


__host__ int bitonic_sort_cuda(int *data, int n, int ascending, bitonic_version_t version);


__host__ int wakeup_cuda(void);


__host__ int print_cuda_device_info(void);
#endif

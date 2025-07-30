#ifndef BITONIC_SORT_V2_CUH
#define BITONIC_SORT_V2_CUH

#include <cuda_runtime.h>

int bitonic_sort_v2(int *host_data, int n, int ascending);

#endif
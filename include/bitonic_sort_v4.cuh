#ifndef BITONIC_SORT_V4_CUH
#define BITONIC_SORT_V4_CUH

#include <cuda_runtime.h>


__host__
int bitonic_sort_v4(int *host_data, int n, int descending);

#endif
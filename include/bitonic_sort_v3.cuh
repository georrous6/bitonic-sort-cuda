#ifndef BITONIC_SORT_V3_CUH
#define BITONIC_SORT_V3_CUH

#include <cuda_runtime.h>


__host__
int bitonic_sort_v3(int *host_data, int n, int descending);

#endif
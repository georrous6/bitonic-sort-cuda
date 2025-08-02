#ifndef BITONIC_SORT_V5_CUH
#define BITONIC_SORT_V5_CUH

#include <cuda_runtime.h>


__host__
int bitonic_sort_v5(int *host_data, int n, int descending);

#endif
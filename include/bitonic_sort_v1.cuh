#ifndef BITONIC_SORT_V1_CUH
#define BITONIC_SORT_V1_CUH

#include <cuda_runtime.h>


__host__
int bitonic_sort_v1(int *host_data, int n, int ascending);


#endif // BITONIC_SORT_V1_CUH
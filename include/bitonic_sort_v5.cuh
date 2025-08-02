#ifndef BITONIC_SORT_V5_CUH
#define BITONIC_SORT_V5_CUH

#include <cuda_runtime.h>

namespace v5 {

__host__
int bitonic_sort(int *host_data, int n, int descending);

}

#endif
#ifndef BITONIC_SORT_V3_CUH
#define BITONIC_SORT_V3_CUH

#include <cuda_runtime.h>

namespace v3 {

__host__
int bitonic_sort(int *host_data, int n, int descending);

}

#endif
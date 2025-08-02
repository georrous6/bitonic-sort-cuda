#ifndef BITONIC_SORT_V2_CUH
#define BITONIC_SORT_V2_CUH

#include <cuda_runtime.h>

namespace v2 {

__host__
int bitonic_sort(int *host_data, int n, int descending);

}

#endif
#ifndef BITONIC_SORT_V1_CUH
#define BITONIC_SORT_V1_CUH

#include <cuda_runtime.h>

namespace v1 {

__host__
int bitonic_sort(int *host_data, int n, int descending);

}


#endif // BITONIC_SORT_V1_CUH
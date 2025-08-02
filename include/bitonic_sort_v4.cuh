#ifndef BITONIC_SORT_V4_CUH
#define BITONIC_SORT_V4_CUH

#include <cuda_runtime.h>

namespace v4 {

__host__
int bitonic_sort(int *host_data, int n, int descending);


__global__
void kernel_compare_and_swap(int *data, int size, int step, int log2step);

}

#endif
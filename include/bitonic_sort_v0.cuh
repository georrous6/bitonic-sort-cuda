#ifndef BITONIC_SORT_V0_CUH
#define BITONIC_SORT_V0_CUH
#include <cuda_runtime.h>

__host__
int bitonic_sort_v0(int *host_data, int n);


__global__ 
void kernel_compare_and_swap_v0(int *data, int n, int size, int step);

#endif
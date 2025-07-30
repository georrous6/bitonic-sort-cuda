#ifndef BITONIC_SORT_V2_CUH
#define BITONIC_SORT_V2_CUH

#include <cuda_runtime.h>


__host__
int bitonic_sort_v2(int *host_data, int n, int ascending);


__global__
void kernel_intra_block_sort_v2(int *data, int n, int ascending);


__global__
void kernel_intra_block_refine_v2(int *data, int n, int ascending, int size);

#endif
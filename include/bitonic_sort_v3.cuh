#ifndef BITONIC_SORT_V3_CUH
#define BITONIC_SORT_V3_CUH

#include <cuda_runtime.h>


__device__ __forceinline__
int get_lower_partner(int tid, int step, int log2step) {
    int blockPair = tid >> log2step;
    int offset    = tid & (step - 1);
    int base = blockPair << (log2step + 1);
    return base + offset;
}


__host__
int bitonic_sort_v3(int *host_data, int n, int descending);

#endif
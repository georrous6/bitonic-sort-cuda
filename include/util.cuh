#ifndef UTIL_CUH
#define UTIL_CUH

#include <cuda_runtime.h>

namespace util {

__device__ __forceinline__
void compare_and_swap(int *arr, int i, int j, int ascending) {
    if ((arr[i] > arr[j]) == ascending) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}


__device__ __forceinline__
int get_lower_partner(int tid, int step, int log2step) {
    int blockPair = tid >> log2step;
    int offset    = tid & (step - 1);
    int base = blockPair << (log2step + 1);
    return base + offset;
}


__global__
void kernel_reverse(int *data, int n);


__host__
int host_to_device_data(int *host_data, int n, int **device_data);


__host__
int device_to_host_data(int *host_data, int n, int *device_data);


__host__
int post_launch_barrier_and_check(void);

}

#endif

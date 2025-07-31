#ifndef UTIL_CUH
#define UTIL_CUH

#include <cuda_runtime.h>


__device__ __forceinline__
void compare_and_swap(int *arr, int i, int j, int ascending) {
    if ((arr[i] > arr[j]) == ascending) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}


__global__
void kernel_reverse(int *data, int n);


__host__
int host_to_device_data(int *host_data, int n, int **device_data);


__host__
int device_to_host_data(int *host_data, int n, int *device_data);


__host__
int post_launch_barrier_and_check(void);

#endif

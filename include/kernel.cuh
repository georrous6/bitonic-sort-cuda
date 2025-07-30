#ifndef KERNEL_CUH
#define KERNEL_CUH
#include <cuda_runtime.h>


__global__
void wakeup_kernel(void);


__device__ __forceinline__
void compare_and_swap(int *arr, int i, int j, int ascending) {
    if ((arr[i] > arr[j]) == ascending) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}


__global__ 
void kernel_v0_compare_and_swap(int *data, int n, int ascending, int size, int step);


__global__
void kernel_v1_intra_block_sort(int *data, int n, int ascending);


__global__
void kernel_v1_intra_block_refine(int *data, int n, int ascending, int size);


__global__
void kernel_v2_intra_block_sort(int *data, int n, int ascending);


__global__
void kernel_v2_intra_block_refine(int *data, int n, int ascending, int size);

#endif

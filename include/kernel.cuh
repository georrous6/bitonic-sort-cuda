#ifndef KERNEL_CUH
#define KERNEL_CUH
#include <cuda_runtime.h>


__global__
void wakeup_kernel(void);


__host__ __device__ __forceinline__
void compare_and_swap(int *arr, int i, int j, int ascending) {
    if ((ascending && arr[i] > arr[j]) || (!ascending && arr[i] < arr[j])) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}


__global__ 
void kernel_v0(int *data, int n, int ascending, int size, int step);


__global__
void kernel_v1_alternating_sort(int *data, int n, int chunk_size, int ascending);


__global__
void kernel_v1_intra_block_sort(int *data, int n, int chunk_size, int ascending, int step_start);

#endif

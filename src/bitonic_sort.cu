#include "bitonic_sort.hpp"
#include <stdio.h>
#include <cuda_runtime.h>

// Swap elements if needed, based on direction
static inline void compare_and_swap(int *arr, int i, int j, int ascending) {
    if ((ascending && arr[i] > arr[j]) || (!ascending && arr[i] < arr[j])) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}


__host__
static void bitonic_sort_serial(int *arr, int n, int ascending) {
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = 0; i < n; i++) {
                int j = i ^ stride;
                if (j > i) {
                    int is_ascending = ((i & size) == 0) ? ascending : !ascending;
                    compare_and_swap(arr, i, j, is_ascending);
                }
            }
        }
    }
}


__global__ 
void bitonic_sort_kernel_v0(int *data, int n, int ascending) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread index is within bounds
    if (idx < n) {
        // Perform bitonic sort logic here
        // This is a placeholder for the actual bitonic sort implementation
        // You would typically implement the bitonic sort algorithm here
    }
}


__host__ 
int bitonic_sort(int *data, int n, int ascending, kernel_version_t kernel_version) {

    // Allocate Unified Memory - accessible from CPU or GPU

    int *data_unified = NULL;
    cudaError_t err = cudaMallocManaged(&data_unified, n * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating unified memory: %s\n", cudaGetErrorString(err));
        fprintf(stderr, "Fallback to serial bitonic sort.\n");
        bitonic_sort_serial(data, n, ascending);
        return EXIT_SUCCESS;
    }
    
    // initialize x and y arrays on the host
    for (int i = 0; i < n; i++) {
        data_unified[i] = data[i];
    }


    // Launch the kernel with appropriate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    switch (kernel_version)
    {
    case KERNEL_V0:
        bitonic_sort_kernel_v0<<<blocksPerGrid, threadsPerBlock>>>(data_unified, n, ascending);
        break;
    case KERNEL_V1:
        // bitonic_sort_kernel_v1<<<blocksPerGrid, threadsPerBlock>>>(data_unified, n, ascending);
        break;
    case KERNEL_V2:
        // bitonic_sort_kernel_v2<<<blocksPerGrid, threadsPerBlock>>>(data_unified, n, ascending);
        break;
    default:
        // bitonic_sort_kernel_v2<<<blocksPerGrid, threadsPerBlock>>>(data_unified, n, ascending);
        fprintf(stderr, "Invalid kernel version specified.\n");
    }
    
    // Check for errors in kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(data_unified); // Free the unified memory before falling back
        fprintf(stderr, "Error launching bitonic sort kernel: %s\n", cudaGetErrorString(err));
        fprintf(stderr, "Fallback to serial bitonic sort.\n");
        bitonic_sort_serial(data, n, ascending);
        return EXIT_SUCCESS;
    }

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(data_unified); // Free the unified memory before falling back
        fprintf(stderr, "Error synchronizing device: %s\n", cudaGetErrorString(err));
        fprintf(stderr, "Fallback to serial bitonic sort.\n");
        bitonic_sort_serial(data, n, ascending);
        return EXIT_SUCCESS;
    }

    // Copy the sorted data back to the host
    for (int i = 0; i < n; i++) {
        data[i] = data_unified[i];
    }

    // Free the unified memory
    cudaFree(data_unified);

    return EXIT_SUCCESS;
}

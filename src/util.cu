#include "util.cuh"
#include <stdio.h>


namespace util {

__host__
int host_to_device_data(int *host_data, int n, int **device_data) {
    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc((void **)device_data, n * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory: %s\n", cudaGetErrorString(err));
        fflush(stderr);
        return EXIT_FAILURE;
    }

    // Copy data from host to device
    err = cudaMemcpy(*device_data, host_data, n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying data to device: %s\n", cudaGetErrorString(err));
        fflush(stderr);
        cudaFree(*device_data);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


__host__
int device_to_host_data(int *host_data, int n, int *device_data) {
    cudaError_t err;

    // Copy data from device to host
    err = cudaMemcpy(host_data, device_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying data back to host: %s\n", cudaGetErrorString(err));
        fflush(stderr);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


__host__
int post_launch_barrier_and_check(void) {
    cudaError_t err;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        fflush(stderr);
        return EXIT_FAILURE;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA synchronization error: %s\n", cudaGetErrorString(err));
        fflush(stderr);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


__global__
void kernel_reverse(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_half = n >> 1;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n_half; i += stride) {
        int temp = data[i];
        int opposite_idx = n - i - 1;
        data[i] = data[opposite_idx];
        data[opposite_idx] = temp;
    }
}

}

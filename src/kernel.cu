#include "kernel.cuh"

__global__
void wakeup_kernel(void) {}


__global__ 
void kernel_v0(int *data, int n, int ascending, int size, int step) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        int j = i ^ step;
        if (j > i) {
            int is_ascending = ((i & size) == 0) ? ascending : !ascending;
            compare_and_swap(data, i, j, is_ascending);
        }
    }
}

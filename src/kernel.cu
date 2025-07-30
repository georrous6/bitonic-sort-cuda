#include "kernel.cuh"

__global__
void wakeup_kernel(void) {}


__global__ 
void kernel_v0_compare_and_swap(int *data, int n, int ascending, int size, int step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        int j = i ^ step;
        if (j > i) {
            int is_asc = ((i & size) == 0) ? ascending : !ascending;
            compare_and_swap(data, i, j, is_asc);
        }
    }
}


__global__ void kernel_v1_intra_block_sort(int *data, int n, int chunk_size, int ascending) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int size = 2; size <= chunk_size; size <<= 1) {
        for (int step = size >> 1; step > 0; step >>= 1) {
            for (int i = idx; i < n; i += stride) {
                int j  = i ^ step;
                if (j > i) {
                    int is_asc = ((i & size) == 0) ? ascending : !ascending;
                    compare_and_swap(data, i, j, is_asc);
                }
            }
            __syncthreads();
        }
    }
}


__global__ 
void kernel_v1_intra_block_refine(int *data, int n, int chunk_size, int ascending, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int step = chunk_size >> 1; step > 0; step >>= 1) {
        for (int i = idx; i < n; i += stride) {
            int j  = i ^ step;
            if (j > i) {
                int is_asc = ((i & size) == 0) ? ascending : !ascending;
                compare_and_swap(data, i, j, is_asc);
            }
        }
        __syncthreads();
    }
}


__global__ 
void kernel_v2_intra_block_sort(int *data, int n, int chunk_size, int ascending) {
    
    extern __shared__ int s_data[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_start = blockIdx.x * chunk_size;
    int tid = threadIdx.x;

    if (idx >= n) return;

    // Load data into shared memory
    for (int i = tid; i < chunk_size; i += blockDim.x) {
        s_data[i] = data[block_start + i];
    }

    __syncthreads();

    for (int size = 2; size <= chunk_size; size <<= 1) {
        for (int step = size >> 1; step > 0; step >>= 1) {
            for (int i = tid; i < chunk_size; i += blockDim.x) {
                int j = i ^ step;
                if (j > i) {
                    int is_asc = ((i & size) == 0) ? ascending : !ascending;
                    compare_and_swap(s_data, i, j, is_asc);
                }
            }
            __syncthreads();
        }
    }

    // Copy data back to global memory
    for (int i = tid; i < chunk_size; i += blockDim.x) {
        data[block_start + i] = s_data[i];
    }
}


__global__
void kernel_v2_intra_block_refine(int *data, int n, int chunk_size, int ascending, int size) {
    extern __shared__ int s_data[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_start = blockIdx.x * chunk_size;
    int tid = threadIdx.x;

    if (idx >= n) return;

    // Load data into shared memory
    for (int i = tid; i < chunk_size; i += blockDim.x) {
        s_data[i] = data[block_start + i];
    }

    __syncthreads();

    for (int step = chunk_size >> 1; step > 0; step >>= 1) {
        for (int i = tid; i < chunk_size; i += blockDim.x) {
            int j  = i ^ step;
            if (j > i) {
                int is_asc = ((i & size) == 0) ? ascending : !ascending;
                compare_and_swap(s_data, i, j, is_asc);
            }
        }
        __syncthreads();
    }

    // Copy data back to global memory
    for (int i = tid; i < chunk_size; i += blockDim.x) {
        data[block_start + i] = s_data[i];
    }
}

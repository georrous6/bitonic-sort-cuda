#include "bitonic_sort_v4.cuh"
#include "util.cuh"
#include "config.cuh"
#include "bitonic_sort_v2.cuh"


// Device constexpr ffs replacement for log2
__device__ __forceinline__
int ctz_cuda(unsigned int x) {
    return __ffs(x) - 1;
}

// Warp-level bitonic merge using shuffle
__device__ __forceinline__
int warp_bitonic_shuffle(int val, int laneMask, bool dir) {
    int peer = __shfl_xor_sync(0xFFFFFFFF, val, laneMask);
    if ((val > peer) == dir) val = peer;
    return val;
}

__device__ __forceinline__
int warp_bitonic_merge(int val, int lane, int size, bool ascending) {
    int maxStep = min(size, WARP_SIZE);
    for (int step = maxStep >> 1; step > 0; step >>= 1) {
        bool dir = ((lane & size) == 0) ? ascending : !ascending;
        val = warp_bitonic_shuffle(val, step, dir);
    }
    return val;
}

// Compute partner index within array
__device__ __forceinline__
int get_lower_partner(int tid, int step, int log2step) {
    int blockPair = tid >> log2step;
    int offset    = tid & (step - 1);
    int base      = blockPair << (log2step + 1);
    return base + offset;
}

// Intra-block sort kernel v4 with shared + warp-shuffle
__global__
void kernel_intra_block_sort_v4(int *data, int max_size) {
    extern __shared__ int s_data[];
    int local_tid = threadIdx.x;
    int offset = 2 * blockIdx.x * blockDim.x;
    int lane = local_tid & (WARP_SIZE - 1);

    // Load two elements into registers
    int idx0 = offset + 2 * local_tid;
    int idx1 = idx0 + 1;
    int val0 = (idx0 < gridDim.x * 2 * blockDim.x * blockDim.x) ? data[idx0] : INT_MAX;
    int val1 = (idx1 < gridDim.x * 2 * blockDim.x * blockDim.x) ? data[idx1] : INT_MAX;

    // Store into shared memory
    s_data[2 * local_tid]     = val0;
    s_data[2 * local_tid + 1] = val1;
    __syncthreads();

    // Bitonic sort in shared memory for step > WARP_SIZE
    for (int size = 2; size <= max_size; size <<= 1) {
        // Shared-memory phases
        if (size > WARP_SIZE) {
            #pragma unroll
            for (int step = size >> 1; step > WARP_SIZE; step >>= 1) {
                int log2step = ctz_cuda(step);
                int i = get_lower_partner(local_tid, step, log2step) + 2*blockDim.x*blockIdx.x;
                int j = i + step;
                bool dir = ((i & size) == 0);
                compare_and_swap(s_data, i - offset, j - offset, dir);
                __syncthreads();
            }
        }
        // Warp-shuffle phase for last WARP_SIZE elements
        // Each thread handles one element in register
        int i_reg = 2 * local_tid;
        int val = s_data[i_reg];
        val = warp_bitonic_merge(val, lane, size, true);
        s_data[i_reg] = val;
        __syncthreads();
    }

    // Write back to global memory
    data[idx0] = s_data[2 * local_tid];
    data[idx1] = s_data[2 * local_tid + 1];
}

// Inter-block compare and swap unchanged
__global__
void kernel_compare_and_swap_v4(int *data, int size, int step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int log2step = ctz_cuda(step);
    int i = get_lower_partner(tid, step, log2step);
    int j = i + step;
    bool dir = ((i & size) == 0);
    compare_and_swap(data, i, j, dir);
}

// Intra-block refinement: similar pattern
__global__
void kernel_intra_block_refine_v4(int *data, int size, int max_size) {
    extern __shared__ int s_data[];
    int local_tid = threadIdx.x;
    int offset = 2 * blockIdx.x * blockDim.x;
    int lane = local_tid & (WARP_SIZE - 1);

    // Load into shared mem
    int idx0 = offset + 2 * local_tid;
    int idx1 = idx0 + 1;
    s_data[2*local_tid]   = data[idx0];
    s_data[2*local_tid+1] = data[idx1];
    __syncthreads();

    // Shared-memory phases
    if (max_size > WARP_SIZE) {
        #pragma unroll
        for (int step = max_size >> 1; step > WARP_SIZE; step >>= 1) {
            int log2step = ctz_cuda(step);
            int i = get_lower_partner(local_tid, step, log2step) + offset;
            int j = i + step;
            bool dir = ((i & size) == 0);
            compare_and_swap(s_data, i-offset, j-offset, dir);
            __syncthreads();
        }
    }
    // Warp-shuffle final
    int i_reg = 2 * local_tid;
    int val = s_data[i_reg];
    val = warp_bitonic_merge(val, lane, size, true);
    s_data[i_reg] = val;
    __syncthreads();

    // Write back
    data[idx0] = s_data[2*local_tid];
    data[idx1] = s_data[2*local_tid+1];
}

// Host wrapper
int bitonic_sort_v4(int *host_data, int n, int descending) {
    int n_half = n >> 1;
    int threadsPerBlock = min(BLOCK_SIZE, n_half);
    int numBlocks = (n_half + threadsPerBlock - 1) / threadsPerBlock;
    int max_size = threadsPerBlock << 1;
    size_t shared_bytes = max_size * sizeof(int);

    int *device_data;
    if (host_to_device_data(host_data, n, &device_data) != EXIT_SUCCESS) return EXIT_FAILURE;

    // Intra-block sort
    kernel_intra_block_sort_v4<<<numBlocks, threadsPerBlock, shared_bytes>>>(device_data, max_size);
    if (post_launch_barrier_and_check()) { cudaFree(device_data); return EXIT_FAILURE; }

    // Global merge
    for (int size = max_size << 1; size <= n; size <<= 1) {
        for (int step = size >> 1; step > 0; step >>= 1) {
            kernel_compare_and_swap_v4<<<numBlocks, threadsPerBlock>>>(device_data, size, step);
            if (post_launch_barrier_and_check()) { cudaFree(device_data); return EXIT_FAILURE; }
        }
    }

    if (descending) {
        kernel_reverse<<<numBlocks, BLOCK_SIZE>>>(device_data, n);
        if (post_launch_barrier_and_check()) { cudaFree(device_data); return EXIT_FAILURE; }
    }

    device_to_host_data(host_data, n, device_data);
    cudaFree(device_data);
    return EXIT_SUCCESS;
}

#ifndef UTIL_CUH
#define UTIL_CUH

#include <cuda_runtime.h>


__host__
int host_to_device_data(int *host_data, int n, int **device_data);


__host__
int device_to_host_data(int *host_data, int n, int *device_data);


__host__
int post_launch_barrier_and_check(void);

#endif

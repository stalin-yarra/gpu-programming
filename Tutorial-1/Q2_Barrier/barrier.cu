
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void barrierKernel()
{
    // Shared counter accessible by all threads in this block
    __shared__ int counter;

    // Thread 0 initializes the counter
    if (threadIdx.x == 0)
    {
        counter = 0;
    }

    // Make sure counter initialization is completed
    // before any thread uses it
    __syncthreads();

    // Simulate some work before the barrier
    if (threadIdx.x % 2 == 0)
    {
        for (volatile int i = 0; i < 1000000; i++);
    }

    // Every thread announces that it has reached the barrier
    atomicAdd(&counter, 1);

    // Wait until every thread in the block has reached
    // this synchronization point
    __syncthreads();

    // After the barrier, all threads should see
    // the complete counter value
    if (threadIdx.x == 0)
    {
        printf("All %d threads reached the barrier.\n", counter);
    }

    // Make sure all threads finish the barrier section
    // before the kernel continues
    __syncthreads();

    // Work after the barrier
    printf("Thread %d passed the barrier.\n", threadIdx.x);
}

int main()
{
    int threads = 8;

    printf("Launching kernel with %d threads...\n\n", threads);

    barrierKernel<<<1, threads>>>();

    cudaError_t error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
    {
        printf("CUDA Error: %s\n",
               cudaGetErrorString(error));
        return 1;
    }

    return 0;
}

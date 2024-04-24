#include <stdio.h>
#include <cuda_runtime.h>
#include <ctime>
#include <cstdlib> 

#define SIZE 1024  // Matrix dimension

// CUDA kernel for multiplying matrices on GPU
__global__ void gpuMatrixMultiply(int *matA, int *matB, int *result)
{
    int sum = 0;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < SIZE && col < SIZE)
    {
        for (int n = 0; n < SIZE; ++n)
            sum += matA[row * SIZE + n] * matB[n * SIZE + col];
        result[row * SIZE + col] = sum;
    }
}

int main()
{
    int *a, *b, *cpuResult, *gpuResult;

    size_t bytes = SIZE * SIZE * sizeof(int); 

    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&cpuResult, bytes);
    cudaMallocManaged(&gpuResult, bytes);

    srand(time(NULL));  // Seed the random number generator

    // Initialize matrices with random values between 1 and 100
    for(int i = 0; i < SIZE; i++)
        for(int j = 0; j < SIZE; j++)
        {
            a[i * SIZE + j] = rand() % 100 + 1;
            b[i * SIZE + j] = rand() % 100 + 1;
            cpuResult[i * SIZE + j] = 0;
            gpuResult[i * SIZE + j] = 0;
        }

    dim3 blocksPerGrid((SIZE + 15) / 16, (SIZE + 15) / 16, 1);
    dim3 threadsPerBlock(16, 16, 1);

    // Timing GPU operation
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent);
    gpuMatrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(a, b, gpuResult);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, startEvent, stopEvent);
    printf("GPU execution time: %f ms\n", gpuTime);

    cudaFree(a); cudaFree(b);
    cudaFree(cpuResult); cudaFree(gpuResult);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}

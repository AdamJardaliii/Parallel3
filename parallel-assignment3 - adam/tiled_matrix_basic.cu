#include <stdio.h>
#include <cuda_runtime.h>
#include <ctime>

#define N 1024
#define TILE_WIDTH 16

__global__ void matrixMulGPU_Tiled(int *a, int *b, int *c) {
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int temp = 0;

    __shared__ int tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ int tileB[TILE_WIDTH][TILE_WIDTH];

    for (int p = 0; p < (N / TILE_WIDTH); ++p) {
        tileA[threadIdx.y][threadIdx.x] = a[row * N + (p * TILE_WIDTH + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = b[(p * TILE_WIDTH + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            temp += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    c[row * N + col] = temp;
}

int main() {
    int *a, *b, *c_gpu_tiled;
    int size = N * N * sizeof(int);

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c_gpu_tiled, size);

    // Initialize matrices with random values between 1 and 100
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a[i * N + j] = rand() % 100 + 1;
            b[i * N + j] = rand() % 100 + 1;
            c_gpu_tiled[i * N + j] = 0; 
        }
    }

    dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH);
    dim3 num_blocks((N + threads_per_block.x - 1) / threads_per_block.x, 
                    (N + threads_per_block.y - 1) / threads_per_block.y);

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMulGPU_Tiled<<<num_blocks, threads_per_block>>>(a, b, c_gpu_tiled);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Tiled time: %f ms\n", milliseconds);

    cudaFree(a); cudaFree(b); cudaFree(c_gpu_tiled);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

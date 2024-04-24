#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>
#include <time.h>

#define N 1024
#define TILE_WIDTH 16

void matrixMulOpenACC_Tiled(int *a, int *b, int *c) {
    #pragma acc data copyin(a[0:N*N], b[0:N*N]) copyout(c[0:N*N])
    {
        #pragma acc parallel loop tile(TILE_WIDTH, TILE_WIDTH)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int sum = 0;
                for (int k = 0; k < N; k++) {
                    sum += a[i * N + k] * b[k * N + j];
                }
                c[i * N + j] = sum;
            }
        }
    }
}

int main() {
    int *a, *b, *c;
    size_t size = N * N * sizeof(int);

    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = rand() % 100 + 1;
            b[i * N + j] = rand() % 100 + 1;
            c[i * N + j] = 0; 
        }
    }

    clock_t start_time = clock();
    matrixMulOpenACC_Tiled(a, b, c);
    clock_t end_time = clock();
    double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("OpenACC Tiled Matrix Multiplication time: %f seconds\n", time_spent);

    free(a);
    free(b);
    free(c);

    return 0;
}

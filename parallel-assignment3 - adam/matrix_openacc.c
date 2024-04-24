#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>
#include <time.h>

#define N 1024  // Matrix dimension

// Function for matrix multiplication using OpenACC
void matrixMulOpenACC(int *a, int *b, int *result)
{
    
#pragma acc data copyin(a[0:N*N], b[0:N*N]) copyout(result[0:N*N])
    {
        
#pragma acc kernels
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int sum = 0;
                for (int k = 0; k < N; k++)
                {
                    sum += a[i * N + k] * b[k * N + j];
                }
                result[i * N + j] = sum;
            }
        }
    }
}

int main()
{
    int *a, *b, *result;
    size_t size = N * N * sizeof(int);

    a = (int *)malloc(size);
    b = (int *)malloc(size);
    result = (int *)malloc(size);
    srand(time(NULL));

    // Initialize matrices with random values between 1 and 100
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a[i * N + j] = rand() % 100 + 1;
            b[i * N + j] = rand() % 100 + 1;
            result[i * N + j] = 0; 
        }
    }

    clock_t start_acc = clock();
    matrixMulOpenACC(a, b, result);
    clock_t end_acc = clock();
    double acc_time_used = ((double)(end_acc - start_acc)) / CLOCKS_PER_SEC;

    printf("OpenACC Matrix Multiplication Time: %f seconds\n", acc_time_used);
    free(a);
    free(b);
    free(result);

    return 0;
}

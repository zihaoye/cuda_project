#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

void random_ints(int* arr, int N){
    srand(time(0));
    for(int i=0; i<N; i++){
        arr[i] = rand() % 100;
    }
}

__global__ void add(int *a, int *b, int *c, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < N) c[index] = a[index] + b[index];
}

int main(){
    int N = 512;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c; // device

    int size = sizeof(int) * N;

    // Allocate space for device copies of a, b, c
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Allocate host space
    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);

    // Memory copy
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(N/256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);

    add<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    //Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}
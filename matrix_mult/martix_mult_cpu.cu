#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>

#include "matrix_mult_gpu.cu"


// Helper function for reading input file
void readMatrixFile(const char* file, std::vector<float> &matrix, int &row, int &col){
    // TODO
}

int MatrixMult(int argc, int **argv, int block_size, const dim3 &dimsA,
                   const dim3 &dimsB){
    unsigned int sizeA = dimsA.x * dimsB.y;
    unsigned int mem_sizeA = sizeA * sizeof(float);
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
    float *h_A;


}
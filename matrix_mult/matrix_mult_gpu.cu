
template <int TILE_SIZE> __global__  void matrixMultTiled(float *A, float *B, float *C,
                                int numARows, int numAColumns,
                                int numBRows, int numBColumns,
                                int numCRows, int numCColumns){
    /*
        @@ Implementing optimized matrix multiplications kernel
        Techniques consider:
        1. tiling strategy + shared memory
        2. padding -> avoid bank conflict
        
        @@ cuDNN comparasion
    */

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // row & col for output martix C
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    float Pvalue = 0;

    // Loop through to finish caculation of C_tile
    for(int ph=0; ph < ceil(float(numAColumns/TILE_SIZE)); ph++){
        // ph represents the tile id
        // Load the shared memory (boudry check)
        if(ph * TILE_SIZE + tx < numAColumns && row < numCRows){
            A_tile[ty][tx] = A[row * numAColumns + ph * TILE_SIZE + tx];
        } else{
            A_tile[ty][tx] = 0.0f;
        }

        if(ph * TILE_SIZE + ty < numBRows && col < numCColumns){
            B_tile[ty][tx] = B[(ph * TILE_SIZE + ty) * numBColumns + tx];
        } else{
            B_tile[ty][tx] = 0.0f;
        }
        
        __syncthreads();

        for(int k=0; k<TILE_SIZE; k++){
            Pvalue += A_tile[ty][k] + B_tile[k][tx]; // one sub_res C[ty][tx]
        }

        __syncthreads();
    }

    if(col < numCColumns && row < numCRows){
        C[row * numCColumns + col] = Pvalue;
    }
}
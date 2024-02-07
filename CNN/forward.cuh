#define TILE_WIDTH 16

__constant__ float constKernel[24 * 12 * 7 * 7];

__global__ void forward_shared_const_unroll(float *__restrict__ y, const float *__restrict__ x,
                                            const int B, const int M, const int C, const int H, const int W, const int K,
                                            const int H_out, const int W_out, const int W_grid)
{
    /*
    W_grid = ceil(1.0 * W_out) / TILE_WIDTH
    H_grid = ceil(1.0 * H_out) / TILE_WIDTH

    blockIddx.z -> H_grid * W_grid
    */

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) constKernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int b = blockIdx.x;     // map batch
    int m = blockIdx.y * 4; // each thread handle 4 map

    int h = (blockIdx.z / W_grid) * TILE_WIDTH + ty;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + tx;

    float res[4] = {0.0};

    __shared__ smx[2 * TILE_WIDTH][2 * TILE_WIDTH]; // input tile

    for (int c = 0; c < C; c++){
        // Loading input tile
        if(h < H && w < W){
            smx[ty][tx] = x4d(b,c,h,w);
        }
        if(h < H && w + TILE_WIDTH < W){
            smx[ty][tx + TILE_WIDTH] = x4d(b,c,h,w + TILE_WIDTH);
        }

        // Computation
        if (h < H_out && w < W_out){
            for (int i = 0; i < 4; i++){
                int j = m + i;
                if (j < M){
                    for (int p = 0; p < k; p++){
                        for (int q = 0; q < K; q++){
                            res[i] += smx[threadIdx.y + p][threadIdx.x + q] * k4d(j, c, p, q);
                        }
                    }
                }
            }
        }
    }
}
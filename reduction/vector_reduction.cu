#define BLOCK_SIZE 512

extern "C" {
__global__ void total(float *input, float *output, int len) {
  //@@ Load a segment of the input vector into shared memory
  //@@ Traverse the reduction tree
  //@@ Write the computed sum of the block to the output vector at the
  //@@ correct index

  __shared__ int sdata[BLOCK_SIZE];

  // load element from global to shared memory (first step during global load)
  // each thread loads 2 elements to the shared memory
  // performs the first step of the reduction
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  if(i+blockDim.x < len){
    sdata[tid] = input[i] + input[i+blockDim.x];
  }
  else if(i < len){
    sdata[tid] = input[i];
  }
  else{
    sdata[tid] = 0;
  }
  __syncthreads();

  // Since the blockSize is 512, we unroll the kernel here:
  if(tid < 256) { sdata[tid] += sdata[tid+256]; } __syncthreads();
  if(tid < 128) { sdata[tid] += sdata[tid+128]; } __syncthreads();
  if(tid < 64) { sdata[tid] += sdata[tid+64]; } __syncthreads();
  if(tid < 32){
    sdata[tid] += sdata[tid + 32];
    __syncwarp();
    sdata[tid] += sdata[tid + 16];
    __syncwarp();
    sdata[tid] += sdata[tid + 8];
    __syncwarp();
    sdata[tid] += sdata[tid + 4];
    __syncwarp();
    sdata[tid] += sdata[tid + 2];
    __syncwarp();
    sdata[tid] += sdata[tid + 1];
  }

  if(tid == 0) output[blockIdx.x] = sdata[0];

}
}
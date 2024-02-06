import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import time

BLOCK_SIZE = 512

mod = drv.module_from_file("vector_reduction.ptx")
reduction = mod.get_function("total")

# Generate some large input data
numInputElements = int(1e6)
input_data = np.random.rand(numInputElements).astype(np.float32)

# Prepare the output data array
numOutputElements = numInputElements // (BLOCK_SIZE * 2)
if numInputElements % (BLOCK_SIZE * 2):
    numOutputElements += 1

output_data = np.zeros(numOutputElements, dtype=np.float32)

# Define block and grid sizes
block = (BLOCK_SIZE, 1, 1)
grid = (numInputElements // (BLOCK_SIZE * 2) + 1, 1)

# Run the kernel and time it
start_time = time.time()
reduction(drv.In(input_data), drv.Out(output_data), np.int32(numInputElements), block=block, grid=grid)
drv.Context.synchronize()  # Ensure completion

final_result = np.sum(output_data)
print("Final result by GPU: ", final_result)
end_time = time.time()

# Print running time
print(f"Running time for GPU: {end_time - start_time} seconds")


// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)
    
__global__ void scan(float * input, float * output, float *tempout, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
	
	__shared__ float xy[2*BLOCK_SIZE];
	int i=threadIdx.x + (blockDim.x*blockIdx.x*2);
	
	if(i<len)
		xy[threadIdx.x] = input[i];
	if(i+BLOCK_SIZE<len)
		xy[threadIdx.x+BLOCK_SIZE] = input[i+BLOCK_SIZE];
	__syncthreads();
	
	for(int stride=1; stride<= BLOCK_SIZE; stride*=2)
	{
		int index=(threadIdx.x +1)*stride*2 -1;
		if(index < 2*BLOCK_SIZE)
			xy[index] += xy[index-stride];
		__syncthreads();
	}
	
	for(int stride=BLOCK_SIZE/2; stride>0; stride/=2)
	{
		__syncthreads();
		int index=(threadIdx.x +1)*stride*2 -1;
		if(index+stride < 2*BLOCK_SIZE)
			xy[index+stride] += xy[index];
		__syncthreads();
	}
	
	if (i < len)
       output[i] = xy[threadIdx.x];
    if (i+ BLOCK_SIZE< len)
       output[i + BLOCK_SIZE] = xy[BLOCK_SIZE + threadIdx.x];
 
    if (len/(BLOCK_SIZE*2) > 1)
       tempout[blockIdx.x] = xy[2 * BLOCK_SIZE - 1];
}

__global__ void add(float *input, float *temp, int len) 
{
    int i=threadIdx.x + (blockDim.x*blockIdx.x*2);
    if (blockIdx.x)
	{
       if (i < len)
          input[i] += temp[blockIdx.x - 1];
       if (i + BLOCK_SIZE < len)
          input[i + BLOCK_SIZE] += temp[blockIdx.x - 1];
    }
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
	float *deviceTempOutput, *deviceTempScanned;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
	
	wbCheck(cudaMalloc((void**)&deviceTempOutput, BLOCK_SIZE*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceTempScanned, BLOCK_SIZE*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
	
	dim3 grid((numElements-1/(BLOCK_SIZE*2))+1,1,1);
	dim3 block(BLOCK_SIZE,1,1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
	
	scan<<<grid, block>>>(deviceInput, deviceOutput, deviceTempOutput, numElements);
	cudaDeviceSynchronize();
	
    scan<<<dim3(1,1,1), block>>>(deviceTempOutput, deviceTempScanned, NULL, BLOCK_SIZE);
    cudaDeviceSynchronize();
	
    add<<<grid, block>>>(deviceOutput, deviceTempScanned, numElements);
 
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}


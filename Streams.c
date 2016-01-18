#include	<wb.h>
#define SEG_SIZE 1024

__global__ void vecAdd(float * in1, float * in2, float * out, int len) 
{
    //@@ Insert code to implement vector addition here
	
	int i=blockDim.x*blockIdx.x + threadIdx.x;
	if(i<len)
	{
		out[i]=in1[i]+in2[i];
	}
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float *d_A0, *d_B0, *d_C0;
	float *d_A1, *d_B1, *d_C1;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
	
	cudaStream_t stream0, stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	
	
	cudaMalloc((void**)&d_A0, SEG_SIZE*sizeof(float));
    cudaMalloc((void**)&d_B0, SEG_SIZE*sizeof(float));
	cudaMalloc((void**)&d_C0, SEG_SIZE*sizeof(float));
	
	cudaMalloc((void**)&d_A1, SEG_SIZE*sizeof(float));
    cudaMalloc((void**)&d_B1, SEG_SIZE*sizeof(float));
	cudaMalloc((void**)&d_C1, SEG_SIZE*sizeof(float));
	
	
	for(int i=0;i<inputLength; i+=SEG_SIZE*2)
	{
		cudaMemcpyAsync(d_A0, hostInput1+i, SEG_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_B0, hostInput2+i, SEG_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream0);
		
		cudaMemcpyAsync(d_A1, hostInput1+SEG_SIZE+i, SEG_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(d_B1, hostInput2+SEG_SIZE+i, SEG_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream1);
		
		vecAdd<<<SEG_SIZE/256, 256, 0, stream0>>>(d_A0, d_B0, d_C0, inputLength);
		vecAdd<<<SEG_SIZE/256, 256, 0, stream1>>>(d_A1, d_B1, d_C1, inputLength);
		
		cudaMemcpyAsync(hostOutput+i, d_C0, SEG_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(hostOutput+SEG_SIZE+i, d_C1, SEG_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream1);
	}


    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}


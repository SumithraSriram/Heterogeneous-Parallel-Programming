// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here

__global__ void floattochar(float *im, unsigned char *imc)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	imc[j][i] = (unsigned char) (255 * im[j][i]);
}

__global__ void rgb2gray(unsigned char *im, unsigned char gray, int w)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int r = im[j][3*i]
    int g = im[j][3*i + 1]
    int b = im[j][3*i + 2]
    gray[(j*w)+i] = (unsigned char) (0.21*r + 0.71*g + 0.07*b)
}

__global__ void histogram(unsigned char *im, unsigned int *histo, int size)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int stride =blockDim.y * gridDim.x;
	
	while(i < size)
	{
		atomicAdd(&(histo[im[index]])),1);
		i+= stride;
	}
}

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
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;
	unsigned char *deviceinputchar;
	unsigned char *devicegray;
	unsigned int *devicehisto;
	
	float * deviceInputImageData;
    float * deviceOutputImageData;

    //@@ Insert more code here

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");

	hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **) &deviceinputchar, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
	cudaMalloc((void **) &devicegray, imageWidth * imageHeight * sizeof(unsigned char));
	cudaMalloc((void **) &devicehisto, imageWidth * imageHeight * sizeof(unsigned char));
	
	cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Doing GPU memory allocation");

	 wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
	
	floattochar<<<((imageHeight-1 )/16 )+1, (imageWidth * imageChannels -1)/16 +1, 1), (16,16,1)>>>(DeviceInputImage, deviceinputchar);
	rgb2gray<<<((imageHeight-1) /16  )+1, (imageWidth -1)/16 +1, 1), (16,16,1)>>>(deviceinputchar, devicegray, imageWidth);
	histogram<<<((imageWidth*imageHeight-1) /HISTOGRAM_LENGTH )+1, 1, 1), (HISTOGRAM_LENGTH,1,1)>>>(devicegray, devicehisto, (imageWidth*imageHeight) );
	
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);

    wbImage_delete(outputImage);
    wbImage_delete(inputImage);



    return 0;
}


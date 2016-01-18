#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2
#define O_Tile_Width 12
#define Block_Width (O_Tile_Width + Mask_width -1)
#define clamp(x) (min(max((x), 0.0), 1.0))
#define Channels 3


//@@ INSERT CODE HERE

__global__ void convolution(float *N, const float * __restrict__ M, float *O, int width, int height)
{
	
	__shared__ float ns[Block_Width][Block_Width][Channels];
	
	
	int tx= threadIdx.x;
	int ty = threadIdx.y;
	
	int row_o=blockIdx.y*O_Tile_Width +ty;
	int col_o=blockIdx.x*O_Tile_Width +tx;
	
	int row_i= row_o - Mask_radius;
	int col_i= col_o - Mask_radius;
	float out;
	
	for(int k=0;k<Channels;k++)
	{
	
	if( (row_i >=0) &&( row_i<height) && (col_i >=0) && (col_i < width))
	{
		ns[ty][tx][k] = N[(row_i * width + col_i)*Channels + k]; 
	}
	else
	{
		ns[ty][tx][k] = 0.0f;
	}
	
	__syncthreads();
	
	if( ty< O_Tile_Width && tx< O_Tile_Width )
	{
		out=0.0f;
		for(int i=0;i<Mask_width;i++)
		{
			for(int j=0;j<Mask_width;j++)
			{
				out += M[i * Mask_width + j] * ns[i + ty][j+tx][k];
			}
		}
	}
	
	__syncthreads();
	
	if((row_o < height) && (col_o < width) &&(tx < O_Tile_Width) && (ty < O_Tile_Width))
	{
		O[(row_o * width + col_o) * Channels + k] = clamp(out);
	}
		
	}
}
				

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
	
	dim3 block(Block_Width, Block_Width,1);
	dim3 grid( (imageWidth-1)/O_Tile_Width +1 , (imageHeight-1)/O_Tile_Width +1 , 1);
	
	convolution<<<grid, block>>>( deviceInputImageData, deviceMaskData, deviceOutputImageData, imageWidth, imageHeight);
	
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
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

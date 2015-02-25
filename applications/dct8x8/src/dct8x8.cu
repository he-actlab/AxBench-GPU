/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
**************************************************************************
* \file dct8x8.cu
* \brief Contains entry point, wrappers to host and device code and benchmark.
*
* This sample implements forward and inverse Discrete Cosine Transform to blocks
* of image pixels (of 8x8 size), as in JPEG standard. The typical work flow is as
* follows:
* 1. Run CPU version (Host code) and measure execution time;
* 2. Run CUDA version (Device code) and measure execution time;
* 3. Output execution timings and calculate CUDA speedup.
*/

#include "Common.h"
#include "DCT8x8_Gold.h"
#include "BmpUtil.h"

/**
*  The number of DCT kernel calls
*/
#define BENCHMARK_SIZE  10

/**
*  The PSNR values over this threshold indicate images equality
*/
#define PSNR_THRESHOLD_EQUAL    40


/**
*  Texture reference that is passed through this global variable into device code.
*  This is done because any conventional passing through argument list way results
*  in compiler internal error. 2008.03.11
*/
texture<float, 2, cudaReadModeElementType> TexSrc;


// includes kernels



/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
**************************************************************************
* \file dct8x8_kernel1.cu
* \brief Contains 1st CUDA implementations of DCT, IDCT and quantization routines,
*        used in JPEG internal data processing. Device code.
*
* This code implements first CUDA versions of forward and inverse Discrete Cosine
* Transform to blocks of image pixels (of 8x8 size), as in JPEG standard. The data
* processing is done using floating point representation.
* The routine that performs quantization of coefficients can be found in
* dct8x8_kernel_quantization.cu file.
*/
#include "Common.h"


/**
*  This unitary matrix performs discrete cosine transform of rows of the matrix to the left
*/
__constant__ float DCTv8matrix[] =
{
    0.3535533905932738f,  0.4903926402016152f,  0.4619397662556434f,  0.4157348061512726f,  0.3535533905932738f,  0.2777851165098011f,  0.1913417161825449f,  0.0975451610080642f,
    0.3535533905932738f,  0.4157348061512726f,  0.1913417161825449f, -0.0975451610080641f, -0.3535533905932737f, -0.4903926402016152f, -0.4619397662556434f, -0.2777851165098011f,
    0.3535533905932738f,  0.2777851165098011f, -0.1913417161825449f, -0.4903926402016152f, -0.3535533905932738f,  0.0975451610080642f,  0.4619397662556433f,  0.4157348061512727f,
    0.3535533905932738f,  0.0975451610080642f, -0.4619397662556434f, -0.2777851165098011f,  0.3535533905932737f,  0.4157348061512727f, -0.1913417161825450f, -0.4903926402016153f,
    0.3535533905932738f, -0.0975451610080641f, -0.4619397662556434f,  0.2777851165098009f,  0.3535533905932738f, -0.4157348061512726f, -0.1913417161825453f,  0.4903926402016152f,
    0.3535533905932738f, -0.2777851165098010f, -0.1913417161825452f,  0.4903926402016153f, -0.3535533905932733f, -0.0975451610080649f,  0.4619397662556437f, -0.4157348061512720f,
    0.3535533905932738f, -0.4157348061512727f,  0.1913417161825450f,  0.0975451610080640f, -0.3535533905932736f,  0.4903926402016152f, -0.4619397662556435f,  0.2777851165098022f,
    0.3535533905932738f, -0.4903926402016152f,  0.4619397662556433f, -0.4157348061512721f,  0.3535533905932733f, -0.2777851165098008f,  0.1913417161825431f, -0.0975451610080625f
};


// Temporary blocks
__shared__ float CurBlockLocal1[BLOCK_SIZE2];
__shared__ float CurBlockLocal2[BLOCK_SIZE2];


/**
**************************************************************************
*  Performs 1st implementation of 8x8 block-wise Forward Discrete Cosine Transform of the given
*  image plane and outputs result to the array of coefficients.
*
* \param Dst            [OUT] - Coefficients plane
* \param ImgWidth       [IN] - Stride of Dst
* \param OffsetXBlocks  [IN] - Offset along X in blocks from which to perform processing
* \param OffsetYBlocks  [IN] - Offset along Y in blocks from which to perform processing
*
* \return None
*/
__global__ void CUDAkernel1DCT(float *Dst, int ImgWidth, int OffsetXBlocks, int OffsetYBlocks)
{
    // Block index
    const int bx = blockIdx.x + OffsetXBlocks;
    const int by = blockIdx.y + OffsetYBlocks;

    // Thread index (current coefficient)
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Texture coordinates
    const float tex_x = (float)((bx << BLOCK_SIZE_LOG2) + tx) + 0.5f;
    const float tex_y = (float)((by << BLOCK_SIZE_LOG2) + ty) + 0.5f;

    //copy current image pixel to the first block
    CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ] = tex2D(TexSrc, tex_x, tex_y);

    // Amir
    float parrotInput[8];
    float parrotOutput[1];

    int firstIndex = 0 * BLOCK_SIZE + tx;
    for(int i = 0; i < BLOCK_SIZE; i++)
    {
        parrotInput[i] = CurBlockLocal1[firstIndex] / 256.0;
        firstIndex += BLOCK_SIZE;
    }
    // Rima

#pragma parrot(input, "CUDAkernel1DCT", [8]parrotInput)

    //synchronize threads to make sure the block is copied
    __syncthreads();

    //calculate the multiplication of DCTv8matrixT * A and place it in the second block
    float curelem = 0;
    int DCTv8matrixIndex = 0 * BLOCK_SIZE + ty;
    int CurBlockLocal1Index = 0 * BLOCK_SIZE + tx;
#pragma unroll

    for (int i=0; i<BLOCK_SIZE; i++)
    {
        curelem += DCTv8matrix[DCTv8matrixIndex] * CurBlockLocal1[CurBlockLocal1Index];
        DCTv8matrixIndex += BLOCK_SIZE;
        CurBlockLocal1Index += BLOCK_SIZE;
    }

    CurBlockLocal2[(ty << BLOCK_SIZE_LOG2) + tx ] = curelem;

    //synchronize threads to make sure the first 2 matrices are multiplied and the result is stored in the second block
    __syncthreads();

    //calculate the multiplication of (DCTv8matrixT * A) * DCTv8matrix and place it in the first block
    curelem = 0;
    int CurBlockLocal2Index = (ty << BLOCK_SIZE_LOG2) + 0;
    DCTv8matrixIndex = 0 * BLOCK_SIZE + tx;
#pragma unroll

    for (int i=0; i<BLOCK_SIZE; i++)
    {
        curelem += CurBlockLocal2[CurBlockLocal2Index] * DCTv8matrix[DCTv8matrixIndex];
        CurBlockLocal2Index += 1;
        DCTv8matrixIndex += BLOCK_SIZE;
    }

    CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ] = curelem;

    //synchronize threads to make sure the matrices are multiplied and the result is stored back in the first block
    __syncthreads();

    parrotOutput[0] = CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ] / 256.0;

#pragma parrot(output, "CUDAkernel1DCT", [1]<-2.0; 2.0>parrotOutput)

    CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ]  = 256.0 * parrotOutput[0];

    //copy current coefficient to its place in the result array
    Dst[ FMUL(((by << BLOCK_SIZE_LOG2) + ty), ImgWidth) + ((bx << BLOCK_SIZE_LOG2) + tx) ] = CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ];
}


/**
**************************************************************************
*  Performs 1st implementation of 8x8 block-wise Inverse Discrete Cosine Transform of the given
*  DCT coefficients plane and outputs result to the image array
*
* \param Dst            [OUT] - Image plane
* \param ImgWidth       [IN] - Stride of Dst
* \param OffsetXBlocks  [IN] - Offset along X in blocks from which to perform processing
* \param OffsetYBlocks  [IN] - Offset along Y in blocks from which to perform processing
*
* \return None
*/
__global__ void CUDAkernel1IDCT(float *Dst, int ImgWidth, int OffsetXBlocks, int OffsetYBlocks)
{
    // Block index
    int bx = blockIdx.x + OffsetXBlocks;
    int by = blockIdx.y + OffsetYBlocks;

    // Thread index (current image pixel)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Texture coordinates
    const float tex_x = (float)((bx << BLOCK_SIZE_LOG2) + tx) + 0.5f;
    const float tex_y = (float)((by << BLOCK_SIZE_LOG2) + ty) + 0.5f;

    //copy current image pixel to the first block
    CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ] = tex2D(TexSrc, tex_x, tex_y);

// Amir
    float parrotInput[8];
    float parrotOutput[1];

    int firstIndex = 0 * BLOCK_SIZE + tx;
    for(int i = 0; i < BLOCK_SIZE; i++)
    {
        parrotInput[i] = CurBlockLocal1[firstIndex] / 256.0;
        firstIndex += BLOCK_SIZE;
    }
    // Rima

#pragma parrot(input, "CUDAkernel1IDCT", [8]parrotInput)


    //synchronize threads to make sure the block is copied
    __syncthreads();

    //calculate the multiplication of DCTv8matrix * A and place it in the second block
    float curelem = 0;
    int DCTv8matrixIndex = (ty << BLOCK_SIZE_LOG2) + 0;
    int CurBlockLocal1Index = 0 * BLOCK_SIZE + tx;
#pragma unroll

    for (int i=0; i<BLOCK_SIZE; i++)
    {
        curelem += DCTv8matrix[DCTv8matrixIndex] * CurBlockLocal1[CurBlockLocal1Index];
        DCTv8matrixIndex += 1;
        CurBlockLocal1Index += BLOCK_SIZE;
    }

    CurBlockLocal2[(ty << BLOCK_SIZE_LOG2) + tx ] = curelem;

    //synchronize threads to make sure the first 2 matrices are multiplied and the result is stored in the second block
    __syncthreads();

    //calculate the multiplication of (DCTv8matrix * A) * DCTv8matrixT and place it in the first block
    curelem = 0;
    int CurBlockLocal2Index = (ty << BLOCK_SIZE_LOG2) + 0;
    DCTv8matrixIndex = (tx << BLOCK_SIZE_LOG2) + 0;
#pragma unroll

    for (int i=0; i<BLOCK_SIZE; i++)
    {
        curelem += CurBlockLocal2[CurBlockLocal2Index] * DCTv8matrix[DCTv8matrixIndex];
        CurBlockLocal2Index += 1;
        DCTv8matrixIndex += 1;
    }

    CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ] = curelem;

    //synchronize threads to make sure the matrices are multiplied and the result is stored back in the first block
    __syncthreads();

    parrotOutput[0] = CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ] / 256.0;

#pragma parrot(output, "CUDAkernel1IDCT", [1]<-2.0; 2.0>parrotOutput)

    CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ]  = 256.0 * parrotOutput[0];

    //copy current coefficient to its place in the result array
    Dst[ FMUL(((by << BLOCK_SIZE_LOG2) + ty), ImgWidth) + ((bx << BLOCK_SIZE_LOG2) + tx) ] = CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ];
}
//----------------------------------------------------

// #include "dct8x8_kernel2.cuh"
// #include "dct8x8_kernel_short.cuh"
#include "dct8x8_kernel_quantization.cuh"

/**
**************************************************************************
*  Wrapper function for 1st CUDA version of DCT, quantization and IDCT implementations
*
* \param ImgSrc         [IN] - Source byte image plane
* \param ImgDst         [IN] - Quantized result byte image plane
* \param Stride         [IN] - Stride for both source and result planes
* \param Size           [IN] - Size of both planes
*
* \return Execution time in milliseconds
*/
float WrapperCUDA1(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size)
{

#pragma parrot.start("CUDAkernel1DCT", "CUDAkernel1IDCT")
//#pragma parrot.start("CUDAkernel1DCT")

    //prepare channel format descriptor for passing texture into kernels
    cudaChannelFormatDesc floattex = cudaCreateChannelDesc<float>();

    //allocate device memory
    cudaArray *Src;
    float *Dst;
    size_t DstStride;
    checkCudaErrors(cudaMallocArray(&Src, &floattex, Size.width, Size.height));
    checkCudaErrors(cudaMallocPitch((void **)(&Dst), &DstStride, Size.width * sizeof(float), Size.height));
    DstStride /= sizeof(float);

    //convert source image to float representation
    int ImgSrcFStride;
    float *ImgSrcF = MallocPlaneFloat(Size.width, Size.height, &ImgSrcFStride);
    CopyByte2Float(ImgSrc, Stride, ImgSrcF, ImgSrcFStride, Size);
    AddFloatPlane(-128.0f, ImgSrcF, ImgSrcFStride, Size);

    //copy from host memory to device
    checkCudaErrors(cudaMemcpy2DToArray(Src, 0, 0,
                                        ImgSrcF, ImgSrcFStride * sizeof(float),
                                        Size.width * sizeof(float), Size.height,
                                        cudaMemcpyHostToDevice));

    //setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);

    //create and start CUDA timer
    StopWatchInterface *timerCUDA = 0;
    sdkCreateTimer(&timerCUDA);
    sdkResetTimer(&timerCUDA);

    //execute DCT kernel and benchmark
    checkCudaErrors(cudaBindTextureToArray(TexSrc, Src));

    for (int i=0; i<BENCHMARK_SIZE; i++)
    {
        sdkStartTimer(&timerCUDA);
        CUDAkernel1DCT<<< grid, threads >>>(Dst, (int) DstStride, 0, 0);
        checkCudaErrors(cudaDeviceSynchronize());
        sdkStopTimer(&timerCUDA);
    }

    checkCudaErrors(cudaUnbindTexture(TexSrc));
    getLastCudaError("Kernel execution failed");

    // finalize CUDA timer
    float TimerCUDASpan = sdkGetAverageTimerValue(&timerCUDA);
    sdkDeleteTimer(&timerCUDA);

    // execute Quantization kernel
    CUDAkernelQuantizationFloat<<< grid, threads >>>(Dst, (int) DstStride);
    getLastCudaError("Kernel execution failed");

    //copy quantized coefficients from host memory to device array
    checkCudaErrors(cudaMemcpy2DToArray(Src, 0, 0,
                                        Dst, DstStride *sizeof(float),
                                        Size.width *sizeof(float), Size.height,
                                        cudaMemcpyDeviceToDevice));

    // execute IDCT kernel
    checkCudaErrors(cudaBindTextureToArray(TexSrc, Src));
    CUDAkernel1IDCT<<< grid, threads >>>(Dst, (int) DstStride, 0, 0);
    checkCudaErrors(cudaUnbindTexture(TexSrc));
    getLastCudaError("Kernel execution failed");

    //copy quantized image block to host
    checkCudaErrors(cudaMemcpy2D(ImgSrcF, ImgSrcFStride *sizeof(float),
                                 Dst, DstStride *sizeof(float),
                                 Size.width *sizeof(float), Size.height,
                                 cudaMemcpyDeviceToHost));

    //convert image back to byte representation
    AddFloatPlane(128.0f, ImgSrcF, ImgSrcFStride, Size);
    CopyFloat2Byte(ImgSrcF, ImgSrcFStride, ImgDst, Stride, Size);

    //clean up memory
    checkCudaErrors(cudaFreeArray(Src));
    checkCudaErrors(cudaFree(Dst));
    FreePlane(ImgSrcF);



#pragma parrot.end("CUDAkernel1DCT", "CUDAkernel1IDCT")
//#pragma parrot.end("CUDAkernel1DCT")

    //return time taken by the operation
    return TimerCUDASpan;
}

/**
**************************************************************************
*  Program entry point
*
* \param argc       [IN] - Number of command-line arguments
* \param argv       [IN] - Array of command-line arguments
*
* \return Status code
*/


int main(int argc, char **argv)
{
    //
    // Sample initialization
    //
    printf("%s Starting...\n\n", argv[0]);

    char *pSampleImageFpath = argv[1];

    //preload image (acquire dimensions)
    int ImgWidth, ImgHeight;
    ROI ImgSize;
    int res = PreLoadBmp(pSampleImageFpath, &ImgWidth, &ImgHeight);
    ImgSize.width = ImgWidth;
    ImgSize.height = ImgHeight;

    //CONSOLE INFORMATION: saying hello to user
    printf("CUDA sample DCT/IDCT implementation\n");
    printf("===================================\n");
    printf("Loading test image: %s... ", pSampleImageFpath);

    if (res)
    {
        printf("\nError: Image file not found or invalid!\n");
        exit(EXIT_FAILURE);
        return 1;
    }

    //check image dimensions are multiples of BLOCK_SIZE
    if (ImgWidth % BLOCK_SIZE != 0 || ImgHeight % BLOCK_SIZE != 0)
    {
        printf("\nError: Input image dimensions must be multiples of 8!\n");
        exit(EXIT_FAILURE);
        return 1;
    }

    printf("[%d x %d]... ", ImgWidth, ImgHeight);

    //allocate image buffers
    int ImgStride;
    byte *ImgSrc = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
    byte *ImgDstCUDA1 = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);

    //load sample image
    LoadBmpAsGray(pSampleImageFpath, ImgStride, ImgSize, ImgSrc);

    //
    // RUNNING WRAPPERS
    //

    //compute CUDA 1 version of DCT/quantization/IDCT
    printf("Success\nRunning CUDA 1 (GPU) version... ");
    float TimeCUDA1 = WrapperCUDA1(ImgSrc, ImgDstCUDA1, ImgStride, ImgSize);

    //dump result of CUDA 1 processing
    printf("Success\nDumping result to %s... ", argv[2]);
    DumpBmpAsGray(argv[2], ImgDstCUDA1, ImgStride, ImgSize);


    float PSNR_Src_DstCUDA1      = CalculatePSNR(ImgSrc, ImgDstCUDA1, ImgStride, ImgSize);


    printf("PSNR Original    <---> GPU(CUDA 1)    : %f\n", PSNR_Src_DstCUDA1);

    //
    // Finalization
    //

    //release byte planes
    FreePlane(ImgSrc);
    FreePlane(ImgDstCUDA1);


    //finalize
    printf("\nTest Summary...\n");

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}

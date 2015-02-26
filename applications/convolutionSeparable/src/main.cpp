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

/*
* This sample implements a separable convolution filter
* of a 2D image with an arbitrary kernel.
*/

// CUDA runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

#include "convolutionSeparable_common.h"

////////////////////////////////////////////////////////////////////////////////
// Reference CPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionRowCPU(
    float *h_Result,
    float *h_Data,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);

extern "C" void convolutionColumnCPU(
    float *h_Result,
    float *h_Data,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);


// Amir
unsigned char *pixels = NULL;
// Rima

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // start logs
    //printf("[%s] - Starting...\n", argv[0]);

    float
    *h_Kernel,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPU;

    float
    *d_Input,
    *d_Output,
    *d_Buffer;


    //const int imageW = 3072;
    //const int imageH = 3072;
    //const int iterations = 16;

    // Amir
    const int imageW = 512;
    const int imageH = 512;
    const int iterations = 1;
    // Rima

    StopWatchInterface *hTimer = NULL;

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    //findCudaDevice(argc, (const char **)argv);

    sdkCreateTimer(&hTimer);

    //printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    //printf("Allocating and initializing host arrays...\n");
    h_Kernel    = (float *)malloc(KERNEL_LENGTH * sizeof(float));
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));
    srand(200);

    // Amir
    h_Kernel[0] = 1.0;
    h_Kernel[1] = 1.0;
    h_Kernel[2] = 1.0;
    h_Kernel[3] = 1.0;
    h_Kernel[4] = 1.0;
    h_Kernel[5] = 1.0;
    h_Kernel[6] = 1.0;
    h_Kernel[7] = 1.0;

    h_Kernel[8] = 0.0;

    h_Kernel[9]  = -1.0;
    h_Kernel[10] = -1.0;
    h_Kernel[11] = -1.0;
    h_Kernel[12] =  1.0;
    h_Kernel[13] =  1.0;
    h_Kernel[14] =  1.0;
    h_Kernel[15] =  1.0;
    h_Kernel[16] =  1.0;

    // Rima

    // for (unsigned int i = 0; i < KERNEL_LENGTH; i++)
    // {
    //     h_Kernel[i] = (float)(rand() % 16);
    // }


    // Amir
    unsigned int w;
    unsigned int h;
    sdkLoadPGM<unsigned char>(argv[1], &pixels, &w, &h); // PGM image
    // Rima

    for (unsigned i = 0; i < imageW * imageH; i++)
    {
        //h_Input[i] = (float)(rand() % 16);
        h_Input[i]  = (float) pixels[i]; // Amir
    }

    //printf("Allocating and initializing CUDA arrays...\n");
    checkCudaErrors(cudaMalloc((void **)&d_Input,   imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Output,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Buffer , imageW * imageH * sizeof(float)));

    setConvolutionKernel(h_Kernel);
    checkCudaErrors(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));

    //printf("Running GPU convolution (%u identical iterations)...\n\n", iterations);

    for (int i = -1; i < iterations; i++)
    {
        //i == -1 -- warmup iteration
        if (i == 0)
        {
            checkCudaErrors(cudaDeviceSynchronize());
            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
        }

        convolutionRowsGPU(
            d_Buffer,
            d_Input,
            imageW,
            imageH
        );

        convolutionColumnsGPU(
            d_Output,
            d_Buffer,
            imageW,
            imageH
        );
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double gpuTime = 0.001 * sdkGetTimerValue(&hTimer) / (double)iterations;
    //printf("convolutionSeparable, Throughput = %.4f MPixels/sec, Time = %.5f s, Size = %u Pixels, NumDevsUsed = %i, Workgroup = %u\n",
         //  (1.0e-6 * (double)(imageW * imageH)/ gpuTime), gpuTime, (imageW * imageH), 1, 0);

    //printf("\nReading back GPU results...\n\n");
    checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

    //printf("Checking the results...\n");
    //printf(" ...running convolutionRowCPU()\n");
    convolutionRowCPU(
        h_Buffer,
        h_Input,
        h_Kernel,
        imageW,
        imageH,
        KERNEL_RADIUS
    );

    //printf(" ...running convolutionColumnCPU()\n");
    convolutionColumnCPU(
        h_OutputCPU,
        h_Buffer,
        h_Kernel,
        imageW,
        imageH,
        KERNEL_RADIUS
    );

    //printf(" ...comparing the results\n");
    double sum = 0, delta = 0;

    for (unsigned i = 0; i < imageW * imageH; i++)
    {
        delta += (h_OutputGPU[i] - h_OutputCPU[i]) * (h_OutputGPU[i] - h_OutputCPU[i]);
        sum   += h_OutputCPU[i] * h_OutputCPU[i];
    }

    double L2norm = sqrt(delta / sum);
    printf("%2.2f", L2norm*100.0);
    //printf("Shutting down...\n");


    checkCudaErrors(cudaFree(d_Buffer));
    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFree(d_Input));
    free(h_OutputGPU);
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Kernel);

    sdkDeleteTimer(&hTimer);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    // if (L2norm > 1e-6)
    // {
    //     printf("Test failed!\n");
    //     exit(EXIT_FAILURE);
    // }

    // printf("Test passed\n");
    exit(EXIT_SUCCESS);
}

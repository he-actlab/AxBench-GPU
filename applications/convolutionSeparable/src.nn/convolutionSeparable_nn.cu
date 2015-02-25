#include "../../../headers/activationFunction.h"

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



#include <assert.h>
#include <helper_cuda.h>
#include "convolutionSeparable_common.h"



////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel[KERNEL_LENGTH];

extern "C" void setConvolutionKernel(float *h_Kernel)
{
    cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float));
}


////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 1

__global__ void convolutionRowsKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int pitch
)
{
    __shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Load main data
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
    }

    //Load left halo
#pragma unroll

    for (int i = 0; i < ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Load right halo
#pragma unroll

    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Compute and store results
    __syncthreads();
#pragma unroll

    // Amir
    float parrotInput[17];
    float parrotOutput[1];
    // Rima

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        float sum = 0;

#pragma unroll

        // Amir
        int parrotIndexLoop = 0;
        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            parrotInput[parrotIndexLoop++] = s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j] / 256.0;
        }
        // Rima

float layer_1_0 = parrotInput[0] * 36.021164 + parrotInput[1] * 35.539227 + parrotInput[2] * 16.319891 + parrotInput[3] * 16.663332 + parrotInput[4] * 13.312828 + parrotInput[5] * -3.753598 + parrotInput[6] * -15.971216 + parrotInput[7] * -74.571922 + parrotInput[8] * -0.774290 + parrotInput[9] * 15.760172 + parrotInput[10] * 4.064026 + parrotInput[11] * 16.833529 + parrotInput[12] * 15.842857 + parrotInput[13] * 13.380342 + parrotInput[14] * 13.063256 + parrotInput[15] * 14.531695 + parrotInput[16] * 14.838722 + 1.0f * 0.842725;

float layer_1_1 = parrotInput[0] * 36.032173 + parrotInput[1] * 35.585796 + parrotInput[2] * 16.316252 + parrotInput[3] * 16.648279 + parrotInput[4] * 13.123125 + parrotInput[5] * -5.539348 + parrotInput[6] * -8.066578 + parrotInput[7] * -55.881817 + parrotInput[8] * -0.206014 + parrotInput[9] * 15.957002 + parrotInput[10] * 5.600278 + parrotInput[11] * 17.247503 + parrotInput[12] * 29.633108 + parrotInput[13] * 14.054038 + parrotInput[14] * 13.668683 + parrotInput[15] * 21.892227 + parrotInput[16] * 20.818293 + 1.0f * 1.314986;

float layer_1_2 = parrotInput[0] * 1.351913 + parrotInput[1] * 1.338836 + parrotInput[2] * 1.670588 + parrotInput[3] * 1.364371 + parrotInput[4] * 1.288478 + parrotInput[5] * -1.227318 + parrotInput[6] * -1.533414 + parrotInput[7] * -1.238412 + parrotInput[8] * -0.032072 + parrotInput[9] * 1.342823 + parrotInput[10] * 1.305236 + parrotInput[11] * 1.541245 + parrotInput[12] * 1.273737 + parrotInput[13] * 1.558126 + parrotInput[14] * 1.194067 + parrotInput[15] * 1.586174 + parrotInput[16] * 1.445397 + 1.0f * -4.076314;

float layer_1_3 = parrotInput[0] * 1.074052 + parrotInput[1] * 0.949440 + parrotInput[2] * 0.969797 + parrotInput[3] * 1.053749 + parrotInput[4] * 1.031119 + parrotInput[5] * -1.045560 + parrotInput[6] * -1.034856 + parrotInput[7] * -1.015039 + parrotInput[8] * -0.018008 + parrotInput[9] * 1.060055 + parrotInput[10] * 1.030391 + parrotInput[11] * 1.086625 + parrotInput[12] * 0.902483 + parrotInput[13] * 1.099750 + parrotInput[14] * 0.958928 + parrotInput[15] * 1.057689 + parrotInput[16] * 0.943299 + 1.0f * -5.769458;

float layer_2_0 = sigmoid(layer_1_0, 0.500000) * 0.404556 + sigmoid(layer_1_1, 0.500000) * 1.269753 + sigmoid(layer_1_2, 0.500000) * 4.856860 + sigmoid(layer_1_3, 0.500000) * 7.633713 + 1.0f * 1.162027;

layer_2_0 = linear(layer_2_0, 0.5);

parrotOutput[0] = layer_2_0;

// parrotOutput[0] = layer_2_0;
// 
//         for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
//         {
//             sum += c_Kernel[KERNEL_RADIUS - j] * (s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j]);
//         }
// 
//         parrotOutput[0] = sum / 256.0;
// 
// #pragma parrot(output, "convolutionRowsKernel", [1]<0.0; 4.0>parrotOutput)

        sum = parrotOutput[0] * 256.0;

        d_Dst[i * ROWS_BLOCKDIM_X] = sum;
    }
}

extern "C" void convolutionRowsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH
)
{
    assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
    assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
    assert(imageH % ROWS_BLOCKDIM_Y == 0);

    dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

#pragma parrot.start("convolutionRowsKernel")

    convolutionRowsKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageW
    );

#pragma parrot.end("convolutionRowsKernel")
    getLastCudaError("convolutionRowsKernel() execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 8
#define   COLUMNS_HALO_STEPS 1

__global__ void convolutionColumnsKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int pitch
)
{
    __shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Main data
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

    //Lower halo
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

    //Compute and store results
    __syncthreads();
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        float sum = 0;
#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
        }

        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
    }
}

extern "C" void convolutionColumnsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH
)
{
    assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
    assert(imageW % COLUMNS_BLOCKDIM_X == 0);
    assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

    convolutionColumnsKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageW
    );
    getLastCudaError("convolutionColumnsKernel() execution failed\n");
}


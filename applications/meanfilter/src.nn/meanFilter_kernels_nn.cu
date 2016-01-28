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
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_string.h>

#include "meanFilter_kernels.h"

// Texture reference for reading image
texture<unsigned char, 2> tex;
extern __shared__ unsigned char LocalBlock[];
static cudaArray *array = NULL;

#define RADIUS 1

#ifdef FIXED_BLOCKWIDTH
#define BlockWidth 80
#define SharedPitch 384
#endif

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


__device__ unsigned char
ComputeMean(unsigned char ul, // upper left
             unsigned char um, // upper middle
             unsigned char ur, // upper right
             unsigned char ml, // middle left
             unsigned char mm, // middle (unused)
             unsigned char mr, // middle right
             unsigned char ll, // lower left
             unsigned char lm, // lower middle
             unsigned char lr, // lower right
             float fScale)
{

    float parrotInput[9];
    float parrotOutput[1];

    parrotInput[0] = (float) ul / 256.0;
    parrotInput[1] = (float) um / 256.0;
    parrotInput[2] = (float) ur / 256.0;
    parrotInput[3] = (float) ml / 256.0;
    parrotInput[4] = (float) mm / 256.0;
    parrotInput[5] = (float) mr / 256.0;
    parrotInput[6] = (float) ll / 256.0;
    parrotInput[7] = (float) lm / 256.0;
    parrotInput[8] = (float) lr / 256.0;
    short Sum = 0;

float layer_1_0 = parrotInput[0] * 0.873429 + parrotInput[1] * -0.111302 + parrotInput[2] * 0.824626 + parrotInput[3] * 0.461390 + parrotInput[4] * 0.408948 + parrotInput[5] * 0.450000 + parrotInput[6] * 0.665357 + parrotInput[7] * 0.500345 + parrotInput[8] * 0.397581 + 1.0f * -1.422081;

float layer_1_1 = parrotInput[0] * -0.416116 + parrotInput[1] * -0.569410 + parrotInput[2] * -0.446042 + parrotInput[3] * -0.650523 + parrotInput[4] * -0.604227 + parrotInput[5] * -0.621536 + parrotInput[6] * -0.462116 + parrotInput[7] * -0.556305 + parrotInput[8] * -0.665235 + 1.0f * 1.393419;

float layer_1_2 = parrotInput[0] * -0.540061 + parrotInput[1] * -2.055409 + parrotInput[2] * -0.513943 + parrotInput[3] * -0.422993 + parrotInput[4] * -0.887786 + parrotInput[5] * -0.533355 + parrotInput[6] * -0.543979 + parrotInput[7] * -0.433968 + parrotInput[8] * -0.517523 + 1.0f * 0.188695;

float layer_1_3 = parrotInput[0] * -0.799834 + parrotInput[1] * -1.847040 + parrotInput[2] * -0.914776 + parrotInput[3] * -0.880958 + parrotInput[4] * -0.924688 + parrotInput[5] * -0.893433 + parrotInput[6] * -1.023796 + parrotInput[7] * -1.256231 + parrotInput[8] * -1.000097 + 1.0f * 6.338632;

float layer_2_0 = sigmoid(layer_1_0, 0.500000) * 1.022724 + sigmoid(layer_1_1, 0.500000) * -2.023565 + sigmoid(layer_1_2, 0.500000) * -0.808142 + sigmoid(layer_1_3, 0.500000) * -0.759580 + 1.0f * 0.463564;

layer_2_0 = sigmoid(layer_2_0, 0.5);

parrotOutput[0] = layer_2_0;

// parrotOutput[0] = layer_2_0;
// 
//     //short Horz = ur + 2*mr + lr - ul - 2*ml - ll;
//     //short Vert = ul + 2*um + ur - ll - 2*lm - lr;
//     //Sum = (short)(fScale*(abs((int)Horz)+abs((int)Vert)));
//     Sum = (short)((ul + um + ur + ml + mm + mr + ll + lm + lr) / 9);
// 
//     if (Sum < 0)
//     {
//         Sum = 0;
//     }
//     else if (Sum > 0xff)
//     {
//         Sum = 0xff;
//     }
// 
//     parrotOutput[0] = Sum / 256.0;
// 
// #pragma parrot(output, "computeMean", [1]<0.0; 1.0>parrotOutput)

    Sum = parrotOutput[0] * 256.0;

    return (unsigned char) Sum;
}

__global__ void
MeanShared(uchar4 *pMeanOriginal, unsigned short MeanPitch,
#ifndef FIXED_BLOCKWIDTH
            short BlockWidth, short SharedPitch,
#endif
            short w, short h, float fScale)
{
    short u = 4*blockIdx.x*BlockWidth;
    short v = blockIdx.y*blockDim.y + threadIdx.y;
    short ib;

    int SharedIdx = threadIdx.y * SharedPitch;

    for (ib = threadIdx.x; ib < BlockWidth+2*RADIUS; ib += blockDim.x)
    {
        LocalBlock[SharedIdx+4*ib+0] = tex2D(tex,
                                             (float)(u+4*ib-RADIUS+0), (float)(v-RADIUS));
        LocalBlock[SharedIdx+4*ib+1] = tex2D(tex,
                                             (float)(u+4*ib-RADIUS+1), (float)(v-RADIUS));
        LocalBlock[SharedIdx+4*ib+2] = tex2D(tex,
                                             (float)(u+4*ib-RADIUS+2), (float)(v-RADIUS));
        LocalBlock[SharedIdx+4*ib+3] = tex2D(tex,
                                             (float)(u+4*ib-RADIUS+3), (float)(v-RADIUS));
    }

    if (threadIdx.y < RADIUS*2)
    {
        //
        // copy trailing RADIUS*2 rows of pixels into shared
        //
        SharedIdx = (blockDim.y+threadIdx.y) * SharedPitch;

        for (ib = threadIdx.x; ib < BlockWidth+2*RADIUS; ib += blockDim.x)
        {
            LocalBlock[SharedIdx+4*ib+0] = tex2D(tex,
                                                 (float)(u+4*ib-RADIUS+0), (float)(v+blockDim.y-RADIUS));
            LocalBlock[SharedIdx+4*ib+1] = tex2D(tex,
                                                 (float)(u+4*ib-RADIUS+1), (float)(v+blockDim.y-RADIUS));
            LocalBlock[SharedIdx+4*ib+2] = tex2D(tex,
                                                 (float)(u+4*ib-RADIUS+2), (float)(v+blockDim.y-RADIUS));
            LocalBlock[SharedIdx+4*ib+3] = tex2D(tex,
                                                 (float)(u+4*ib-RADIUS+3), (float)(v+blockDim.y-RADIUS));
        }
    }

    __syncthreads();

    u >>= 2;    // index as uchar4 from here
    uchar4 *pMean = (uchar4 *)(((char *) pMeanOriginal)+v*MeanPitch);
    SharedIdx = threadIdx.y * SharedPitch;

    for (ib = threadIdx.x; ib < BlockWidth; ib += blockDim.x)
    {

        unsigned char pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+0];
        unsigned char pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+1];
        unsigned char pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+2];
        unsigned char pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+0];
        unsigned char pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+1];
        unsigned char pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+2];
        unsigned char pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+0];
        unsigned char pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+1];
        unsigned char pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+2];

        uchar4 out;

        out.x = ComputeMean(pix00, pix01, pix02,
                             pix10, pix11, pix12,
                             pix20, pix21, pix22, fScale);

        pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+3];
        pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+3];
        pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+3];
        out.y = ComputeMean(pix01, pix02, pix00,
                             pix11, pix12, pix10,
                             pix21, pix22, pix20, fScale);

        pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+4];
        pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+4];
        pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+4];
        out.z = ComputeMean(pix02, pix00, pix01,
                             pix12, pix10, pix11,
                             pix22, pix20, pix21, fScale);

        pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+5];
        pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+5];
        pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+5];
        out.w = ComputeMean(pix00, pix01, pix02,
                             pix10, pix11, pix12,
                             pix20, pix21, pix22, fScale);

        if (u+ib < w/4 && v < h)
        {
            pMean[u+ib] = out;
        }
    }

    __syncthreads();
}

__global__ void
MeanCopyImage(Pixel *pMeanOriginal, unsigned int Pitch,
               int w, int h, float fscale)
{
    unsigned char *pMean =
        (unsigned char *)(((char *) pMeanOriginal)+blockIdx.x*Pitch);

    for (int i = threadIdx.x; i < w; i += blockDim.x)
    {
        pMean[i] = min(max((tex2D(tex, (float) i, (float) blockIdx.x) * fscale), 0.f), 255.f);
    }
}

__global__ void
MeanTex(Pixel *pMeanOriginal, unsigned int Pitch,
         int w, int h, float fScale)
{
    unsigned char *pMean =
        (unsigned char *)(((char *) pMeanOriginal)+blockIdx.x*Pitch);

    for (int i = threadIdx.x; i < w; i += blockDim.x)
    {
        unsigned char pix00 = tex2D(tex, (float) i-1, (float) blockIdx.x-1);
        unsigned char pix01 = tex2D(tex, (float) i+0, (float) blockIdx.x-1);
        unsigned char pix02 = tex2D(tex, (float) i+1, (float) blockIdx.x-1);
        unsigned char pix10 = tex2D(tex, (float) i-1, (float) blockIdx.x+0);
        unsigned char pix11 = tex2D(tex, (float) i+0, (float) blockIdx.x+0);
        unsigned char pix12 = tex2D(tex, (float) i+1, (float) blockIdx.x+0);
        unsigned char pix20 = tex2D(tex, (float) i-1, (float) blockIdx.x+1);
        unsigned char pix21 = tex2D(tex, (float) i+0, (float) blockIdx.x+1);
        unsigned char pix22 = tex2D(tex, (float) i+1, (float) blockIdx.x+1);
        pMean[i] = ComputeMean(pix00, pix01, pix02,
                                 pix10, pix11, pix12,
                                 pix20, pix21, pix22, fScale);
    }
}

extern "C" void setupTexture(int iw, int ih, Pixel *data, int Bpp)
{
    cudaChannelFormatDesc desc;

    if (Bpp == 1)
    {
        desc = cudaCreateChannelDesc<unsigned char>();
    }
    else
    {
        desc = cudaCreateChannelDesc<uchar4>();
    }

    checkCudaErrors(cudaMallocArray(&array, &desc, iw, ih));
    checkCudaErrors(cudaMemcpyToArray(array, 0, 0, data, Bpp*sizeof(Pixel)*iw*ih, cudaMemcpyHostToDevice));
}

extern "C" void deleteTexture(void)
{
    checkCudaErrors(cudaFreeArray(array));
}


// Wrapper for the __global__ call that sets up the texture and threads
extern "C" void MeanFilter(Pixel *odata, int iw, int ih, enum MeanDisplayMode mode, float fScale)
{
#pragma parrot.start("computeMean")

    //cudaStatus = cudaMemcpyToSymbol(dIndex, &hIndex, sizeof(int));

    checkCudaErrors(cudaBindTextureToArray(tex, array));

    switch (mode)
    {
        case MeanDISPLAY_IMAGE:
            MeanCopyImage<<<ih, 384>>>(odata, iw, iw, ih, fScale);
            break;

        case MeanDISPLAY_MeanTEX:
            MeanTex<<<ih, 384>>>(odata, iw, iw, ih, fScale);
            break;

        case MeanDISPLAY_MeanSHARED:
            {
                dim3 threads(16,4);
#ifndef FIXED_BLOCKWIDTH
                int BlockWidth = 80; // must be divisible by 16 for coalescing
#endif
                dim3 blocks = dim3(iw/(4*BlockWidth)+(0!=iw%(4*BlockWidth)),
                                   ih/threads.y+(0!=ih%threads.y));
                int SharedPitch = ~0x3f&(4*(BlockWidth+2*RADIUS)+0x3f);
                int sharedMem = SharedPitch*(threads.y+2*RADIUS);

                // for the shared kernel, width must be divisible by 4
                iw &= ~3;

                MeanShared<<<blocks, threads, sharedMem>>>((uchar4 *) odata,
                                                            iw,
#ifndef FIXED_BLOCKWIDTH
                                                            BlockWidth, SharedPitch,
#endif
                                                            iw, ih, fScale);
            }
            break;
    }

    checkCudaErrors(cudaUnbindTexture(tex));
    //cudaStatus = cudaMemcpyFromSymbol(hData, dData, SIZE * sizeof(float));
    //cudaStatus = cudaMemcpyFromSymbol(&hIndex, dIndex, sizeof(int));

#pragma parrot.end("computeMean")

}

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

#include "sobel-kernels.h"

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
ComputeSobel(unsigned char ul, // upper left
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

    short Sum = 0;

    // amir - observe inputs/outputs
    float parrotInput[9];
    float parrotOutput[1];

    parrotInput[0] = ul/255.0;
    parrotInput[1] = um/255.0;
    parrotInput[2] = ur/255.0;

    parrotInput[3] = ml/255.0;
    parrotInput[4] = mm/255.0;
    parrotInput[5] = mr/255.0;

    parrotInput[6] = ll/255.0;
    parrotInput[7] = lm/255.0;
    parrotInput[8] = lr/255.0;
    // rima

float layer_1_0 = parrotInput[0] * -0.191051 + parrotInput[1] * 5.621017 + parrotInput[2] * 7.995543 + parrotInput[3] * -1.598323 + parrotInput[4] * -0.290214 + parrotInput[5] * 3.349111 + parrotInput[6] * -13.368077 + parrotInput[7] * -0.402933 + parrotInput[8] * -1.122515 + 1.0f * -2.615931;

float layer_1_1 = parrotInput[0] * 0.279182 + parrotInput[1] * 0.361959 + parrotInput[2] * 6.572276 + parrotInput[3] * 0.413543 + parrotInput[4] * 0.348704 + parrotInput[5] * 1.923079 + parrotInput[6] * -8.722689 + parrotInput[7] * -0.528851 + parrotInput[8] * -1.847938 + 1.0f * 1.308268;

float layer_1_2 = parrotInput[0] * -7.564032 + parrotInput[1] * -21.111149 + parrotInput[2] * -1.159062 + parrotInput[3] * -1.654327 + parrotInput[4] * 0.228796 + parrotInput[5] * 4.454957 + parrotInput[6] * 4.947469 + parrotInput[7] * 7.080230 + parrotInput[8] * 15.945056 + 1.0f * 1.940841;

float layer_1_3 = parrotInput[0] * -0.005571 + parrotInput[1] * -6.269250 + parrotInput[2] * -0.585969 + parrotInput[3] * 0.484850 + parrotInput[4] * 0.331184 + parrotInput[5] * 0.030375 + parrotInput[6] * 2.231464 + parrotInput[7] * 0.838111 + parrotInput[8] * 2.725872 + 1.0f * -0.263775;

float layer_2_0 = sigmoid(layer_1_0, 0.500000) * 12.093943 + sigmoid(layer_1_1, 0.500000) * -3.555785 + sigmoid(layer_1_2, 0.500000) * -6.115889 + sigmoid(layer_1_3, 0.500000) * 13.702597 + 1.0f * -0.076973;

layer_2_0 = sigmoid(layer_2_0, 0.5);

parrotOutput[0] = layer_2_0;

// parrotOutput[0] = layer_2_0;
// 
//     short Horz = ur + 2*mr + lr - ul - 2*ml - ll;
//     short Vert = ul + 2*um + ur - ll - 2*lm - lr;
//     Sum = (short)(fScale*(abs((int)Horz)+abs((int)Vert)));
// 
//     if (Sum < 0)
//     {
//         Sum = 0;
//     }
//     else if (Sum > 0xff)
//     {
//         Sum = 0xff;
//     }
//     parrotOutput[0] = Sum/255.0;
// 
// #pragma parrot(output, "sobel", [1]<0.0; 1.0>parrotOutput)

    return (unsigned char) parrotOutput[0] * 255.0;
}

__global__ void
SobelShared(uchar4 *pSobelOriginal, unsigned short SobelPitch,
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
    uchar4 *pSobel = (uchar4 *)(((char *) pSobelOriginal)+v*SobelPitch);
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

        out.x = ComputeSobel(pix00, pix01, pix02,
                             pix10, pix11, pix12,
                             pix20, pix21, pix22, fScale);

        pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+3];
        pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+3];
        pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+3];
        out.y = ComputeSobel(pix01, pix02, pix00,
                             pix11, pix12, pix10,
                             pix21, pix22, pix20, fScale);

        pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+4];
        pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+4];
        pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+4];
        out.z = ComputeSobel(pix02, pix00, pix01,
                             pix12, pix10, pix11,
                             pix22, pix20, pix21, fScale);

        pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+5];
        pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+5];
        pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+5];
        out.w = ComputeSobel(pix00, pix01, pix02,
                             pix10, pix11, pix12,
                             pix20, pix21, pix22, fScale);

        if (u+ib < w/4 && v < h)
        {
            pSobel[u+ib] = out;
        }
    }

    __syncthreads();
}

__global__ void
SobelCopyImage(Pixel *pSobelOriginal, unsigned int Pitch,
               int w, int h, float fscale)
{
    unsigned char *pSobel =
        (unsigned char *)(((char *) pSobelOriginal)+blockIdx.x*Pitch);

    for (int i = threadIdx.x; i < w; i += blockDim.x)
    {
        pSobel[i] = min(max((tex2D(tex, (float) i, (float) blockIdx.x) * fscale), 0.f), 255.f);
    }
}

__global__ void
SobelTex(Pixel *pSobelOriginal, unsigned int Pitch,
         int w, int h, float fScale)
{
    unsigned char *pSobel =
        (unsigned char *)(((char *) pSobelOriginal)+blockIdx.x*Pitch);

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
        pSobel[i] = ComputeSobel(pix00, pix01, pix02,
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
extern "C" void sobelFilter(Pixel *odata, int iw, int ih, enum SobelDisplayMode mode, float fScale)
{

// amir - start approximable region
#pragma parrot.start("sobel")

    checkCudaErrors(cudaBindTextureToArray(tex, array));

    switch (mode)
    {
        case SOBELDISPLAY_IMAGE:
            SobelCopyImage<<<ih, 384>>>(odata, iw, iw, ih, fScale);
            break;

        case SOBELDISPLAY_SOBELTEX:
            SobelTex<<<ih, 384>>>(odata, iw, iw, ih, fScale);
            break;

        case SOBELDISPLAY_SOBELSHARED:
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

                SobelShared<<<blocks, threads, sharedMem>>>((uchar4 *) odata,
                                                            iw,
#ifndef FIXED_BLOCKWIDTH
                                                            BlockWidth, SharedPitch,
#endif
                                                            iw, ih, fScale);
            }
            break;
    }

    checkCudaErrors(cudaUnbindTexture(tex));
// rima - end approximable region
#pragma parrot.end("sobel")

}

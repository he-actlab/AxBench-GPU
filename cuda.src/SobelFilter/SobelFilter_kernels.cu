/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
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
#include <cuda_runtime_api.h>    // includes cuda.h and cuda_runtime_api.h

#include "SobelFilter_kernels.h"

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

inline void __checkCudaErrors( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
	    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}




// #ifdef NPU_SW

// __device__ float SigmoidCompute(float sumIn, float steepness)
// {
//     return (1/(1 + exp(-2.0*steepness*sumIn)));
// }


// // connected to neuron 10
// #define W0_0    -1.05274772644042968750e+01
// #define W0_1    -4.01711106300354003906e-01
// #define W0_2    1.62393264770507812500e+01
// #define W0_3    -3.89461669921875000000e+01
// #define W0_4    -5.32371473312377929688e+00
// #define W0_5    2.06709060668945312500e+01
// #define W0_6    -3.98109912872314453125e+00
// #define W0_7    6.74645423889160156250e+00
// #define W0_8    1.24631452560424804688e+01
// #define W0_9    1.09825592041015625000e+01

// // connected to neuron 11
// #define W1_0    1.01067628860473632812e+01
// #define W1_1    1.22519922256469726562e+01
// #define W1_2    2.23827342987060546875e+01
// #define W1_3    -1.27395610809326171875e+01
// #define W1_4    -1.45472073554992675781e+00
// #define W1_5    1.20425784587860107422e+00
// #define W1_6    -1.19039707183837890625e+01
// #define W1_7    -3.90514183044433593750e+00
// #define W1_8    -1.68651103973388671875e+01
// #define W1_9    2.05108809471130371094e+00

// // connected to neuron 12
// #define W2_0    -1.44402217864990234375e+00
// #define W2_1    -2.59088993072509765625e-01
// #define W2_2    1.59111385345458984375e+01
// #define W2_3    -2.61691837310791015625e+01
// #define W2_4    -7.27083325386047363281e-01
// #define W2_5    1.84614334106445312500e+01
// #define W2_6    -1.79996528625488281250e+01
// #define W2_7    8.82707297801971435547e-01
// #define W2_8    1.16309385299682617188e+01
// #define W2_9    1.57061350345611572266e+00

// // connected to neuron 13
// #define W3_0   1.62415099143981933594e+00 
// #define W3_1   4.77367639541625976562e+00
// #define W3_2   7.85900068283081054688e+00 
// #define W3_3   -3.69264793395996093750e+00 
// #define W3_4   -8.33369195461273193359e-01 
// #define W3_5   9.14693415164947509766e-01 
// #define W3_6   -1.03142404556274414062e+01 
// #define W3_7   -1.25234091281890869141e+00 
// #define W3_8   4.47446554899215698242e-01 
// #define W3_9   -8.65514576435089111328e-01 


// // connected to neuron 14
// #define W4_0   -7.62572526931762695312e+00
// #define W4_1   6.75057983398437500000e+00
// #define W4_2   6.35087251663208007812e+00
// #define W4_3   -1.17990770339965820312e+01
// #define W4_4   -1.01941728591918945312e+01
// #define W4_5   6.33560538291931152344e-01
// #define W4_6   -8.81372261047363281250e+00
// #define W4_7   2.34069395065307617188e+00
// #define W4_8   2.09181118011474609375e+01
// #define W4_9   -2.75569367408752441406e+00

// // connected to neuron 15
// #define W5_0   -1.17923583984375000000e+01
// #define W5_1   -5.62463521957397460938e+00
// #define W5_2   5.28839015960693359375e+00
// #define W5_3   3.18482041358947753906e+00
// #define W5_4   -3.05424270629882812500e+01
// #define W5_5   -3.58090400695800781250e+00
// #define W5_6   5.69021558761596679688e+00
// #define W5_7   3.54497413635253906250e+01
// #define W5_8   1.02873611450195312500e+01
// #define W5_9   1.40749084949493408203e+00

// // connected to neuron 16
// #define W6_0   -9.57335758209228515625e+00
// #define W6_1   -3.54479408264160156250e+00
// #define W6_2   1.35955131053924560547e+00
// #define W6_3   -1.17612047195434570312e+01
// #define W6_4   -8.43982279300689697266e-01
// #define W6_5   3.33366727828979492188e+00
// #define W6_6   8.33324372768402099609e-01
// #define W6_7   1.89971435070037841797e+00
// #define W6_8   1.81064224243164062500e+01
// #define W6_9   1.84060740470886230469e+00

// // connected to neuron 17
// #define W7_0   -2.29993915557861328125e+00
// #define W7_1   -5.84920823574066162109e-01
// #define W7_2   9.43587481975555419922e-01
// #define W7_3   -4.12832117080688476562e+00
// #define W7_4   -1.20650017261505126953e+00
// #define W7_5   1.81758499145507812500e+00
// #define W7_6   2.96122908592224121094e-01
// #define W7_7   1.03168737888336181641e+00
// #define W7_8   4.05338525772094726562e+00
// #define W7_9   -4.52693551778793334961e-01


// // connected to out
// #define W10_0  -4.24744367599487304688e+00
// #define W11_0  -2.23642325401306152344e+00 
// #define W12_0  -3.17701768875122070312e+00 
// #define W13_0  6.92118072509765625000e+00 
// #define W14_0  -4.69872379302978515625e+00 
// #define W15_0  -1.08694171905517578125e+00 
// #define W16_0  -6.79240322113037109375e+00 
// #define W17_0  2.22970027923583984375e+01 
// #define W18_0  3.02914524078369140625e+00

// __device__ unsigned char
// ComputeSobel(unsigned char ul_in, // upper left
//              unsigned char um_in, // upper middle
//              unsigned char ur_in, // upper right
//              unsigned char ml_in, // middle left
//              unsigned char mm_in, // middle (unused)
//              unsigned char mr_in, // middle right
//              unsigned char ll_in, // lower left
//              unsigned char lm_in, // lower middle
//              unsigned char lr_in, // lower right
//              float fScale )
// {


//     float ul = ul_in / 256.0;
//     float um = um_in / 256.0;
//     float ur = ur_in / 256.0;
//     float ml = ml_in / 256.0;
//     float mm = mm_in / 256.0;
//     float mr = mr_in / 256.0;
//     float ll = ll_in / 256.0;
//     float lm = lm_in / 256.0;
//     float lr = lr_in / 256.0;




//     float neuron10 =  SigmoidCompute( ul * W0_0 + um * W0_1 + ur * W0_2 + 
//                                 ml * W0_3 + mm * W0_4 + mr * W0_5 + 
//                                 ll * W0_6 + lm * W0_7 + lr * W0_8 + 
//                                 // bias
//                                 W0_9, 0.5);


//     float neuron11 =  SigmoidCompute( ul * W1_0 + um * W1_1 + ur * W1_2 + 
//                                 ml * W1_3 + mm * W1_4 + mr * W1_5 + 
//                                 ll * W1_6 + lm * W1_7 + lr * W1_8 + 
//                                 // bias
//                                 W1_9, 0.5);


//     float neuron12 =  SigmoidCompute( ul * W2_0 + um * W2_1 + ur * W2_2 + 
//                                 ml * W2_3 + mm * W2_4 + mr * W2_5 + 
//                                 ll * W2_6 + lm * W2_7 + lr * W2_8 + 
//                                 // bias
//                                 W2_9, 0.5);


//     float neuron13 =  SigmoidCompute( ul * W3_0 + um * W3_1 + ur * W3_2 + 
//                                 ml * W3_3 + mm * W3_4 + mr * W3_5 + 
//                                 ll * W3_6 + lm * W3_7 + lr * W3_8 + 
//                                 // bias
//                                 W3_9, 0.5);

//     float neuron14 =  SigmoidCompute( ul * W4_0 + um * W4_1 + ur * W4_2 + 
//                                 ml * W4_3 + mm * W4_4 + mr * W4_5 + 
//                                 ll * W4_6 + lm * W4_7 + lr * W4_8 + 
//                                 // bias
//                                 W4_9, 0.5);

//     float neuron15 =  SigmoidCompute( ul * W5_0 + um * W5_1 + ur * W5_2 + 
//                                 ml * W5_3 + mm * W5_4 + mr * W5_5 + 
//                                 ll * W5_6 + lm * W5_7 + lr * W5_8 + 
//                                 // bias
//                                 W5_9, 0.5);

//     float neuron16 =  SigmoidCompute( ul * W6_0 + um * W6_1 + ur * W6_2 + 
//                                 ml * W6_3 + mm * W6_4 + mr * W6_5 + 
//                                 ll * W6_6 + lm * W6_7 + lr * W6_8 + 
//                                 // bias
//                                 W6_9, 0.5);

//     float neuron17 =  SigmoidCompute( ul * W7_0 + um * W7_1 + ur * W7_2 + 
//                                 ml * W7_3 + mm * W7_4 + mr * W7_5 + 
//                                 ll * W7_6 + lm * W7_7 + lr * W7_8 + 
//                                 // bias
//                                 W7_9, 0.5);


//     float out      =  SigmoidCompute( neuron10 * W10_0 + 
//                                 neuron11 * W11_0 + 
//                                 neuron12 * W12_0 + 
//                                 neuron13 * W13_0 + 
//                                 neuron14 * W14_0 + 
//                                 neuron15 * W15_0 + 
//                                 neuron16 * W16_0 + 
//                                 neuron17 * W17_0 + 
//                                 //bias
//                                 W18_0, 0.5);

//     return (unsigned char) (out * 256.0);
// }
// #endif


// #ifdef NPU_OBSERVATION
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
             float fScale )
{



    short Horz = ur + 2*mr + lr - ul - 2*ml - ll;
    short Vert = ul + 2*um + ur - ll - 2*lm - lr;
    short Sum = (short) (fScale*(abs((int)Horz)+abs((int)Vert)));
    if ( Sum < 0 ) return 0; else if ( Sum > 0xff ) return 0xff;
    asm volatile ("mov.f32 %0, %0;" : "=f"(Sum));
    return (unsigned char) Sum;
}
//#endif

__global__ void 
SobelShared( uchar4 *pSobelOriginal, unsigned short SobelPitch, 
#ifndef FIXED_BLOCKWIDTH
             short BlockWidth, short SharedPitch,
#endif
             short w, short h, float fScale )
{ 
    short u = 4*blockIdx.x*BlockWidth;
    short v = blockIdx.y*blockDim.y + threadIdx.y;
    short ib;

    int SharedIdx = threadIdx.y * SharedPitch;

    for ( ib = threadIdx.x; ib < BlockWidth+2*RADIUS; ib += blockDim.x ) {
        LocalBlock[SharedIdx+4*ib+0] = tex2D( tex, 
            (float) (u+4*ib-RADIUS+0), (float) (v-RADIUS) );
        LocalBlock[SharedIdx+4*ib+1] = tex2D( tex, 
            (float) (u+4*ib-RADIUS+1), (float) (v-RADIUS) );
        LocalBlock[SharedIdx+4*ib+2] = tex2D( tex, 
            (float) (u+4*ib-RADIUS+2), (float) (v-RADIUS) );
        LocalBlock[SharedIdx+4*ib+3] = tex2D( tex, 
            (float) (u+4*ib-RADIUS+3), (float) (v-RADIUS) );
    }
    if ( threadIdx.y < RADIUS*2 ) {
        //
        // copy trailing RADIUS*2 rows of pixels into shared
        //
        SharedIdx = (blockDim.y+threadIdx.y) * SharedPitch;
        for ( ib = threadIdx.x; ib < BlockWidth+2*RADIUS; ib += blockDim.x ) {
            LocalBlock[SharedIdx+4*ib+0] = tex2D( tex, 
                (float) (u+4*ib-RADIUS+0), (float) (v+blockDim.y-RADIUS) );
            LocalBlock[SharedIdx+4*ib+1] = tex2D( tex, 
                (float) (u+4*ib-RADIUS+1), (float) (v+blockDim.y-RADIUS) );
            LocalBlock[SharedIdx+4*ib+2] = tex2D( tex, 
                (float) (u+4*ib-RADIUS+2), (float) (v+blockDim.y-RADIUS) );
            LocalBlock[SharedIdx+4*ib+3] = tex2D( tex, 
                (float) (u+4*ib-RADIUS+3), (float) (v+blockDim.y-RADIUS) );
        }
    }

    __syncthreads();

    u >>= 2;    // index as uchar4 from here
    uchar4 *pSobel = (uchar4 *) (((char *) pSobelOriginal)+v*SobelPitch);
    SharedIdx = threadIdx.y * SharedPitch;

    for ( ib = threadIdx.x; ib < BlockWidth; ib += blockDim.x ) {

        unsigned char pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+0];
        //int pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+0];
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
                             pix20, pix21, pix22, fScale );

        pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+3];
        pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+3];
        pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+3];
        out.y = ComputeSobel(pix01, pix02, pix00, 
                             pix11, pix12, pix10, 
                             pix21, pix22, pix20, fScale );

        pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+4];
        pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+4];
        pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+4];
        out.z = ComputeSobel( pix02, pix00, pix01, 
                              pix12, pix10, pix11, 
                              pix22, pix20, pix21, fScale );

        pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+5];
        pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+5];
        pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+5];
        out.w = ComputeSobel( pix00, pix01, pix02, 
                              pix10, pix11, pix12, 
                              pix20, pix21, pix22, fScale );
        if ( u+ib < w/4 && v < h ) {
            pSobel[u+ib] = out;
        }
    }

    __syncthreads();
}

__global__ void 
SobelCopyImage( Pixel *pSobelOriginal, unsigned int Pitch, 
                int w, int h, float fscale )
{ 
    unsigned char *pSobel = 
      (unsigned char *) (((char *) pSobelOriginal)+blockIdx.x*Pitch);
    for ( int i = threadIdx.x; i < w; i += blockDim.x ) {
        pSobel[i] = min( max((tex2D( tex, (float) i, (float) blockIdx.x ) * fscale), 0.f), 255.f);
    }
}

__global__ void 
SobelTex( Pixel *pSobelOriginal, unsigned int Pitch, 
          int w, int h, float fScale )
{ 
    unsigned char *pSobel = 
      (unsigned char *) (((char *) pSobelOriginal)+blockIdx.x*Pitch);
    for ( int i = threadIdx.x; i < w; i += blockDim.x ) {
        unsigned char pix00 = tex2D( tex, (float) i-1, (float) blockIdx.x-1 );
        unsigned char pix01 = tex2D( tex, (float) i+0, (float) blockIdx.x-1 );
        unsigned char pix02 = tex2D( tex, (float) i+1, (float) blockIdx.x-1 );
        unsigned char pix10 = tex2D( tex, (float) i-1, (float) blockIdx.x+0 );
        unsigned char pix11 = tex2D( tex, (float) i+0, (float) blockIdx.x+0 );
        unsigned char pix12 = tex2D( tex, (float) i+1, (float) blockIdx.x+0 );
        unsigned char pix20 = tex2D( tex, (float) i-1, (float) blockIdx.x+1 );
        unsigned char pix21 = tex2D( tex, (float) i+0, (float) blockIdx.x+1 );
        unsigned char pix22 = tex2D( tex, (float) i+1, (float) blockIdx.x+1 );
        pSobel[i] = ComputeSobel(pix00, pix01, pix02, 
                                 pix10, pix11, pix12,
                                 pix20, pix21, pix22, fScale );
    }
}

extern "C" void setupTexture(int iw, int ih, Pixel *data, int Bpp)
{
    cudaChannelFormatDesc desc;
    
    if (Bpp == 1) {
        desc = cudaCreateChannelDesc<unsigned char>();
    } else {
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
    checkCudaErrors(cudaBindTextureToArray(tex, array));

    switch ( mode ) {
        case  SOBELDISPLAY_IMAGE: 
            SobelCopyImage<<<ih, 384>>>(odata, iw, iw, ih, fScale );
            break;
        case SOBELDISPLAY_SOBELTEX:
            SobelTex<<<ih, 384>>>(odata, iw, iw, ih, fScale );
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
                                                		    iw, ih, fScale );
        }
        break;
    }

    checkCudaErrors(cudaUnbindTexture(tex));
}

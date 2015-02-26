#include "../../../headers/activationFunction.h"

/** @internal
 ** @file:       quickshift.cpp
 ** @author:     Brian Fulkerson
 ** @author:     Andrea Vedaldi
 ** @brief:      Quickshift command line
 **/

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include "quickshift_common.h"
#include <cutil_inline.h>

texture<float, 3, cudaReadModeElementType> texI;
texture<float, 2, cudaReadModeElementType> texE;

#define USE_TEX_E 1
#define USE_TEX_I 1

#if USE_TEX_I
  #define TEXI(x,y,c) tex3D(texI, x + 0.5f, y + 0.5f, c + 0.5f)
#else
  #define TEXI(x,y,c) I [ (x) + N1*(y) + N2*N1*k ]
#endif

#if USE_TEX_E
  #define TEXE(x,y) tex2D(texE, x + 0.5f, y + 0.5f)
#else
  #define TEXE(x,y) E [ (x) + N1* (y)]
#endif

#define distance(I,N1,N2,K,v,j1,j2,dist)      \
{                                             \
  dist = 0 ;                                  \
  int d1 = j1 - i1 ;                          \
  int d2 = j2 - i2 ;                          \
  int k ;                                     \
  dist += d1*d1 + d2*d2 ;                     \
  for (k = 0 ; k < K ; ++k) {                 \
    float d =  v[k] - TEXI(j1,j2,k);          \
    dist += d*d ;                             \
  }                                           \
}



__device__  float myDistance(const float * I, int K, float* v, int i1, int i2, int j1, int j2)
{
  float dist = 0.0;
  for (int k = 0; k < K ; ++k)
  {
    float d = v[k] - TEXI(j1, j2, k);
    dist += d*d;
  }
  return dist;
}


extern "C"
int iDivUp(int num, int denom)
{
  return (num % denom != 0) ? (num / denom + 1) : (num / denom);
}


extern "C"
__global__ void find_neighbors_gpu(const float * I, int N1, int N2, int K, float * E, float tau2, int tR, float * map, float * gaps)
{
  int i1 = blockIdx.y * blockDim.y + threadIdx.y;
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (i1 >= N1 || i2 >= N2) return; // out of bounds

  int j1,j2;

  /* Quickshift assigns each i to the closest j which has an increase in the
   * density (E). If there is no j s.t. Ej > Ei, then gaps_i == inf (a root
   * node in one of the trees of merges).
   */

  float E0 = TEXE(i1, i2) ;
  float d_best = INF ;
  float j1_best = i1   ;
  float j2_best = i2   ;

  int j1min = VL_MAX(i1 - tR, 0   ) ;
  int j1max = VL_MIN(i1 + tR, N1-1) ;
  int j2min = VL_MAX(i2 - tR, 0   ) ;
  int j2max = VL_MIN(i2 + tR, N2-1) ;

  /* Cache the center value in registers */
  float v[3];
  for (int k = 0 ; k < K ; ++k) {
    v[k] =  TEXI(i1,i2,k);
    }

  for (j2 = j2min ; j2 <= j2max ; ++ j2) {
    for (j1 = j1min ; j1 <= j1max ; ++ j1) {
      if (TEXE(j1,j2) > E0) {
        float Dij;
        distance(I,N1,N2,K, v, j1,j2,Dij) ;
        if (Dij <= tau2 && Dij < d_best) {
          d_best = Dij ;
          j1_best = j1 ;
          j2_best = j2 ;
        }
      }
    }
  }

  /* map is the index of the best pair */
  /* gaps_i is the minimal distance, inf implies no Ej > Ei within
   * distance tau from the point */
  map [i1 + N1 * i2] = j1_best + N1 * j2_best ; /* + 1 ; */
  if (map[i1 + N1 * i2] != i1 + N1 * i2)
    gaps[i1 + N1 * i2] = sqrt(d_best) ;
  else
    gaps[i1 + N1 * i2] = d_best; /* inf */
}

extern "C"
__global__ void compute_E_gpu(const float * I, int N1, int N2, int K, int R, float
    sigma, float * E, float * n, float * M)
{
  int i1 = blockIdx.y * blockDim.y + threadIdx.y;
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (i1 >= N1 || i2 >= N2) return; // out of bounds
  int j1,j2;

  /* -----------------------------------------------------------------
   *                                                 E = - [oN'*F]', M
   * -------------------------------------------------------------- */

  /*
     D_ij = d(x_i,x_j)
     E_ij = exp(- .5 * D_ij / sigma^2) ;
     F_ij = - E_ij
     E_i  = sum_j E_ij
     M_di = sum_j X_j F_ij

     E is the parzen window estimate of the density
     0 = dissimilar to everything, windowsize = identical
  */

  int j1min = VL_MAX(i1 - R, 0   ) ;
  int j1max = VL_MIN(i1 + R, N1-1) ;
  int j2min = VL_MAX(i2 - R, 0   ) ;
  int j2max = VL_MIN(i2 + R, N2-1) ;
  float Ei = 0;

  /* Cache the center value in registers */
  float v[3];
  for (int k = 0 ; k < K ; ++k) {
    v[k] =  TEXI(i1,i2,k);
    }


  // Amir
  float parrotInput[6];
  float parrotOutput[1];
  // Rima

  /* For each pixel in the window compute the distance between it and the
   * source pixel */
  for (j2 = j2min ; j2 <= j2max ; ++ j2) {
    for (j1 = j1min ; j1 <= j1max ; ++ j1) {
      float Dij;


      //distance(I, N1, N2, K,v ,j1, j2, Dij) ;
      parrotInput[0] = v[0];
      parrotInput[1] = v[1];
      parrotInput[2] = v[2];
      parrotInput[3] = TEXI(j1,j2,0);
      parrotInput[4] = TEXI(j1,j2,1);
      parrotInput[5] = TEXI(j1,j2,2);
      float Fij;

float layer_1_0 = parrotInput[0] * -1.522834 + parrotInput[1] * -0.662261 + parrotInput[2] * -1500.000000 + parrotInput[3] * 1500.000000 + parrotInput[4] * 1500.000000 + parrotInput[5] * 209.477997 + 1.0f * -0.517169;

float layer_1_1 = parrotInput[0] * -0.387868 + parrotInput[1] * -0.369250 + parrotInput[2] * 0.427768 + parrotInput[3] * 0.192909 + parrotInput[4] * -0.445800 + parrotInput[5] * 0.655006 + 1.0f * -1.834540;

float layer_1_2 = parrotInput[0] * 136.454300 + parrotInput[1] * -2.051326 + parrotInput[2] * 1500.000000 + parrotInput[3] * -135.596634 + parrotInput[4] * -1442.484253 + parrotInput[5] * -122.538925 + 1.0f * -1.537653;

float layer_1_3 = parrotInput[0] * -0.296651 + parrotInput[1] * 0.333198 + parrotInput[2] * -0.753976 + parrotInput[3] * -0.367202 + parrotInput[4] * -0.499004 + parrotInput[5] * 1.655033 + 1.0f * -2.038385;

float layer_2_0 = sigmoid(layer_1_0, 0.500000) * 0.001715 + sigmoid(layer_1_1, 0.500000) * 0.146615 + sigmoid(layer_1_2, 0.500000) * -0.007966 + sigmoid(layer_1_3, 0.500000) * -0.142282 + 1.0f * 0.011396;

layer_2_0 = linear(layer_2_0, 0.5);

parrotOutput[0] = layer_2_0;

// parrotOutput[0] = layer_2_0;
// 
//       Dij = myDistance(I, K, v, i1, i2, j1, j2);
//       int d1 = j1 - i1;
//       int d2 = j2 - i2;
//       Dij += d1 * d1 + d2 * d2;
//       //distance(I,N1,N2,K, v, j1,j2,Dij) ;
// 
//       /* Make distance a similarity */
//       Fij = - exp(- Dij / (2*sigma*sigma)) ;
// 
//       parrotOutput[0] =  - Fij;
// 
// #pragma parrot(output, "compute_E_gpu", [1]<-1.0; 1.0>parrotOutput)

      Fij =  - parrotOutput[0];

      /* E is E_i above */
      Ei += -Fij;

    } /* j1 */
  } /* j2 */
  /* Normalize */
  E [i1 + N1 * i2] = Ei / ((j1max-j1min)*(j2max-j2min));
}


extern "C"
void quickshift_gpu(image_t im, float sigma, float tau, float * map, float * gaps, float * E)
{
#if USE_TEX_I
  //printf("quickshiftGPU: using texture for I\n");
  cudaArray * cu_array_I;

  // Allocate array
  cudaChannelFormatDesc descriptionI = cudaCreateChannelDesc<float>();

  cudaExtent const ext = {im.N1, im.N2, im.K};
  cudaMalloc3DArray(&cu_array_I, &descriptionI, ext);

  cudaMemcpy3DParms copyParams = {0};
  copyParams.extent = make_cudaExtent(im.N1, im.N2, im.K);
  copyParams.kind = cudaMemcpyHostToDevice;
  copyParams.dstArray = cu_array_I;
  // The pitched pointer is really tricky to get right. We give the
  // pitch of a row, then the number of elements in a row, then the
  // height, and we omit the 3rd dimension.
  copyParams.srcPtr = make_cudaPitchedPtr(
  (void*)&im.I[0], ext.width*sizeof(float), ext.width, ext.height);
  CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));

  CUDA_SAFE_CALL(cudaBindTextureToArray(texI, cu_array_I,
        descriptionI));

  texI.normalized = false;
  texI.filterMode = cudaFilterModePoint;
#endif


  float *map_d, *E_d, *gaps_d, *I;

  int verb = 1 ;

  float tau2;

  int K;
  int N1,N2, R, tR;

  N1 = im.N1;
  N2 = im.N2;
  K = im.K;

  //d = 2 + K ; /* Total dimensions include spatial component (x,y) */

  tau2  = tau*tau;

  unsigned int size = im.N1*im.N2 * sizeof(float);
  cutilSafeCall( cudaMalloc( (void**) &I, size*im.K));
  cutilSafeCall( cudaMalloc( (void**) &map_d, size));
  cutilSafeCall( cudaMalloc( (void**) &gaps_d, size));
  cutilSafeCall( cudaMalloc( (void**) &E_d, size));

  cutilSafeCall( cudaMemcpy( I, im.I, size*im.K, cudaMemcpyHostToDevice));
  cutilSafeCall( cudaMemset( E_d, 0, size));

  R = (int) ceil (3 * sigma) ;
  tR = (int) ceil (tau) ;

  if (verb) {
    //printf("quickshiftGPU: [N1,N2,K]: [%d,%d,%d]\n", N1,N2,K) ;
    //printf("quickshiftGPU: type: quick\n");
    //printf("quickshiftGPU: sigma:   %g\n", sigma) ;
    /* R is ceil(3 * sigma) and determines the window size to accumulate
     * similarity */
    //printf("quickshiftGPU: R:       %d\n", R) ;
    //printf("quickshiftGPU: tau:     %g\n", tau) ;
    //printf("quickshiftGPU: tR:      %d\n", tR) ;
  }

  unsigned int Etimer;
  cutilCheckError( cutCreateTimer(&Etimer) );
  cutilCheckError( cutResetTimer(Etimer) );
  cutilCheckError( cutStartTimer(Etimer) );

  dim3 dimBlock(32,4,1);
  dim3 dimGrid(iDivUp(N2, dimBlock.x), iDivUp(N1, dimBlock.y), 1);

#pragma parrot.start("compute_E_gpu")

  compute_E_gpu <<<dimGrid,dimBlock>>> (I, N1, N2, K, R, sigma, E_d, 0, 0);

#pragma parrot.end("compute_E_gpu")

  cutilSafeCall( cudaThreadSynchronize() );
  // check if kernel execution generated an error
  cutilCheckMsg("Kernel execution failed");

  cutilSafeCall( cudaMemcpy(E, E_d, size, cudaMemcpyDeviceToHost));

  cutilCheckError( cutStopTimer(Etimer) );
  float ETime = cutGetTimerValue(Etimer);
  //printf("ComputeE: %fms\n", ETime);

  unsigned int Ntimer;
  cutilCheckError( cutCreateTimer(&Ntimer) );
  cutilCheckError( cutResetTimer(Ntimer) );
  cutilCheckError( cutStartTimer(Ntimer) );

  /* Texture map E */
#if USE_TEX_E
  //printf("quickshiftGPU: using texture for E\n");
  cudaChannelFormatDesc descriptionE = cudaCreateChannelDesc<float>();

  cudaArray * cu_array_E;
  cudaMallocArray(&cu_array_E, &descriptionE, im.N1, im.N2);

  CUDA_SAFE_CALL(cudaMemcpyToArray(cu_array_E, 0, 0, E,
        sizeof(float)*im.N1*im.N2, cudaMemcpyHostToDevice));

  texE.normalized = false;
  texE.filterMode = cudaFilterModePoint;

  CUDA_SAFE_CALL( cudaBindTextureToArray(texE, cu_array_E,
        descriptionE));
  cutilCheckMsg("Texture setup failed");

  cutilSafeCall( cudaThreadSynchronize() );
#endif

  /* -----------------------------------------------------------------
   *                                               Find best neighbors
   * -------------------------------------------------------------- */

  find_neighbors_gpu <<<dimGrid,dimBlock>>> (I, N1 ,N2, K, E_d, tau2,
      tR, map_d, gaps_d);

  cutilSafeCall( cudaThreadSynchronize() );
  // check if kernel execution generated an error
  cutilCheckMsg("Kernel execution failed");

  cudaMemcpy(map, map_d, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(gaps, gaps_d, size, cudaMemcpyDeviceToHost);

  cutilCheckError( cutStopTimer(Ntimer) );
  float NTime = cutGetTimerValue(Ntimer);
  //printf("ComputeN: %fms\n", NTime);
  //printf("dimGrid: %d %d\n", dimGrid.x, dimGrid.y);
  //printf("dimBlock: %d %d\n", dimBlock.x, dimBlock.y);

  cutilSafeCall(cudaFree(I));
  cutilSafeCall(cudaFree(map_d));
  cutilSafeCall(cudaFree(gaps_d));
  cutilSafeCall(cudaFree(E_d));

}

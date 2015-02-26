#ifndef __NPUKERNEL_H_
#define __NPUKERNEL_H_

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>

#ifdef NPU
__device__ unsigned char
ComputeSobel(unsigned char ul_in, // upper left
			 unsigned char um_in, // upper middle
			 unsigned char ur_in, // upper right
			 unsigned char ml_in, // middle left
			 unsigned char mm_in, // middle (unused)
			 unsigned char mr_in, // middle right
			 unsigned char ll_in, // lower left
			 unsigned char lm_in, // lower middle
			 unsigned char lr_in, // lower right
			 float fScale);

__device__ float af0(float, float);
__device__ float af1(float, float);
__device__ float af2(float, float);
__device__ float af3(float, float);
__device__ float af4(float, float);
__device__ float af5(float, float);
__device__ float af6(float, float);
__device__ float af7(float, float);
__device__ float af8(float, float);
__device__ float af9(float, float);
__device__ float af10(float, float);
__device__ float af11(float, float);
__device__ float af12(float, float);
__device__ float af13(float, float);
__device__ float af14(float, float);
__device__ float af15(float, float);
__device__ float af16(float, float);
__device__ float af17(float, float);

#define BIAS 1

// Neuron nomenclature: <Layer>_<Row>, both zero indexed

// Activation function of each neuron
#define af__0_0(s) af0(s)
#define af__0_1(s) af0(s)
#define af__0_2(s) af0(s)
#define af__0_3(s) af0(s)
#define af__0_4(s) af0(s)
#define af__0_5(s) af0(s)
#define af__0_6(s) af0(s)
#define af__0_7(s) af0(s)
#define af__0_8(s) af0(s)
#define af__0_9(s) af0(s)
#define af__1_0(s) af3(s)
#define af__1_1(s) af3(s)
#define af__1_2(s) af3(s)
#define af__1_3(s) af3(s)
#define af__1_4(s) af3(s)
#define af__1_5(s) af3(s)
#define af__1_6(s) af3(s)
#define af__1_7(s) af3(s)
#define af__1_8(s) af3(s)
#define af__2_0(s) af3(s)
#define af__2_1(s) af3(s)

// Activation steepness of each neuron
#define as__0_0 0.0
#define as__0_1 0.0
#define as__0_2 0.0
#define as__0_3 0.0
#define as__0_4 0.0
#define as__0_5 0.0
#define as__0_6 0.0
#define as__0_7 0.0
#define as__0_8 0.0
#define as__0_9 0.0
#define as__1_0 0.5
#define as__1_1 0.5
#define as__1_2 0.5
#define as__1_3 0.5
#define as__1_4 0.5
#define as__1_5 0.5
#define as__1_6 0.5
#define as__1_7 0.5
#define as__1_8 0.0
#define as__2_0 0.5
#define as__2_1 0.0

// Weights of each connection
#define w__0_1__1_2 20.4074363708
#define w__0_1__1_3 -0.174281224608
#define w__0_1__1_0 9.25054359436
#define w__0_1__1_1 0.918348073959
#define w__0_1__1_6 -2.32161331177
#define w__0_1__1_7 -11.4356584549
#define w__0_1__1_4 -12.686876297
#define w__0_1__1_5 0.918335437775
#define w__0_1__1_8 0.0
#define w__1_2__2_1 0.0
#define w__0_2__1_8 0.0
#define w__0_2__1_3 3.75048756599
#define w__0_2__1_2 25.9546051025
#define w__0_2__1_1 29.5378265381
#define w__0_2__1_0 19.6618022919
#define w__0_2__1_7 -6.45654726028
#define w__0_2__1_6 11.6792440414
#define w__0_2__1_5 2.13222026825
#define w__0_2__1_4 1.79341208935
#define w__0_4__1_8 0.0
#define w__1_6__2_1 0.0
#define w__1_6__2_0 -2.81729698181
#define w__0_4__1_5 1.12437438965
#define w__0_4__1_4 2.17328071594
#define w__0_4__1_7 0.913624763489
#define w__0_4__1_6 -0.589273929596
#define w__0_4__1_1 1.56612455845
#define w__0_4__1_0 -0.110017366707
#define w__0_4__1_3 0.311530262232
#define w__0_4__1_2 0.982897520065
#define w__0_5__1_8 0.0
#define w__0_5__1_6 6.07433748245
#define w__0_5__1_7 10.6060943604
#define w__0_5__1_4 20.4440059662
#define w__0_5__1_5 1500.0
#define w__0_5__1_2 2.8714056015
#define w__0_5__1_3 3.629966259
#define w__0_5__1_0 3.17400503159
#define w__0_5__1_1 35.5815849304
#define w__0_9__1_6 2.49581360817
#define w__1_1__2_0 -1.9589369297
#define w__0_7__1_5 0.676191747189
#define w__1_5__2_0 2.3742415905
#define w__1_1__2_1 0.0
#define w__0_0__1_1 -17.7363414764
#define w__0_0__1_0 4.37373161316
#define w__0_0__1_3 -2.84029078484
#define w__0_0__1_2 5.70691490173
#define w__0_0__1_5 -0.62045019865
#define w__0_0__1_4 -18.4688968658
#define w__0_0__1_7 -16.8891067505
#define w__0_0__1_6 -6.40613222122
#define w__0_0__1_8 0.0
#define w__0_7__1_8 0.0
#define w__0_8__1_1 14.5300321579
#define w__1_0__2_1 0.0
#define w__1_0__2_0 -4.61353588104
#define w__0_8__1_0 -6.716796875
#define w__0_7__1_0 -7.18473482132
#define w__0_7__1_1 -4.39541769028
#define w__0_7__1_2 -23.1197376251
#define w__0_7__1_3 0.239412426949
#define w__0_7__1_4 13.3713817596
#define w__0_9__1_0 6.49069356918
#define w__0_7__1_6 1.52481377125
#define w__0_7__1_7 15.0694332123
#define w__0_8__1_8 0.0
#define w__0_9__1_8 0.0
#define w__0_9__1_1 2.28915715218
#define w__0_9__1_2 2.22969079018
#define w__0_9__1_3 0.18257035315
#define w__0_8__1_3 3.07957172394
#define w__0_8__1_2 -4.84524917603
#define w__0_8__1_5 1500.0
#define w__0_9__1_7 1.71538960934
#define w__0_9__1_4 9.08893871307
#define w__0_8__1_4 19.213684082
#define w__0_8__1_7 15.0401363373
#define w__0_8__1_6 10.9188461304
#define w__0_9__1_5 9.04582309723
#define w__1_3__2_0 14.0717115402
#define w__1_3__2_1 0.0
#define w__1_7__2_0 -3.44167852402
#define w__1_7__2_1 0.0
#define w__1_8__2_1 0.0
#define w__1_8__2_0 5.80922460556
#define w__0_6__1_7 -1.22573280334
#define w__0_6__1_6 -5.28293657303
#define w__0_6__1_5 1500.0
#define w__0_6__1_4 -1.05261623859
#define w__0_6__1_3 -3.74990677834
#define w__0_6__1_2 -20.607542038
#define w__0_6__1_1 -29.0600967407
#define w__0_6__1_0 -19.6648674011
#define w__0_6__1_8 0.0
#define w__1_4__2_1 0.0
#define w__1_4__2_0 -4.30801439285
#define w__1_5__2_1 0.0
#define w__1_2__2_0 -2.42484307289
#define w__0_3__1_8 0.0
#define w__0_3__1_4 -22.9626140594
#define w__0_3__1_5 37.5691452026
#define w__0_3__1_6 -15.8191728592
#define w__0_3__1_7 -5.31044340134
#define w__0_3__1_0 -4.28493261337
#define w__0_3__1_1 -30.9982128143
#define w__0_3__1_2 -7.46965312958
#define w__0_3__1_3 -4.29126882553

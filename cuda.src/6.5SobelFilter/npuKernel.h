// TODO: Auto generate

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
#define as__0_0 0.00000000000000000000e+00
#define as__0_1 0.00000000000000000000e+00
#define as__0_2 0.00000000000000000000e+00
#define as__0_3 0.00000000000000000000e+00
#define as__0_4 0.00000000000000000000e+00
#define as__0_5 0.00000000000000000000e+00
#define as__0_6 0.00000000000000000000e+00
#define as__0_7 0.00000000000000000000e+00
#define as__0_8 0.00000000000000000000e+00
#define as__0_9 0.00000000000000000000e+00
#define as__1_0 5.00000000000000000000e-01
#define as__1_1 5.00000000000000000000e-01
#define as__1_2 5.00000000000000000000e-01
#define as__1_3 5.00000000000000000000e-01
#define as__1_4 5.00000000000000000000e-01
#define as__1_5 5.00000000000000000000e-01
#define as__1_6 5.00000000000000000000e-01
#define as__1_7 5.00000000000000000000e-01
#define as__1_8 0.00000000000000000000e+00 
#define as__2_0 5.00000000000000000000e-01
#define as__2_1 0.00000000000000000000e+00

// Neuron 1_0: Num_inputs: 10
// Weights from 0_* to 1_0 
#define w__0_0__1_0 4.37373161315917968750e+00
#define w__0_1__1_0 9.25054359436035156250e+00
#define w__0_2__1_0 1.96618022918701171875e+01
#define w__0_3__1_0 -4.28493261337280273438e+00
#define w__0_4__1_0 -1.10017366707324981689e-01
#define w__0_5__1_0 3.17400503158569335938e+00
#define w__0_6__1_0 -1.96648674011230468750e+01
#define w__0_7__1_0 -7.18473482131958007812e+00
#define w__0_8__1_0 -6.71679687500000000000e+00
#define w__0_9__1_0 6.49069356918334960938e+00

// Neuron 1_1: Num_inputs: 10
// Weights from 0_* to 1_1
#define w__0_0__1_1 -1.77363414764404296875e+01 
#define w__0_1__1_1 9.18348073959350585938e-01
#define w__0_2__1_1 2.95378265380859375000e+01
#define w__0_3__1_1 -3.09982128143310546875e+01
#define w__0_4__1_1 1.56612455844879150391e+00
#define w__0_5__1_1 3.55815849304199218750e+01
#define w__0_6__1_1 -2.90600967407226562500e+01
#define w__0_7__1_1 -4.39541769027709960938e+00
#define w__0_8__1_1 1.45300321578979492188e+01
#define w__0_9__1_1 1.45300321578979492188e+01

// Neuron 1_2: Num_inputs: 10
// Weights from 0_* to 1_2
#define w__0_0__1_2 5.70691490173339843750e+00
#define w__0_1__1_2 2.04074363708496093750e+01
#define w__0_2__1_2 2.59546051025390625000e+01
#define w__0_3__1_2 -7.46965312957763671875e+00
#define w__0_4__1_2 9.82897520065307617188e-01
#define w__0_5__1_2 2.87140560150146484375e+00
#define w__0_6__1_2 -2.06075420379638671875e+01
#define w__0_7__1_2 -2.31197376251220703125e+01
#define w__0_8__1_2 -4.84524917602539062500e+00
#define w__0_9__1_2 2.22969079017639160156e+00

// Neuron 1_3: Num_inputs: 10
// Weights from 0_* to 1_3
#define w__0_0__1_3 -2.84029078483581542969e+00
#define w__0_1__1_3 -1.74281224608421325684e-01
#define w__0_2__1_3 3.75048756599426269531e+00
#define w__0_3__1_3 -4.29126882553100585938e+00
#define w__0_4__1_3 3.11530262231826782227e-01
#define w__0_5__1_3 3.62996625900268554688e+00
#define w__0_6__1_3 -3.74990677833557128906e+00
#define w__0_7__1_3 2.39412426948547363281e-01
#define w__0_8__1_3 3.07957172393798828125e+00
#define w__0_9__1_3 1.82570353150367736816e-01

// Neuron 1_4: Num_inputs: 10
// Weights from 0_* to 1_4
#define w__0_0__1_4 -1.84688968658447265625e+01
#define w__0_1__1_4 -1.26868762969970703125e+01
#define w__0_2__1_4 1.79341208934783935547e+00
#define w__0_3__1_4 -2.29626140594482421875e+01
#define w__0_4__1_4 2.17328071594238281250e+00
#define w__0_5__1_4 2.04440059661865234375e+01
#define w__0_6__1_4 -1.05261623859405517578e+00
#define w__0_7__1_4 1.33713817596435546875e+01
#define w__0_8__1_4 1.92136840820312500000e+01
#define w__0_9__1_4 9.08893871307373046875e+00

// Neuron 1_5: Num_inputs: 10
// Weights from 0_* to 1_5
#define w__0_0__1_5 -6.20450198650360107422e-01 
#define w__0_1__1_5 9.18335437774658203125e-01
#define w__0_2__1_5 2.13222026824951171875e+00
#define w__0_3__1_5 3.75691452026367187500e+01
#define w__0_4__1_5 1.12437438964843750000e+00
#define w__0_5__1_5 1.50000000000000000000e+03
#define w__0_6__1_5 1.50000000000000000000e+03
#define w__0_7__1_5 6.76191747188568115234e-01
#define w__0_8__1_5 1.50000000000000000000e+03
#define w__0_9__1_5 9.04582309722900390625e+00

// Neuron 1_6: Num_inputs: 10
// Weights from 0_* to 1_6
#define w__0_0__1_6 -6.40613222122192382812e+00
#define w__0_1__1_6 -2.32161331176757812500e+00
#define w__0_2__1_6 1.16792440414428710938e+01
#define w__0_3__1_6 -1.58191728591918945312e+01
#define w__0_4__1_6 -5.89273929595947265625e-01
#define w__0_5__1_6 6.07433748245239257812e+00
#define w__0_6__1_6 -5.28293657302856445312e+00
#define w__0_7__1_6 1.52481377124786376953e+00
#define w__0_8__1_6 1.09188461303710937500e+01
#define w__0_9__1_6 2.49581360816955566406e+00

// Neuron 1_7: Num_inputs: 10
// Weights from 0_* to 1_7
#define w__0_0__1_7 -1.68891067504882812500e+01
#define w__0_1__1_7 -1.14356584548950195312e+01
#define w__0_2__1_7 -6.45654726028442382812e+00
#define w__0_3__1_7 -5.31044340133666992188e+00
#define w__0_4__1_7 9.13624763488769531250e-01
#define w__0_5__1_7 1.06060943603515625000e+01
#define w__0_6__1_7 -1.22573280334472656250e+00
#define w__0_7__1_7 1.50694332122802734375e+01
#define w__0_8__1_7 1.50401363372802734375e+01
#define w__0_9__1_7 1.71538960933685302734e+00

// Neuron 1_8: Num_inputs: 0
// Weights from 0_* to 1_3
#define w__0_0__1_8 0
#define w__0_1__1_8 0
#define w__0_2__1_8 0
#define w__0_3__1_8 0
#define w__0_4__1_8 0
#define w__0_5__1_8 0
#define w__0_6__1_8 0
#define w__0_7__1_8 0
#define w__0_8__1_8 0
#define w__0_9__1_8 0

// Neuron 2_0: Num_inputs: 9
// Weights from 1_* to 2_0
#define w__1_0__2_0 -4.61353588104248046875e+00
#define w__1_1__2_0 -1.95893692970275878906e+00
#define w__1_2__2_0 -2.42484307289123535156e+00
#define w__1_3__2_0 1.40717115402221679688e+01
#define w__1_4__2_0 -4.30801439285278320312e+00
#define w__1_5__2_0 2.37424159049987792969e+00
#define w__1_6__2_0 -2.81729698181152343750e+00
#define w__1_7__2_0 -3.44167852401733398438e+00 
#define w__1_8__2_0 5.80922460556030273438e+00

// Neuron 2_1: Num_inputs: 0
// Weights from 1_* to 2_0
#define w__1_0__2_1 0
#define w__1_1__2_1 0
#define w__1_2__2_1 0
#define w__1_3__2_1 0
#define w__1_4__2_1 0
#define w__1_5__2_1 0
#define w__1_6__2_1 0
#define w__1_7__2_1 0
#define w__1_8__2_1 0


#endif
#endif 

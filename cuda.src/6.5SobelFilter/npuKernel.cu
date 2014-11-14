// TODO: Auto generate

#include "npuKernel.h"

// Define various activation functions from fann_activationfunc_enum
// http://leenissen.dk/fann/html/files/fann_data-h.html
// Date: Nov 13, 2014
__device__ float af0(float sumIn, float steepness) {
    return sumIn * steepness;
}

//__device__ float af1(float sumIn, float steepness) {    
//}

//__device__ float af2(float sumIn, float steepness) {
//}

__device__ float af3(float sumIn, float steepness) {
    return ( 1.0f / (1 + exp(-2 * steepness * sumIn)) );
}

//__device__ float af4(float sumIn, float steepness) {
//}

__device__ float af5(float sumIn, float steepness) {
    return ( 2.0f / (1 + exp(-2 * steepness * sumIn)) - 1 );
}

//__device__ float af6(float sumIn, float steepness) {
//}

__device__ float af7(float sumIn, float steepness) {
    return ( exp(-sumIn*steepness*sumIn*steepness) );
}

__device__ float af8(float sumIn, float steepness) {
    return ( 2*exp(-sumIn*steepness*sumIn*steepness) - 1 );
}

__device__ float af9(float sumIn, float steepness) {
    return ( 0.5*sumIn*steepness/(1 + fabs(sumIn*steepness)) + 0.5 );
}

__device__ float af10(float sumIn, float steepness) {
    return ( sumIn*steepness/(1 + fabs(sumIn*steepness)) );
}

__device__ float af11(float sumIn, float steepness) {
    return ( sumIn * steepness );
}

__device__ float af12(float sumIn, float steepness) {
    return ( sumIn * steepness );
}

__device__ float af13(float sumIn, float steepness) {
    return ( sin(sumIn*steepness) );
}

__device__ float af14(float sumIn, float steepness) {
    return ( cos(sumIn*steepness) );
}

__device__ float af15(float sumIn, float steepness) {
    return ( sin(sumIn*steepness)*0.5 + 0.5 );
}

__device__ float af16(float sumIn, float steepness) {
    return ( cos(sumIn*steepness)*0.5 + 0.5 );
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
             float fScale) {

// Programmer needs to write this in the script
    float ul = ul_in / 256.0;
    float um = um_in / 256.0;
    float ur = ur_in / 256.0;
    float ml = ml_in / 256.0;
    float mm = mm_in / 256.0;
    float mr = mr_in / 256.0;
    float ll = ll_in / 256.0;
    float lm = lm_in / 256.0;
    float lr = lr_in / 256.0;
// End Programmer

    float sum = 0.0;    
    
    sum = ul*w__0_0__1_0 + um*w__0_1__1_0 + ur*w__0_2__1_0 +
            ml*w__0_3__1_0 + mm*w__0_4__1_0 + mr*w__0_5__1_0 +
            ll*w__0_6__1_0 + lm*w__0_7__1_0 + lr*w__0_8__1_0 +
            BIAS*w__0_9__1_0;        
    float n__1_0 = af__1_0(sum, as__1_0); 
    
    sum = ul*w__0_0__1_1 + um*w__0_1__1_1 + ur*w__0_2__1_1 +
            ml*w__0_3__1_1 + mm*w__0_4__1_1 + mr*w__0_5__1_1 +
            ll*w__0_6__1_1 + lm*w__0_7__1_1 + lr*w__0_8__1_1 +
            BIAS*w__0_9__1_1;        
    float n__1_1 = af__1_1(sum, as__1_1); 
    
    sum = ul*w__0_0__1_2 + um*w__0_1__1_2 + ur*w__0_2__1_2 +
            ml*w__0_3__1_2 + mm*w__0_4__1_2 + mr*w__0_5__1_2 +
            ll*w__0_6__1_2 + lm*w__0_7__1_2 + lr*w__0_8__1_2 +
            BIAS*w__0_9__1_2;        
    float n__1_2 = af__1_2(sum, as__1_2); 
    
    sum = ul*w__0_0__1_3 + um*w__0_1__1_3 + ur*w__0_2__1_3 +
            ml*w__0_3__1_3 + mm*w__0_4__1_3 + mr*w__0_5__1_3 +
            ll*w__0_6__1_3 + lm*w__0_7__1_3 + lr*w__0_8__1_3 +
            BIAS*w__0_9__1_3;        
    float n__1_3 = af__1_3(sum, as__1_3); 
    
    sum = ul*w__0_0__1_4 + um*w__0_1__1_4 + ur*w__0_2__1_4 +
            ml*w__0_3__1_4 + mm*w__0_4__1_4 + mr*w__0_5__1_4 +
            ll*w__0_6__1_4 + lm*w__0_7__1_4 + lr*w__0_8__1_4 +
            BIAS*w__0_9__1_4;        
    float n__1_4 = af__1_4(sum, as__1_4); 
    
    sum = ul*w__0_0__1_5 + um*w__0_1__1_5 + ur*w__0_2__1_5 +
            ml*w__0_3__1_5 + mm*w__0_4__1_5 + mr*w__0_5__1_5 +
            ll*w__0_6__1_5 + lm*w__0_7__1_5 + lr*w__0_8__1_5 +
            BIAS*w__0_9__1_5;        
    float n__1_5 = af__1_5(sum, as__1_5); 
    
    sum = ul*w__0_0__1_6 + um*w__0_1__1_6 + ur*w__0_2__1_6 +
            ml*w__0_3__1_6 + mm*w__0_4__1_6 + mr*w__0_5__1_6 +
            ll*w__0_6__1_6 + lm*w__0_7__1_6 + lr*w__0_8__1_6 +
            BIAS*w__0_9__1_6;        
    float n__1_6 = af__1_6(sum, as__1_6); 
    
    sum = ul*w__0_0__1_7 + um*w__0_1__1_7 + ur*w__0_2__1_7 +
            ml*w__0_3__1_7 + mm*w__0_4__1_7 + mr*w__0_5__1_7 +
            ll*w__0_6__1_7 + lm*w__0_7__1_7 + lr*w__0_8__1_7 +
            BIAS*w__0_9__1_7;        
    float n__1_7 = af__1_7(sum, as__1_7); 
    
    sum = ul*w__0_0__1_8 + um*w__0_1__1_8 + ur*w__0_2__1_8 +
            ml*w__0_3__1_8 + mm*w__0_4__1_8 + mr*w__0_5__1_8 +
            ll*w__0_6__1_8 + lm*w__0_7__1_8 + lr*w__0_8__1_8 +
            BIAS*w__0_9__1_8;        
    float n__1_8 = af__1_8(sum, as__1_8); 
    
    sum = n__1_0*w__1_0__2_0 + 
            n__1_1*w__1_1__2_0 + 
            n__1_2*w__1_2__2_0 + 
            n__1_3*w__1_3__2_0 + 
            n__1_4*w__1_4__2_0 + 
            n__1_5*w__1_5__2_0 + 
            n__1_6*w__1_6__2_0 + 
            n__1_7*w__1_7__2_0 + 
            n__1_8*w__1_8__2_0 ; 
    float n__2_0 = af__2_0(sum, as__2_0); 
    
    sum = n__1_0*w__1_0__2_1 + 
            n__1_1*w__1_1__2_1 + 
            n__1_2*w__1_2__2_1 + 
            n__1_3*w__1_3__2_1 + 
            n__1_4*w__1_4__2_1 + 
            n__1_5*w__1_5__2_1 + 
            n__1_6*w__1_6__2_1 + 
            n__1_7*w__1_7__2_1 + 
            n__1_8*w__1_8__2_1 ; 
    float n__2_1 = af__2_1(sum, as__2_1); 

// Programmer needs to write this in the script
    return (unsigned char) (n__2_0 * 256.0);
// End programmer
}

// TODO: Auto generate

#ifndef __NPUKERNEL_H_
#define __NPUKERNEL_H_

#include <cuda_runtime_api.h>
#include <stdio.h>

#ifdef NPU
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
             float fScale);

// Define weights

#endif
#endif 

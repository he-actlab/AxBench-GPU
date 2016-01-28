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

/*
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */


//#include <helper_functions.h>   // helper functions for string parsing
//#include <helper_cuda.h>        // helper functions CUDA error checking and initialization
 // Amir
#include <fstream>
 using namespace std;
// Rima
////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(
    float *h_CallResult,
    float *h_PutResult,
    float *h_StockPrice,
    float *h_OptionStrike,
    float *h_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
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



///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
__device__ inline float cndGPU(float d)
{
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float
    K = 1.0f / (1.0f + 0.2316419f * fabsf(d));

    float
    cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ inline void BlackScholesBodyGPU(
    float &CallResult,
    float &PutResult,
    float S, //Stock price
    float X, //Option strike
    float T, //Option years
    float R, //Riskless rate
    float V  //Volatility rate
)
{
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    float parrotInput[3];
    float parrotOutput[1];

    parrotInput[0] = S;
    parrotInput[1] = X;
    parrotInput[2] = T;

float layer_1_0 = parrotInput[0] * -0.332116 + parrotInput[1] * 0.250840 + parrotInput[2] * -0.614888 + 1.0f * 0.694309;

float layer_1_1 = parrotInput[0] * -0.306451 + parrotInput[1] * 0.223694 + parrotInput[2] * -0.540508 + 1.0f * 0.649405;

float layer_1_2 = parrotInput[0] * -0.169577 + parrotInput[1] * 0.004298 + parrotInput[2] * -0.169569 + 1.0f * 0.049087;

float layer_1_3 = parrotInput[0] * 0.084604 + parrotInput[1] * 0.292642 + parrotInput[2] * -0.444550 + 1.0f * 0.614365;

float layer_1_4 = parrotInput[0] * -0.136793 + parrotInput[1] * 0.013866 + parrotInput[2] * -0.033237 + 1.0f * 0.058597;

float layer_1_5 = parrotInput[0] * 0.234285 + parrotInput[1] * -0.026773 + parrotInput[2] * 0.109796 + 1.0f * -6.014935;

float layer_1_6 = parrotInput[0] * -0.375738 + parrotInput[1] * 0.310065 + parrotInput[2] * -0.832586 + 1.0f * 0.898780;

float layer_1_7 = parrotInput[0] * -0.360932 + parrotInput[1] * 0.241707 + parrotInput[2] * -0.495324 + 1.0f * 0.644464;

float layer_1_8 = parrotInput[0] * -0.020845 + parrotInput[1] * 0.030548 + parrotInput[2] * -0.445393 + 1.0f * 0.253491;

float layer_1_9 = parrotInput[0] * 1.112529 + parrotInput[1] * -2.702432 + parrotInput[2] * -14.137556 + 1.0f * -0.149519;

float layer_1_10 = parrotInput[0] * -0.125802 + parrotInput[1] * 0.254723 + parrotInput[2] * -0.630701 + 1.0f * 0.716845;

float layer_1_11 = parrotInput[0] * -0.216828 + parrotInput[1] * 0.308279 + parrotInput[2] * -0.674003 + 1.0f * 0.587158;

float layer_1_12 = parrotInput[0] * -0.152571 + parrotInput[1] * 0.140373 + parrotInput[2] * -0.532035 + 1.0f * 0.836703;

float layer_1_13 = parrotInput[0] * -0.349835 + parrotInput[1] * 0.298816 + parrotInput[2] * -0.854428 + 1.0f * 1.057956;

float layer_1_14 = parrotInput[0] * -0.353645 + parrotInput[1] * 0.274070 + parrotInput[2] * -0.684168 + 1.0f * 0.727741;

float layer_1_15 = parrotInput[0] * -0.177731 + parrotInput[1] * 0.284129 + parrotInput[2] * -0.480911 + 1.0f * 0.624303;

float layer_2_0 = sigmoid(layer_1_0, 0.500000) * -0.073969 + sigmoid(layer_1_1, 0.500000) * -0.014646 + sigmoid(layer_1_2, 0.500000) * -19.731762 + sigmoid(layer_1_3, 0.500000) * -0.020306 + sigmoid(layer_1_4, 0.500000) * -5.598285 + sigmoid(layer_1_5, 0.500000) * 1.677688 + sigmoid(layer_1_6, 0.500000) * -0.023441 + sigmoid(layer_1_7, 0.500000) * -0.005766 + sigmoid(layer_1_8, 0.500000) * -3.517432 + sigmoid(layer_1_9, 0.500000) * 1500.000000 + sigmoid(layer_1_10, 0.500000) * 0.025085 + sigmoid(layer_1_11, 0.500000) * 0.008612 + sigmoid(layer_1_12, 0.500000) * -0.494611 + sigmoid(layer_1_13, 0.500000) * -0.057754 + sigmoid(layer_1_14, 0.500000) * -0.011988 + sigmoid(layer_1_15, 0.500000) * 0.012520 + 1.0f * 0.137436;

layer_2_0 = sigmoid(layer_2_0, 0.5);

parrotOutput[0] = layer_2_0;

// parrotOutput[0] = layer_2_0;
// 
//     sqrtT = sqrtf(T);
//     d1 = (__logf(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
//     d2 = d1 - V * sqrtT;
// 
//     CNDD1 = cndGPU(d1);
//     CNDD2 = cndGPU(d2);
// 
//     //Calculate Call and Put simultaneously
//     expRT = __expf(- R * T);
//     CallResult = S * CNDD1 - X * expRT * CNDD2;
//     parrotOutput[0] = CallResult / 10.0;
// 
// #pragma parrot(output, "BlackScholesBodyGPU", [1]<0.0;0.9>parrotOutput)

    CallResult = parrotOutput[0] * 10.0;
    PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}


////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void BlackScholesGPU(
    float *d_CallResult,
    float *d_PutResult,
    float *d_StockPrice,
    float *d_OptionStrike,
    float *d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
)
{
    ////Thread index
    //const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    ////Total number of threads in execution grid
    //const int THREAD_N = blockDim.x * gridDim.x;

    const int opt = blockDim.x * blockIdx.x + threadIdx.x;

    //No matter how small is execution grid or how large OptN is,
    //exactly OptN indices will be processed with perfect memory coalescing
    //for (int opt = tid; opt < optN; opt += THREAD_N)
    if (opt < optN)
        BlackScholesBodyGPU(
            d_CallResult[opt],
            d_PutResult[opt],
            d_StockPrice[opt],
            d_OptionStrike[opt],
            d_OptionYears[opt],
            Riskfree,
            Volatility
        );
}

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int OPT_N = 4000000;
const int  NUM_ITERATIONS = 1; // Amir: Change number of iteration


const int          OPT_SZ = OPT_N * sizeof(float);
const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{

#pragma parrot.start("BlackScholesBodyGPU")
    // Start logs
    //printf("[%s] - Starting...\n", argv[0]);

    //'h_' prefix - CPU (host) memory space
    float
    //Results calculated by CPU for reference
    *h_CallResultCPU,
    *h_PutResultCPU,
    //CPU copy of GPU results
    *h_CallResultGPU,
    *h_PutResultGPU,
    //CPU instance of input data
    *h_StockPrice,
    *h_OptionStrike,
    *h_OptionYears;

    //'d_' prefix - GPU (device) memory space
    float
    //Results calculated by GPU
    *d_CallResult,
    *d_PutResult,
    //GPU instance of input data
    *d_StockPrice,
    *d_OptionStrike,
    *d_OptionYears;

    //double
    //delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime;
    //double gpuTime;

    //StopWatchInterface *hTimer = NULL;
    int i;

    //findCudaDevice(argc, (const char **)argv);

    //sdkCreateTimer(&hTimer);

    //printf("Initializing data...\n");
    //printf("...allocating CPU memory for options.\n");
    h_CallResultCPU = (float *)malloc(OPT_SZ);
    h_PutResultCPU  = (float *)malloc(OPT_SZ);
    h_CallResultGPU = (float *)malloc(OPT_SZ);
    h_PutResultGPU  = (float *)malloc(OPT_SZ);
    h_StockPrice    = (float *)malloc(OPT_SZ);
    h_OptionStrike  = (float *)malloc(OPT_SZ);
    h_OptionYears   = (float *)malloc(OPT_SZ);

    //printf("...allocating GPU memory for options.\n");
    cudaMalloc((void **)&d_CallResult,   OPT_SZ);
    cudaMalloc((void **)&d_PutResult,    OPT_SZ);
    cudaMalloc((void **)&d_StockPrice,   OPT_SZ);
    cudaMalloc((void **)&d_OptionStrike, OPT_SZ);
    cudaMalloc((void **)&d_OptionYears,  OPT_SZ);

    //printf("...generating input data in CPU mem.\n");
    srand(5347);

    // Amir
    std::ifstream dataFile(argv[1]);
    int numberOptions;
    dataFile >> numberOptions;
    //std::cout << "Total number of options:  " << numberOptions << std::endl;
    float stockPrice, optionStrike, optionYear;
    // Rima

    //Generate options set
    for (i = 0; i < numberOptions; i++)
    {
        h_CallResultCPU[i] = 0.0f;
        h_PutResultCPU[i]  = -1.0f;
        //h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        //h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        //h_OptionYears[i]   = RandFloat(0.25f, 10.0f);

        // Amir
        dataFile >> stockPrice >> optionStrike >> optionYear;
        h_StockPrice[i] = stockPrice;
        h_OptionStrike[i] = optionStrike;
        h_OptionYears[i] =  optionYear;
        // Rima
    }

    int optionSize = numberOptions * sizeof(float);

    //printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing
    cudaMemcpy(d_StockPrice,  h_StockPrice,   optionSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionStrike, h_OptionStrike,  optionSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionYears,  h_OptionYears,   optionSize, cudaMemcpyHostToDevice);
    //printf("Data init done.\n\n");




    cudaDeviceSynchronize();


    for (i = 0; i < NUM_ITERATIONS; i++)
    {
        BlackScholesGPU<<<DIV_UP(numberOptions, 128), 128/*480, 128*/>>>(
            d_CallResult,
            d_PutResult,
            d_StockPrice,
            d_OptionStrike,
            d_OptionYears,
            RISKFREE,
            VOLATILITY,
            numberOptions
        );

    }

    cudaDeviceSynchronize();

    //Read back GPU results to compare them to CPU results
    cudaMemcpy(h_CallResultGPU, d_CallResult, optionSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_PutResultGPU,  d_PutResult,  optionSize, cudaMemcpyDeviceToHost);



    // Amir
    ofstream callResultFile;
    callResultFile.open(argv[2]);
    for (i = 0 ; i < numberOptions; i++)
    {
        callResultFile << h_CallResultGPU[i] << std::endl;
    }
    callResultFile.close();
    // Rima


#pragma parrot.end("BlackScholesBodyGPU")

    // printf("Shutting down...\n");
    // printf("...releasing GPU memory.\n");
    cudaFree(d_OptionYears);
    cudaFree(d_OptionStrike);
    cudaFree(d_StockPrice);
    cudaFree(d_PutResult);
    cudaFree(d_CallResult);

    //printf("...releasing CPU memory.\n");
    free(h_OptionYears);
    free(h_OptionStrike);
    free(h_StockPrice);
    free(h_PutResultGPU);
    free(h_CallResultGPU);
    free(h_PutResultCPU);
    free(h_CallResultCPU);

    cudaDeviceReset();

    //printf("Test passed\n");
    exit(EXIT_SUCCESS);
}

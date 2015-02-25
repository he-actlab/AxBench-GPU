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



////////////////////////////////////////////////////////////////////////////////
// KNN kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void KNN(
    TColor *dst,
    int imageW,
    int imageH,
    float Noise,
    float lerpC
)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    //Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

    if (ix < imageW && iy < imageH)
    {
        //Normalized counter for the weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float3 clr = {0, 0, 0};

        // Amir
        float parrotInput[81];
        float parrotOutput[1];
        // Rima


        //Center of the KNN window
        float4 clr00 = tex2D(texImage, x, y);

        // Amir
        // extract all the input data
        int index = 0;
        for (float i = -KNN_WINDOW_RADIUS; i <= KNN_WINDOW_RADIUS; i++)
            for (float j = -KNN_WINDOW_RADIUS; j <= KNN_WINDOW_RADIUS; j++)
            {
                float4 clrIJ = tex2D(texImage, x + j, y + i);
                parrotInput[index++] = clrIJ.x;
            }

        // Rima

float layer_1_0 = parrotInput[0] * -0.096926 + parrotInput[1] * 0.070523 + parrotInput[2] * -0.105036 + parrotInput[3] * 0.386045 + parrotInput[4] * -0.170193 + parrotInput[5] * -0.040189 + parrotInput[6] * -0.080159 + parrotInput[7] * -0.104354 + parrotInput[8] * 0.012936 + parrotInput[9] * -0.149094 + parrotInput[10] * -0.178201 + parrotInput[11] * -0.092541 + parrotInput[12] * -0.108650 + parrotInput[13] * -0.025495 + parrotInput[14] * 0.122688 + parrotInput[15] * -0.148161 + parrotInput[16] * -0.067186 + parrotInput[17] * -0.115818 + parrotInput[18] * -0.195421 + parrotInput[19] * -0.051400 + parrotInput[20] * -0.159678 + parrotInput[21] * 0.157946 + parrotInput[22] * 0.083086 + parrotInput[23] * -0.251081 + parrotInput[24] * -3.373682 + parrotInput[25] * -0.168307 + parrotInput[26] * -0.151057 + parrotInput[27] * -0.058161 + parrotInput[28] * 0.019227 + parrotInput[29] * -0.091428 + parrotInput[30] * -0.138767 + parrotInput[31] * -0.133537 + parrotInput[32] * 0.021159 + parrotInput[33] * -0.255032 + parrotInput[34] * 0.083946 + parrotInput[35] * 0.000143 + parrotInput[36] * -0.014353 + parrotInput[37] * -0.154418 + parrotInput[38] * -0.162392 + parrotInput[39] * 0.068697 + parrotInput[40] * -0.104329 + parrotInput[41] * -0.030100 + parrotInput[42] * -0.065456 + parrotInput[43] * 0.494435 + parrotInput[44] * 0.273778 + parrotInput[45] * 0.100187 + parrotInput[46] * -0.005222 + parrotInput[47] * -0.031940 + parrotInput[48] * -0.078587 + parrotInput[49] * 1500.000000 + parrotInput[50] * 1500.000000 + parrotInput[51] * 1500.000000 + parrotInput[52] * 1500.000000 + parrotInput[53] * 1500.000000 + parrotInput[54] * 1500.000000 + parrotInput[55] * 1500.000000 + parrotInput[56] * 1500.000000 + parrotInput[57] * 1500.000000 + parrotInput[58] * 1500.000000 + parrotInput[59] * 1500.000000 + parrotInput[60] * 1500.000000 + parrotInput[61] * 1500.000000 + parrotInput[62] * 1500.000000 + parrotInput[63] * 1500.000000 + parrotInput[64] * 1500.000000 + parrotInput[65] * 1500.000000 + parrotInput[66] * 1500.000000 + parrotInput[67] * 1500.000000 + parrotInput[68] * 1500.000000 + parrotInput[69] * 1500.000000 + parrotInput[70] * 1500.000000 + parrotInput[71] * 1500.000000 + parrotInput[72] * 1500.000000 + parrotInput[73] * 1500.000000 + parrotInput[74] * 1500.000000 + parrotInput[75] * 1500.000000 + parrotInput[76] * 1500.000000 + parrotInput[77] * 1500.000000 + parrotInput[78] * 1500.000000 + parrotInput[79] * 1500.000000 + parrotInput[80] * 1500.000000 + 1.0f * 0.571530;

float layer_1_1 = parrotInput[0] * -2.557614 + parrotInput[1] * 0.435976 + parrotInput[2] * 0.919837 + parrotInput[3] * -0.113462 + parrotInput[4] * -0.007375 + parrotInput[5] * 0.135958 + parrotInput[6] * 0.709400 + parrotInput[7] * -0.179930 + parrotInput[8] * 0.117803 + parrotInput[9] * 1.834325 + parrotInput[10] * -0.172303 + parrotInput[11] * -0.317853 + parrotInput[12] * 0.015317 + parrotInput[13] * 0.394269 + parrotInput[14] * 0.112206 + parrotInput[15] * 0.020016 + parrotInput[16] * -0.088251 + parrotInput[17] * -1.114340 + parrotInput[18] * -0.362688 + parrotInput[19] * 0.323048 + parrotInput[20] * -1.407858 + parrotInput[21] * 0.136338 + parrotInput[22] * 0.084514 + parrotInput[23] * -0.517802 + parrotInput[24] * -4.505941 + parrotInput[25] * -0.554871 + parrotInput[26] * -0.018777 + parrotInput[27] * 0.091261 + parrotInput[28] * -0.175078 + parrotInput[29] * -0.190075 + parrotInput[30] * -1.131927 + parrotInput[31] * -0.935292 + parrotInput[32] * -0.010035 + parrotInput[33] * -0.298582 + parrotInput[34] * 0.385064 + parrotInput[35] * -0.051921 + parrotInput[36] * -0.413576 + parrotInput[37] * -0.177271 + parrotInput[38] * -0.157787 + parrotInput[39] * -0.201542 + parrotInput[40] * -0.076607 + parrotInput[41] * 0.284991 + parrotInput[42] * -1.964817 + parrotInput[43] * 0.108425 + parrotInput[44] * -0.112578 + parrotInput[45] * -1.353715 + parrotInput[46] * -1.124061 + parrotInput[47] * -0.136481 + parrotInput[48] * -2.310552 + parrotInput[49] * 1500.000000 + parrotInput[50] * 1500.000000 + parrotInput[51] * 1500.000000 + parrotInput[52] * 1500.000000 + parrotInput[53] * 1500.000000 + parrotInput[54] * 1500.000000 + parrotInput[55] * 1500.000000 + parrotInput[56] * 1500.000000 + parrotInput[57] * 1500.000000 + parrotInput[58] * 1500.000000 + parrotInput[59] * 1500.000000 + parrotInput[60] * 1500.000000 + parrotInput[61] * 1500.000000 + parrotInput[62] * 1500.000000 + parrotInput[63] * 1500.000000 + parrotInput[64] * 1500.000000 + parrotInput[65] * 1500.000000 + parrotInput[66] * 1500.000000 + parrotInput[67] * 1500.000000 + parrotInput[68] * 1500.000000 + parrotInput[69] * 1500.000000 + parrotInput[70] * 1500.000000 + parrotInput[71] * 1500.000000 + parrotInput[72] * 1500.000000 + parrotInput[73] * 1500.000000 + parrotInput[74] * 1500.000000 + parrotInput[75] * 1500.000000 + parrotInput[76] * 1500.000000 + parrotInput[77] * 1500.000000 + parrotInput[78] * 1500.000000 + parrotInput[79] * 1500.000000 + parrotInput[80] * 1500.000000 + 1.0f * 5.844006;

float layer_1_2 = parrotInput[0] * -0.116560 + parrotInput[1] * -0.742706 + parrotInput[2] * -0.493501 + parrotInput[3] * -0.336304 + parrotInput[4] * -0.000909 + parrotInput[5] * -0.066339 + parrotInput[6] * -0.148044 + parrotInput[7] * -0.236128 + parrotInput[8] * -0.317273 + parrotInput[9] * -0.934638 + parrotInput[10] * 0.006710 + parrotInput[11] * -0.177343 + parrotInput[12] * -0.242296 + parrotInput[13] * -0.375173 + parrotInput[14] * -0.962868 + parrotInput[15] * -0.022183 + parrotInput[16] * -0.685929 + parrotInput[17] * -0.634398 + parrotInput[18] * -0.240390 + parrotInput[19] * -0.015456 + parrotInput[20] * -0.143650 + parrotInput[21] * -0.579735 + parrotInput[22] * -0.140896 + parrotInput[23] * -0.409378 + parrotInput[24] * -1.759079 + parrotInput[25] * -0.153768 + parrotInput[26] * -0.045588 + parrotInput[27] * -0.411093 + parrotInput[28] * -0.329784 + parrotInput[29] * -0.142675 + parrotInput[30] * -0.008369 + parrotInput[31] * -0.516992 + parrotInput[32] * -0.041337 + parrotInput[33] * -0.175382 + parrotInput[34] * -0.587586 + parrotInput[35] * 0.071739 + parrotInput[36] * -0.363504 + parrotInput[37] * 0.003173 + parrotInput[38] * -0.187472 + parrotInput[39] * -0.199217 + parrotInput[40] * -0.174681 + parrotInput[41] * -0.378997 + parrotInput[42] * -0.369374 + parrotInput[43] * -0.208416 + parrotInput[44] * -0.173094 + parrotInput[45] * 0.038276 + parrotInput[46] * -0.238956 + parrotInput[47] * -0.337090 + parrotInput[48] * -0.195075 + parrotInput[49] * 1500.000000 + parrotInput[50] * 1500.000000 + parrotInput[51] * 1500.000000 + parrotInput[52] * 1500.000000 + parrotInput[53] * 1500.000000 + parrotInput[54] * 1500.000000 + parrotInput[55] * 1500.000000 + parrotInput[56] * 1500.000000 + parrotInput[57] * 1500.000000 + parrotInput[58] * 1500.000000 + parrotInput[59] * 1500.000000 + parrotInput[60] * 1500.000000 + parrotInput[61] * 1500.000000 + parrotInput[62] * 1500.000000 + parrotInput[63] * 1500.000000 + parrotInput[64] * 1500.000000 + parrotInput[65] * 1500.000000 + parrotInput[66] * 1500.000000 + parrotInput[67] * 1500.000000 + parrotInput[68] * 1500.000000 + parrotInput[69] * 1500.000000 + parrotInput[70] * 1500.000000 + parrotInput[71] * 1500.000000 + parrotInput[72] * 1500.000000 + parrotInput[73] * 1500.000000 + parrotInput[74] * 1500.000000 + parrotInput[75] * 1500.000000 + parrotInput[76] * 1500.000000 + parrotInput[77] * 1500.000000 + parrotInput[78] * 1500.000000 + parrotInput[79] * 1500.000000 + parrotInput[80] * 1500.000000 + 1.0f * 0.441188;

float layer_1_3 = parrotInput[0] * -1.706265 + parrotInput[1] * -1.962238 + parrotInput[2] * -0.041662 + parrotInput[3] * 0.191742 + parrotInput[4] * -0.587728 + parrotInput[5] * 0.104866 + parrotInput[6] * 0.297617 + parrotInput[7] * -0.190316 + parrotInput[8] * 0.133684 + parrotInput[9] * -0.330246 + parrotInput[10] * -1.083180 + parrotInput[11] * -0.197981 + parrotInput[12] * 0.447252 + parrotInput[13] * 0.151060 + parrotInput[14] * 0.055858 + parrotInput[15] * -0.036884 + parrotInput[16] * 0.130737 + parrotInput[17] * -2.743937 + parrotInput[18] * -0.662717 + parrotInput[19] * 0.351881 + parrotInput[20] * 0.306565 + parrotInput[21] * 0.094753 + parrotInput[22] * 0.031192 + parrotInput[23] * -0.661642 + parrotInput[24] * -5.472023 + parrotInput[25] * -1.292212 + parrotInput[26] * 1.163155 + parrotInput[27] * 1.166636 + parrotInput[28] * -0.013062 + parrotInput[29] * 0.039309 + parrotInput[30] * -0.892030 + parrotInput[31] * -2.189285 + parrotInput[32] * -0.303053 + parrotInput[33] * 0.129285 + parrotInput[34] * 0.657254 + parrotInput[35] * -0.044009 + parrotInput[36] * 0.074180 + parrotInput[37] * -0.429012 + parrotInput[38] * -1.409947 + parrotInput[39] * 0.636166 + parrotInput[40] * 0.393564 + parrotInput[41] * 0.583437 + parrotInput[42] * 0.040164 + parrotInput[43] * 0.022291 + parrotInput[44] * -0.136863 + parrotInput[45] * -0.283103 + parrotInput[46] * 0.221537 + parrotInput[47] * 0.086566 + parrotInput[48] * 0.026949 + parrotInput[49] * 1500.000000 + parrotInput[50] * 1500.000000 + parrotInput[51] * 1500.000000 + parrotInput[52] * 1500.000000 + parrotInput[53] * 1500.000000 + parrotInput[54] * 1500.000000 + parrotInput[55] * 1500.000000 + parrotInput[56] * 1500.000000 + parrotInput[57] * 1500.000000 + parrotInput[58] * 1500.000000 + parrotInput[59] * 1500.000000 + parrotInput[60] * 1500.000000 + parrotInput[61] * 1500.000000 + parrotInput[62] * 1500.000000 + parrotInput[63] * 1500.000000 + parrotInput[64] * 1500.000000 + parrotInput[65] * 1500.000000 + parrotInput[66] * 1500.000000 + parrotInput[67] * 1500.000000 + parrotInput[68] * 1500.000000 + parrotInput[69] * 1500.000000 + parrotInput[70] * 1500.000000 + parrotInput[71] * 1500.000000 + parrotInput[72] * 1500.000000 + parrotInput[73] * 1500.000000 + parrotInput[74] * 1500.000000 + parrotInput[75] * 1500.000000 + parrotInput[76] * 1500.000000 + parrotInput[77] * 1500.000000 + parrotInput[78] * 1500.000000 + parrotInput[79] * 1500.000000 + parrotInput[80] * 1500.000000 + 1.0f * 7.974354;

float layer_1_4 = parrotInput[0] * 0.050936 + parrotInput[1] * 0.575795 + parrotInput[2] * -0.380772 + parrotInput[3] * 0.046577 + parrotInput[4] * 0.145047 + parrotInput[5] * 0.999916 + parrotInput[6] * 0.683352 + parrotInput[7] * 0.594906 + parrotInput[8] * 0.026834 + parrotInput[9] * 0.523834 + parrotInput[10] * -0.247787 + parrotInput[11] * -0.099455 + parrotInput[12] * 0.358585 + parrotInput[13] * 0.278964 + parrotInput[14] * 0.273800 + parrotInput[15] * 0.265118 + parrotInput[16] * -0.286249 + parrotInput[17] * -1.175553 + parrotInput[18] * -0.178343 + parrotInput[19] * 0.416827 + parrotInput[20] * -1.575743 + parrotInput[21] * 0.054348 + parrotInput[22] * -0.151071 + parrotInput[23] * -0.416965 + parrotInput[24] * -5.963019 + parrotInput[25] * -0.334977 + parrotInput[26] * 0.065397 + parrotInput[27] * 0.219529 + parrotInput[28] * -0.035264 + parrotInput[29] * 0.073890 + parrotInput[30] * -2.576822 + parrotInput[31] * -1.112967 + parrotInput[32] * -0.300696 + parrotInput[33] * -0.025542 + parrotInput[34] * 0.002749 + parrotInput[35] * -0.295943 + parrotInput[36] * -0.447477 + parrotInput[37] * -0.342457 + parrotInput[38] * -0.080785 + parrotInput[39] * -0.184739 + parrotInput[40] * 0.824989 + parrotInput[41] * 0.160091 + parrotInput[42] * -1.646625 + parrotInput[43] * -0.083441 + parrotInput[44] * 0.003826 + parrotInput[45] * -0.332433 + parrotInput[46] * -0.171541 + parrotInput[47] * -0.014815 + parrotInput[48] * -1.574032 + parrotInput[49] * 1500.000000 + parrotInput[50] * 1500.000000 + parrotInput[51] * 1500.000000 + parrotInput[52] * 1500.000000 + parrotInput[53] * 1500.000000 + parrotInput[54] * 1500.000000 + parrotInput[55] * 1500.000000 + parrotInput[56] * 1500.000000 + parrotInput[57] * 1500.000000 + parrotInput[58] * 1500.000000 + parrotInput[59] * 1500.000000 + parrotInput[60] * 1500.000000 + parrotInput[61] * 1500.000000 + parrotInput[62] * 1500.000000 + parrotInput[63] * 1500.000000 + parrotInput[64] * 1500.000000 + parrotInput[65] * 1500.000000 + parrotInput[66] * 1500.000000 + parrotInput[67] * 1500.000000 + parrotInput[68] * 1500.000000 + parrotInput[69] * 1500.000000 + parrotInput[70] * 1500.000000 + parrotInput[71] * 1500.000000 + parrotInput[72] * 1500.000000 + parrotInput[73] * 1500.000000 + parrotInput[74] * 1500.000000 + parrotInput[75] * 1500.000000 + parrotInput[76] * 1500.000000 + parrotInput[77] * 1500.000000 + parrotInput[78] * 1500.000000 + parrotInput[79] * 1500.000000 + parrotInput[80] * 1500.000000 + 1.0f * 6.587285;

float layer_1_5 = parrotInput[0] * 1.484017 + parrotInput[1] * 0.183945 + parrotInput[2] * 0.304439 + parrotInput[3] * -0.150977 + parrotInput[4] * -0.870468 + parrotInput[5] * -0.812230 + parrotInput[6] * -1.761558 + parrotInput[7] * 0.259221 + parrotInput[8] * 0.034391 + parrotInput[9] * 0.085238 + parrotInput[10] * -0.016121 + parrotInput[11] * -0.598464 + parrotInput[12] * -0.192595 + parrotInput[13] * -0.044021 + parrotInput[14] * 0.504554 + parrotInput[15] * 0.024784 + parrotInput[16] * 0.091585 + parrotInput[17] * -0.906366 + parrotInput[18] * -3.527908 + parrotInput[19] * -1.079825 + parrotInput[20] * 0.428395 + parrotInput[21] * 0.176796 + parrotInput[22] * 0.116080 + parrotInput[23] * -0.110080 + parrotInput[24] * -13.848407 + parrotInput[25] * -1.302125 + parrotInput[26] * -0.183225 + parrotInput[27] * 0.117972 + parrotInput[28] * -0.316474 + parrotInput[29] * -0.082666 + parrotInput[30] * -0.154670 + parrotInput[31] * -0.519661 + parrotInput[32] * -0.129481 + parrotInput[33] * -0.052986 + parrotInput[34] * 0.375490 + parrotInput[35] * 0.161186 + parrotInput[36] * -0.209149 + parrotInput[37] * -0.031106 + parrotInput[38] * 0.317077 + parrotInput[39] * 0.699491 + parrotInput[40] * 0.051629 + parrotInput[41] * 0.283181 + parrotInput[42] * 1.687586 + parrotInput[43] * 0.631310 + parrotInput[44] * 0.322295 + parrotInput[45] * -0.027460 + parrotInput[46] * 0.941584 + parrotInput[47] * 0.568673 + parrotInput[48] * 0.087517 + parrotInput[49] * 1500.000000 + parrotInput[50] * 1500.000000 + parrotInput[51] * 1500.000000 + parrotInput[52] * 1500.000000 + parrotInput[53] * 1500.000000 + parrotInput[54] * 1500.000000 + parrotInput[55] * 1500.000000 + parrotInput[56] * 1500.000000 + parrotInput[57] * 1500.000000 + parrotInput[58] * 1500.000000 + parrotInput[59] * 1500.000000 + parrotInput[60] * 1500.000000 + parrotInput[61] * 1500.000000 + parrotInput[62] * 1500.000000 + parrotInput[63] * 1500.000000 + parrotInput[64] * 1500.000000 + parrotInput[65] * 1500.000000 + parrotInput[66] * 1500.000000 + parrotInput[67] * 1500.000000 + parrotInput[68] * 1500.000000 + parrotInput[69] * 1500.000000 + parrotInput[70] * 1500.000000 + parrotInput[71] * 1500.000000 + parrotInput[72] * 1500.000000 + parrotInput[73] * 1500.000000 + parrotInput[74] * 1500.000000 + parrotInput[75] * 1500.000000 + parrotInput[76] * 1500.000000 + parrotInput[77] * 1500.000000 + parrotInput[78] * 1500.000000 + parrotInput[79] * 1500.000000 + parrotInput[80] * 1500.000000 + 1.0f * -0.772622;

float layer_1_6 = parrotInput[0] * -0.234732 + parrotInput[1] * -0.043292 + parrotInput[2] * -0.304138 + parrotInput[3] * -0.627250 + parrotInput[4] * -0.072424 + parrotInput[5] * 0.009230 + parrotInput[6] * 0.365317 + parrotInput[7] * -0.162984 + parrotInput[8] * -0.303549 + parrotInput[9] * -0.173109 + parrotInput[10] * -0.227493 + parrotInput[11] * -0.081715 + parrotInput[12] * -0.106809 + parrotInput[13] * -0.066878 + parrotInput[14] * -0.003557 + parrotInput[15] * -0.189740 + parrotInput[16] * -0.327014 + parrotInput[17] * 0.533848 + parrotInput[18] * 0.714082 + parrotInput[19] * -0.129782 + parrotInput[20] * -0.086374 + parrotInput[21] * -0.359231 + parrotInput[22] * -0.193658 + parrotInput[23] * -0.463978 + parrotInput[24] * -1.818332 + parrotInput[25] * -0.270026 + parrotInput[26] * -0.010083 + parrotInput[27] * -0.010990 + parrotInput[28] * -0.106876 + parrotInput[29] * -0.178134 + parrotInput[30] * -0.285876 + parrotInput[31] * -0.086699 + parrotInput[32] * -0.305365 + parrotInput[33] * -0.147757 + parrotInput[34] * -0.186303 + parrotInput[35] * -0.125085 + parrotInput[36] * -0.493273 + parrotInput[37] * -0.183417 + parrotInput[38] * -0.285313 + parrotInput[39] * -0.229855 + parrotInput[40] * -0.058710 + parrotInput[41] * -0.211642 + parrotInput[42] * -0.410024 + parrotInput[43] * -0.853140 + parrotInput[44] * -0.794030 + parrotInput[45] * -0.043224 + parrotInput[46] * -0.285317 + parrotInput[47] * -0.162315 + parrotInput[48] * -0.169765 + parrotInput[49] * 1500.000000 + parrotInput[50] * 1500.000000 + parrotInput[51] * 1500.000000 + parrotInput[52] * 1500.000000 + parrotInput[53] * 1500.000000 + parrotInput[54] * 1500.000000 + parrotInput[55] * 1500.000000 + parrotInput[56] * 1500.000000 + parrotInput[57] * 1500.000000 + parrotInput[58] * 1500.000000 + parrotInput[59] * 1500.000000 + parrotInput[60] * 1500.000000 + parrotInput[61] * 1500.000000 + parrotInput[62] * 1500.000000 + parrotInput[63] * 1500.000000 + parrotInput[64] * 1500.000000 + parrotInput[65] * 1500.000000 + parrotInput[66] * 1500.000000 + parrotInput[67] * 1500.000000 + parrotInput[68] * 1500.000000 + parrotInput[69] * 1500.000000 + parrotInput[70] * 1500.000000 + parrotInput[71] * 1500.000000 + parrotInput[72] * 1500.000000 + parrotInput[73] * 1500.000000 + parrotInput[74] * 1500.000000 + parrotInput[75] * 1500.000000 + parrotInput[76] * 1500.000000 + parrotInput[77] * 1500.000000 + parrotInput[78] * 1500.000000 + parrotInput[79] * 1500.000000 + parrotInput[80] * 1500.000000 + 1.0f * 0.581766;

float layer_1_7 = parrotInput[0] * -0.341533 + parrotInput[1] * 0.181774 + parrotInput[2] * -0.058665 + parrotInput[3] * 0.318831 + parrotInput[4] * 0.300294 + parrotInput[5] * 0.392395 + parrotInput[6] * 0.114612 + parrotInput[7] * 0.382304 + parrotInput[8] * -0.104131 + parrotInput[9] * -0.569474 + parrotInput[10] * -0.247301 + parrotInput[11] * -0.099900 + parrotInput[12] * 0.315121 + parrotInput[13] * -0.086548 + parrotInput[14] * 0.105990 + parrotInput[15] * -0.093193 + parrotInput[16] * -0.116588 + parrotInput[17] * -1.370755 + parrotInput[18] * 0.500618 + parrotInput[19] * 0.056385 + parrotInput[20] * -0.386958 + parrotInput[21] * 0.006940 + parrotInput[22] * -0.048990 + parrotInput[23] * -0.332421 + parrotInput[24] * -7.830618 + parrotInput[25] * 0.701420 + parrotInput[26] * 0.029884 + parrotInput[27] * 0.185449 + parrotInput[28] * -0.277012 + parrotInput[29] * -0.130468 + parrotInput[30] * -0.832614 + parrotInput[31] * -0.995097 + parrotInput[32] * 0.039925 + parrotInput[33] * 0.114066 + parrotInput[34] * 0.189223 + parrotInput[35] * -0.055817 + parrotInput[36] * -0.353610 + parrotInput[37] * -0.059443 + parrotInput[38] * -0.416615 + parrotInput[39] * 0.495926 + parrotInput[40] * 0.227872 + parrotInput[41] * 0.483618 + parrotInput[42] * -0.344418 + parrotInput[43] * -0.039663 + parrotInput[44] * 0.215596 + parrotInput[45] * 0.482645 + parrotInput[46] * -0.522343 + parrotInput[47] * 0.071521 + parrotInput[48] * -0.149972 + parrotInput[49] * 1500.000000 + parrotInput[50] * 1500.000000 + parrotInput[51] * 1500.000000 + parrotInput[52] * 1500.000000 + parrotInput[53] * 1500.000000 + parrotInput[54] * 1500.000000 + parrotInput[55] * 1500.000000 + parrotInput[56] * 1500.000000 + parrotInput[57] * 1500.000000 + parrotInput[58] * 1500.000000 + parrotInput[59] * 1500.000000 + parrotInput[60] * 1500.000000 + parrotInput[61] * 1500.000000 + parrotInput[62] * 1500.000000 + parrotInput[63] * 1500.000000 + parrotInput[64] * 1500.000000 + parrotInput[65] * 1500.000000 + parrotInput[66] * 1500.000000 + parrotInput[67] * 1500.000000 + parrotInput[68] * 1500.000000 + parrotInput[69] * 1500.000000 + parrotInput[70] * 1500.000000 + parrotInput[71] * 1500.000000 + parrotInput[72] * 1500.000000 + parrotInput[73] * 1500.000000 + parrotInput[74] * 1500.000000 + parrotInput[75] * 1500.000000 + parrotInput[76] * 1500.000000 + parrotInput[77] * 1500.000000 + parrotInput[78] * 1500.000000 + parrotInput[79] * 1500.000000 + parrotInput[80] * 1500.000000 + 1.0f * 6.078910;

float layer_2_0 = sigmoid(layer_1_0, 0.500000) * -1.167937 + sigmoid(layer_1_1, 0.500000) * -0.420580 + sigmoid(layer_1_2, 0.500000) * -1.396175 + sigmoid(layer_1_3, 0.500000) * -0.242896 + sigmoid(layer_1_4, 0.500000) * -0.345613 + sigmoid(layer_1_5, 0.500000) * -2.490019 + sigmoid(layer_1_6, 0.500000) * -1.183610 + sigmoid(layer_1_7, 0.500000) * -0.591413 + 1.0f * 0.981257;

layer_2_0 = sigmoid(layer_2_0, 0.5);

parrotOutput[0] = layer_2_0;

// parrotOutput[0] = layer_2_0;
// 
// 
//         //Cycle through KNN window, surrounding (x, y) texel
//         for (float i = -KNN_WINDOW_RADIUS; i <= KNN_WINDOW_RADIUS; i++)
//             for (float j = -KNN_WINDOW_RADIUS; j <= KNN_WINDOW_RADIUS; j++)
//             {
//                 float4     clrIJ = tex2D(texImage, x + j, y + i);
//                 float distanceIJ = vecLen(clr00, clrIJ);
// 
//                 //Derive final weight from color distance
//                 float   weightIJ = __expf(- (distanceIJ * Noise + (i * i + j * j) * INV_KNN_WINDOW_AREA));
// 
//                 //Accumulate (x + j, y + i) texel color with computed weight
//                 clr.x += clrIJ.x * weightIJ;
//                 clr.y += clrIJ.y * weightIJ;
//                 clr.z += clrIJ.z * weightIJ;
// 
//                 //Sum of weights for color normalization to [0..1] range
//                 sumWeights     += weightIJ;
// 
//                 //Update weight counter, if KNN weight for current window texel
//                 //exceeds the weight threshold
//                 fCount         += (weightIJ > KNN_WEIGHT_THRESHOLD) ? INV_KNN_WINDOW_AREA : 0;
//             }
// 
//         //Normalize result color by sum of weights
//         sumWeights = 1.0f / sumWeights;
//         clr.x *= sumWeights;
//         clr.y *= sumWeights;
//         clr.z *= sumWeights;
// 
//         //Choose LERP quotent basing on how many texels
//         //within the KNN window exceeded the weight threshold
//         float lerpQ = (fCount > KNN_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;
// 
//         //Write final result to global memory
//         clr.x = lerpf(clr.x, clr00.x, lerpQ);
//         clr.y = lerpf(clr.y, clr00.y, lerpQ);
//         clr.z = lerpf(clr.z, clr00.z, lerpQ);
// 
//         parrotOutput[0] = clr.x;
// 
// #pragma parrot(output, "KNN", [1]<0.0; 1.0>parrotOutput)

        clr.x = parrotOutput[0];
        clr.y = parrotOutput[0];
        clr.z = parrotOutput[0];

        dst[imageW * iy + ix] = make_color(clr.x, clr.y, clr.z, 0);
    };
}

extern "C"
void cuda_KNN(
    TColor *d_dst,
    int imageW,
    int imageH,
    float Noise,
    float lerpC
)
{

#pragma parrot.start("KNN")

    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    KNN<<<grid, threads>>>(d_dst, imageW, imageH, Noise, lerpC);
#pragma parrot.end("KNN")
}


////////////////////////////////////////////////////////////////////////////////
// Stripped KNN kernel, only highlighting areas with different LERP directions
////////////////////////////////////////////////////////////////////////////////
__global__ void KNNdiag(
    TColor *dst,
    int imageW,
    int imageH,
    float Noise,
    float lerpC
)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    //Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

    if (ix < imageW && iy < imageH)
    {
        //Normalized counter for the weight threshold
        float  fCount = 0;
        //Center of the KNN window
        float4  clr00 = tex2D(texImage, x, y);

        //Cycle through KNN window, surrounding (x, y) texel
        for (float i = -KNN_WINDOW_RADIUS; i <= KNN_WINDOW_RADIUS; i++)
            for (float j = -KNN_WINDOW_RADIUS; j <= KNN_WINDOW_RADIUS; j++)
            {
                float4     clrIJ = tex2D(texImage, x + j, y + i);
                float distanceIJ = vecLen(clr00, clrIJ);

                //Derive final weight from color and geometric distance
                float weightIJ  = __expf(- (distanceIJ * Noise + (i * i + j * j) * INV_KNN_WINDOW_AREA));

                //Update weight counter, if KNN weight for current window texel
                //exceeds the weight threshold
                fCount += (weightIJ > KNN_WEIGHT_THRESHOLD) ? INV_KNN_WINDOW_AREA : 0.0f;
            }

        //Choose LERP quotent basing on how many texels
        //within the KNN window exceeded the weight threshold
        float lerpQ = (fCount > KNN_LERP_THRESHOLD) ? 1.0f : 0;

        //Write final result to global memory
        dst[imageW * iy + ix] = make_color(lerpQ, 0, (1.0f - lerpQ), 0);
    };
}

extern "C"
void cuda_KNNdiag(
    TColor *d_dst,
    int imageW,
    int imageH,
    float Noise,
    float lerpC
)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    KNNdiag<<<grid, threads>>>(d_dst, imageW, imageH, Noise, lerpC);
}

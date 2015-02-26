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

float layer_1_0 = parrotInput[0] * -0.176098 + parrotInput[1] * -0.213037 + parrotInput[2] * 0.162896 + parrotInput[3] * 0.092744 + parrotInput[4] * -0.052455 + parrotInput[5] * -0.276945 + parrotInput[6] * 0.059870 + parrotInput[7] * -0.130744 + parrotInput[8] * -0.080881 + parrotInput[9] * 0.029798 + parrotInput[10] * 0.055122 + parrotInput[11] * 0.003821 + parrotInput[12] * -0.057243 + parrotInput[13] * -0.184090 + parrotInput[14] * 0.087984 + parrotInput[15] * -0.030698 + parrotInput[16] * 0.045266 + parrotInput[17] * 0.176448 + parrotInput[18] * 0.195008 + parrotInput[19] * -0.175087 + parrotInput[20] * 0.142187 + parrotInput[21] * 0.143309 + parrotInput[22] * 0.070349 + parrotInput[23] * 0.098052 + parrotInput[24] * 5.188277 + parrotInput[25] * 0.174840 + parrotInput[26] * -0.025251 + parrotInput[27] * -0.024480 + parrotInput[28] * -0.050815 + parrotInput[29] * -0.412459 + parrotInput[30] * 0.059549 + parrotInput[31] * -0.274402 + parrotInput[32] * -0.477339 + parrotInput[33] * 0.119806 + parrotInput[34] * -0.362079 + parrotInput[35] * -0.019488 + parrotInput[36] * 0.007272 + parrotInput[37] * 0.326232 + parrotInput[38] * 0.011120 + parrotInput[39] * 0.091180 + parrotInput[40] * -0.500858 + parrotInput[41] * 0.014205 + parrotInput[42] * 0.166200 + parrotInput[43] * 0.095575 + parrotInput[44] * -0.420920 + parrotInput[45] * -0.032414 + parrotInput[46] * -0.237159 + parrotInput[47] * -0.090963 + parrotInput[48] * -0.154114 + parrotInput[49] * 1500.000000 + parrotInput[50] * 1500.000000 + parrotInput[51] * 1500.000000 + parrotInput[52] * 1500.000000 + parrotInput[53] * 1500.000000 + parrotInput[54] * 1500.000000 + parrotInput[55] * 1500.000000 + parrotInput[56] * 1500.000000 + parrotInput[57] * 1500.000000 + parrotInput[58] * 1500.000000 + parrotInput[59] * 1500.000000 + parrotInput[60] * 1500.000000 + parrotInput[61] * 1500.000000 + parrotInput[62] * 1500.000000 + parrotInput[63] * 1500.000000 + parrotInput[64] * 1500.000000 + parrotInput[65] * 1500.000000 + parrotInput[66] * 1500.000000 + parrotInput[67] * 1500.000000 + parrotInput[68] * 1500.000000 + parrotInput[69] * 1500.000000 + parrotInput[70] * 1500.000000 + parrotInput[71] * 1500.000000 + parrotInput[72] * 1500.000000 + parrotInput[73] * 1500.000000 + parrotInput[74] * 1500.000000 + parrotInput[75] * 1500.000000 + parrotInput[76] * 1500.000000 + parrotInput[77] * 1500.000000 + parrotInput[78] * 1500.000000 + parrotInput[79] * 1500.000000 + parrotInput[80] * 1500.000000 + 1.0f * -3.453459;

float layer_1_1 = parrotInput[0] * 0.059702 + parrotInput[1] * 0.012386 + parrotInput[2] * 0.165436 + parrotInput[3] * 0.063401 + parrotInput[4] * 0.105801 + parrotInput[5] * -0.119256 + parrotInput[6] * 0.052568 + parrotInput[7] * 0.131922 + parrotInput[8] * 0.004427 + parrotInput[9] * -0.038390 + parrotInput[10] * 0.165491 + parrotInput[11] * 0.205722 + parrotInput[12] * 0.095147 + parrotInput[13] * 0.077176 + parrotInput[14] * 0.127320 + parrotInput[15] * 0.061834 + parrotInput[16] * 0.174639 + parrotInput[17] * 0.165196 + parrotInput[18] * 0.147996 + parrotInput[19] * 0.255573 + parrotInput[20] * 0.181281 + parrotInput[21] * 0.096087 + parrotInput[22] * 0.042694 + parrotInput[23] * 0.277867 + parrotInput[24] * 1.502027 + parrotInput[25] * 0.197524 + parrotInput[26] * 0.293482 + parrotInput[27] * 0.271828 + parrotInput[28] * 0.083429 + parrotInput[29] * -0.006115 + parrotInput[30] * 0.126187 + parrotInput[31] * 0.184244 + parrotInput[32] * 0.085778 + parrotInput[33] * 0.195016 + parrotInput[34] * -0.376742 + parrotInput[35] * 0.124913 + parrotInput[36] * 0.004493 + parrotInput[37] * 0.271078 + parrotInput[38] * 0.112402 + parrotInput[39] * 0.120517 + parrotInput[40] * 0.087675 + parrotInput[41] * 0.095663 + parrotInput[42] * -0.002554 + parrotInput[43] * 0.090044 + parrotInput[44] * 0.052589 + parrotInput[45] * 0.241323 + parrotInput[46] * -0.075961 + parrotInput[47] * -0.149040 + parrotInput[48] * -0.011347 + parrotInput[49] * 1500.000000 + parrotInput[50] * 1500.000000 + parrotInput[51] * 1500.000000 + parrotInput[52] * 1500.000000 + parrotInput[53] * 1500.000000 + parrotInput[54] * 1500.000000 + parrotInput[55] * 1500.000000 + parrotInput[56] * 1500.000000 + parrotInput[57] * 1500.000000 + parrotInput[58] * 1500.000000 + parrotInput[59] * 1500.000000 + parrotInput[60] * 1500.000000 + parrotInput[61] * 1500.000000 + parrotInput[62] * 1500.000000 + parrotInput[63] * 1500.000000 + parrotInput[64] * 1500.000000 + parrotInput[65] * 1500.000000 + parrotInput[66] * 1500.000000 + parrotInput[67] * 1500.000000 + parrotInput[68] * 1500.000000 + parrotInput[69] * 1500.000000 + parrotInput[70] * 1500.000000 + parrotInput[71] * 1500.000000 + parrotInput[72] * 1500.000000 + parrotInput[73] * 1500.000000 + parrotInput[74] * 1500.000000 + parrotInput[75] * 1500.000000 + parrotInput[76] * 1500.000000 + parrotInput[77] * 1500.000000 + parrotInput[78] * 1500.000000 + parrotInput[79] * 1500.000000 + parrotInput[80] * 1500.000000 + 1.0f * -4.774903;

float layer_1_2 = parrotInput[0] * 1.197808 + parrotInput[1] * 1.075821 + parrotInput[2] * 1.815905 + parrotInput[3] * 1.684144 + parrotInput[4] * 1.214210 + parrotInput[5] * 1.209286 + parrotInput[6] * 1.233584 + parrotInput[7] * 1.352515 + parrotInput[8] * 1.181854 + parrotInput[9] * 1.146126 + parrotInput[10] * 1.842557 + parrotInput[11] * 1.549105 + parrotInput[12] * 1.809610 + parrotInput[13] * 1.067104 + parrotInput[14] * 1.273846 + parrotInput[15] * 1.147620 + parrotInput[16] * 1.113262 + parrotInput[17] * 7.843622 + parrotInput[18] * 1.429745 + parrotInput[19] * 1.312906 + parrotInput[20] * 1.588966 + parrotInput[21] * 1.426155 + parrotInput[22] * 1.188257 + parrotInput[23] * 6.145129 + parrotInput[24] * 1500.000000 + parrotInput[25] * 5.498497 + parrotInput[26] * 3.687244 + parrotInput[27] * 2.051784 + parrotInput[28] * 1.071461 + parrotInput[29] * 1.198633 + parrotInput[30] * 6.161727 + parrotInput[31] * 3.571698 + parrotInput[32] * 5.399592 + parrotInput[33] * 1.084113 + parrotInput[34] * 1.842607 + parrotInput[35] * 1.246523 + parrotInput[36] * 1.090759 + parrotInput[37] * 5.320215 + parrotInput[38] * 1.265950 + parrotInput[39] * 2.799775 + parrotInput[40] * 1.149971 + parrotInput[41] * 1.203130 + parrotInput[42] * 1.270939 + parrotInput[43] * 1.435914 + parrotInput[44] * 1.105093 + parrotInput[45] * 31.484169 + parrotInput[46] * 1.234712 + parrotInput[47] * 1.149319 + parrotInput[48] * 1.418992 + parrotInput[49] * 1500.000000 + parrotInput[50] * 1500.000000 + parrotInput[51] * 1500.000000 + parrotInput[52] * 1500.000000 + parrotInput[53] * 1500.000000 + parrotInput[54] * 1500.000000 + parrotInput[55] * 1500.000000 + parrotInput[56] * 1500.000000 + parrotInput[57] * 1500.000000 + parrotInput[58] * 1500.000000 + parrotInput[59] * 1500.000000 + parrotInput[60] * 1500.000000 + parrotInput[61] * 1500.000000 + parrotInput[62] * 1500.000000 + parrotInput[63] * 1500.000000 + parrotInput[64] * 1500.000000 + parrotInput[65] * 1500.000000 + parrotInput[66] * 1500.000000 + parrotInput[67] * 1500.000000 + parrotInput[68] * 1500.000000 + parrotInput[69] * 1500.000000 + parrotInput[70] * 1500.000000 + parrotInput[71] * 1500.000000 + parrotInput[72] * 1500.000000 + parrotInput[73] * 1500.000000 + parrotInput[74] * 1500.000000 + parrotInput[75] * 1500.000000 + parrotInput[76] * 1500.000000 + parrotInput[77] * 1500.000000 + parrotInput[78] * 1500.000000 + parrotInput[79] * 1500.000000 + parrotInput[80] * 1500.000000 + 1.0f * -0.333092;

float layer_1_3 = parrotInput[0] * 0.054086 + parrotInput[1] * -0.079871 + parrotInput[2] * -0.103604 + parrotInput[3] * 0.534553 + parrotInput[4] * 0.187272 + parrotInput[5] * 0.133372 + parrotInput[6] * 0.008929 + parrotInput[7] * -0.018122 + parrotInput[8] * -0.073386 + parrotInput[9] * 0.083034 + parrotInput[10] * -0.028474 + parrotInput[11] * 0.098435 + parrotInput[12] * -0.176775 + parrotInput[13] * 0.109466 + parrotInput[14] * 0.061033 + parrotInput[15] * 0.416503 + parrotInput[16] * -0.098955 + parrotInput[17] * -0.145379 + parrotInput[18] * -0.150465 + parrotInput[19] * -0.040908 + parrotInput[20] * -0.083613 + parrotInput[21] * -0.218616 + parrotInput[22] * -0.019593 + parrotInput[23] * -0.132739 + parrotInput[24] * -3.576231 + parrotInput[25] * -0.105500 + parrotInput[26] * -0.137603 + parrotInput[27] * -0.096526 + parrotInput[28] * 0.039330 + parrotInput[29] * -0.111588 + parrotInput[30] * -0.352680 + parrotInput[31] * -0.170346 + parrotInput[32] * -0.082761 + parrotInput[33] * -0.084403 + parrotInput[34] * -0.035254 + parrotInput[35] * -0.071866 + parrotInput[36] * 0.099458 + parrotInput[37] * -0.075843 + parrotInput[38] * 0.014494 + parrotInput[39] * -0.049317 + parrotInput[40] * -0.109560 + parrotInput[41] * -0.123539 + parrotInput[42] * 0.307943 + parrotInput[43] * -0.011058 + parrotInput[44] * -0.011084 + parrotInput[45] * -0.056653 + parrotInput[46] * -0.111857 + parrotInput[47] * -0.085918 + parrotInput[48] * -0.164390 + parrotInput[49] * 1500.000000 + parrotInput[50] * 1500.000000 + parrotInput[51] * 1500.000000 + parrotInput[52] * 1500.000000 + parrotInput[53] * 1500.000000 + parrotInput[54] * 1500.000000 + parrotInput[55] * 1500.000000 + parrotInput[56] * 1500.000000 + parrotInput[57] * 1500.000000 + parrotInput[58] * 1500.000000 + parrotInput[59] * 1500.000000 + parrotInput[60] * 1500.000000 + parrotInput[61] * 1500.000000 + parrotInput[62] * 1500.000000 + parrotInput[63] * 1500.000000 + parrotInput[64] * 1500.000000 + parrotInput[65] * 1500.000000 + parrotInput[66] * 1500.000000 + parrotInput[67] * 1500.000000 + parrotInput[68] * 1500.000000 + parrotInput[69] * 1500.000000 + parrotInput[70] * 1500.000000 + parrotInput[71] * 1500.000000 + parrotInput[72] * 1500.000000 + parrotInput[73] * 1500.000000 + parrotInput[74] * 1500.000000 + parrotInput[75] * 1500.000000 + parrotInput[76] * 1500.000000 + parrotInput[77] * 1500.000000 + parrotInput[78] * 1500.000000 + parrotInput[79] * 1500.000000 + parrotInput[80] * 1500.000000 + 1.0f * 1.939350;

float layer_1_4 = parrotInput[0] * 0.144545 + parrotInput[1] * 0.072854 + parrotInput[2] * 0.119667 + parrotInput[3] * 0.093599 + parrotInput[4] * 0.139083 + parrotInput[5] * 0.178064 + parrotInput[6] * 0.223043 + parrotInput[7] * 0.108601 + parrotInput[8] * 0.064152 + parrotInput[9] * 0.164397 + parrotInput[10] * 0.092460 + parrotInput[11] * 0.205673 + parrotInput[12] * 0.155831 + parrotInput[13] * 0.100000 + parrotInput[14] * 0.080260 + parrotInput[15] * 0.130281 + parrotInput[16] * 0.112220 + parrotInput[17] * 0.198109 + parrotInput[18] * 0.177311 + parrotInput[19] * 0.208705 + parrotInput[20] * 0.112653 + parrotInput[21] * 0.033310 + parrotInput[22] * 0.200673 + parrotInput[23] * 0.305631 + parrotInput[24] * 1.497331 + parrotInput[25] * 0.192685 + parrotInput[26] * 0.087052 + parrotInput[27] * 0.220034 + parrotInput[28] * 0.226788 + parrotInput[29] * 0.071983 + parrotInput[30] * 0.232754 + parrotInput[31] * 0.160233 + parrotInput[32] * -0.570079 + parrotInput[33] * 0.066020 + parrotInput[34] * 0.143717 + parrotInput[35] * 0.207353 + parrotInput[36] * 0.043732 + parrotInput[37] * 0.109639 + parrotInput[38] * 0.060056 + parrotInput[39] * 0.194697 + parrotInput[40] * 0.114393 + parrotInput[41] * 0.053485 + parrotInput[42] * 0.086305 + parrotInput[43] * 0.119793 + parrotInput[44] * 0.129937 + parrotInput[45] * 0.211867 + parrotInput[46] * 0.176162 + parrotInput[47] * 0.065208 + parrotInput[48] * 0.097218 + parrotInput[49] * 1500.000000 + parrotInput[50] * 1500.000000 + parrotInput[51] * 1500.000000 + parrotInput[52] * 1500.000000 + parrotInput[53] * 1500.000000 + parrotInput[54] * 1500.000000 + parrotInput[55] * 1500.000000 + parrotInput[56] * 1500.000000 + parrotInput[57] * 1500.000000 + parrotInput[58] * 1500.000000 + parrotInput[59] * 1500.000000 + parrotInput[60] * 1500.000000 + parrotInput[61] * 1500.000000 + parrotInput[62] * 1500.000000 + parrotInput[63] * 1500.000000 + parrotInput[64] * 1500.000000 + parrotInput[65] * 1500.000000 + parrotInput[66] * 1500.000000 + parrotInput[67] * 1500.000000 + parrotInput[68] * 1500.000000 + parrotInput[69] * 1500.000000 + parrotInput[70] * 1500.000000 + parrotInput[71] * 1500.000000 + parrotInput[72] * 1500.000000 + parrotInput[73] * 1500.000000 + parrotInput[74] * 1500.000000 + parrotInput[75] * 1500.000000 + parrotInput[76] * 1500.000000 + parrotInput[77] * 1500.000000 + parrotInput[78] * 1500.000000 + parrotInput[79] * 1500.000000 + parrotInput[80] * 1500.000000 + 1.0f * -5.894183;

float layer_1_5 = parrotInput[0] * 0.060305 + parrotInput[1] * 0.152859 + parrotInput[2] * -0.023070 + parrotInput[3] * 0.110235 + parrotInput[4] * 0.106042 + parrotInput[5] * 0.094811 + parrotInput[6] * 0.177957 + parrotInput[7] * -0.052346 + parrotInput[8] * -0.041812 + parrotInput[9] * -0.077201 + parrotInput[10] * 0.008608 + parrotInput[11] * 0.005700 + parrotInput[12] * -0.173124 + parrotInput[13] * 0.029568 + parrotInput[14] * 0.074611 + parrotInput[15] * 0.137043 + parrotInput[16] * -0.068740 + parrotInput[17] * -0.169142 + parrotInput[18] * -0.191419 + parrotInput[19] * -0.016596 + parrotInput[20] * -0.098174 + parrotInput[21] * 0.005803 + parrotInput[22] * 0.069566 + parrotInput[23] * -0.407964 + parrotInput[24] * -3.143995 + parrotInput[25] * -0.287286 + parrotInput[26] * -0.165163 + parrotInput[27] * -0.111520 + parrotInput[28] * 0.143082 + parrotInput[29] * 0.070434 + parrotInput[30] * -0.282525 + parrotInput[31] * -0.180626 + parrotInput[32] * -0.335739 + parrotInput[33] * -0.119120 + parrotInput[34] * 0.517907 + parrotInput[35] * -0.021795 + parrotInput[36] * 0.021734 + parrotInput[37] * -0.230869 + parrotInput[38] * -0.087904 + parrotInput[39] * -0.179749 + parrotInput[40] * -0.276809 + parrotInput[41] * -0.010764 + parrotInput[42] * -0.006197 + parrotInput[43] * -0.044433 + parrotInput[44] * -0.152282 + parrotInput[45] * 0.031550 + parrotInput[46] * 0.031152 + parrotInput[47] * -0.111074 + parrotInput[48] * 0.031875 + parrotInput[49] * 1500.000000 + parrotInput[50] * 1500.000000 + parrotInput[51] * 1500.000000 + parrotInput[52] * 1500.000000 + parrotInput[53] * 1500.000000 + parrotInput[54] * 1500.000000 + parrotInput[55] * 1500.000000 + parrotInput[56] * 1500.000000 + parrotInput[57] * 1500.000000 + parrotInput[58] * 1500.000000 + parrotInput[59] * 1500.000000 + parrotInput[60] * 1500.000000 + parrotInput[61] * 1500.000000 + parrotInput[62] * 1500.000000 + parrotInput[63] * 1500.000000 + parrotInput[64] * 1500.000000 + parrotInput[65] * 1500.000000 + parrotInput[66] * 1500.000000 + parrotInput[67] * 1500.000000 + parrotInput[68] * 1500.000000 + parrotInput[69] * 1500.000000 + parrotInput[70] * 1500.000000 + parrotInput[71] * 1500.000000 + parrotInput[72] * 1500.000000 + parrotInput[73] * 1500.000000 + parrotInput[74] * 1500.000000 + parrotInput[75] * 1500.000000 + parrotInput[76] * 1500.000000 + parrotInput[77] * 1500.000000 + parrotInput[78] * 1500.000000 + parrotInput[79] * 1500.000000 + parrotInput[80] * 1500.000000 + 1.0f * 1.900337;

float layer_1_6 = parrotInput[0] * 0.714254 + parrotInput[1] * 0.960973 + parrotInput[2] * 1.410457 + parrotInput[3] * 1.233969 + parrotInput[4] * 1.081554 + parrotInput[5] * 20.176991 + parrotInput[6] * 1.247722 + parrotInput[7] * 0.936441 + parrotInput[8] * 0.956027 + parrotInput[9] * 1.064349 + parrotInput[10] * 1.220200 + parrotInput[11] * 1.162662 + parrotInput[12] * 3.961239 + parrotInput[13] * 1.034354 + parrotInput[14] * 1.012258 + parrotInput[15] * 0.852589 + parrotInput[16] * 1.088163 + parrotInput[17] * 1.084190 + parrotInput[18] * 1.210777 + parrotInput[19] * 1.111514 + parrotInput[20] * 1.142427 + parrotInput[21] * 0.988460 + parrotInput[22] * 0.843780 + parrotInput[23] * -7.203819 + parrotInput[24] * 118.047310 + parrotInput[25] * 1.127673 + parrotInput[26] * 1.230497 + parrotInput[27] * 1.434803 + parrotInput[28] * -3.050859 + parrotInput[29] * 1.016821 + parrotInput[30] * 1.149210 + parrotInput[31] * 1.055359 + parrotInput[32] * 1.083919 + parrotInput[33] * 1.036572 + parrotInput[34] * 14.817440 + parrotInput[35] * -0.281530 + parrotInput[36] * 0.854791 + parrotInput[37] * 1.127031 + parrotInput[38] * 1.227009 + parrotInput[39] * 1.507550 + parrotInput[40] * 23.253736 + parrotInput[41] * 1.005270 + parrotInput[42] * -1.589525 + parrotInput[43] * 1.038221 + parrotInput[44] * 0.939065 + parrotInput[45] * 1.021985 + parrotInput[46] * 1.128630 + parrotInput[47] * 1.199664 + parrotInput[48] * 15.574191 + parrotInput[49] * 1500.000000 + parrotInput[50] * 1500.000000 + parrotInput[51] * 1500.000000 + parrotInput[52] * 1500.000000 + parrotInput[53] * 1500.000000 + parrotInput[54] * 1500.000000 + parrotInput[55] * 1500.000000 + parrotInput[56] * 1500.000000 + parrotInput[57] * 1500.000000 + parrotInput[58] * 1500.000000 + parrotInput[59] * 1500.000000 + parrotInput[60] * 1500.000000 + parrotInput[61] * 1500.000000 + parrotInput[62] * 1500.000000 + parrotInput[63] * 1500.000000 + parrotInput[64] * 1500.000000 + parrotInput[65] * 1500.000000 + parrotInput[66] * 1500.000000 + parrotInput[67] * 1500.000000 + parrotInput[68] * 1500.000000 + parrotInput[69] * 1500.000000 + parrotInput[70] * 1500.000000 + parrotInput[71] * 1500.000000 + parrotInput[72] * 1500.000000 + parrotInput[73] * 1500.000000 + parrotInput[74] * 1500.000000 + parrotInput[75] * 1500.000000 + parrotInput[76] * 1500.000000 + parrotInput[77] * 1500.000000 + parrotInput[78] * 1500.000000 + parrotInput[79] * 1500.000000 + parrotInput[80] * 1500.000000 + 1.0f * -12.515392;

float layer_1_7 = parrotInput[0] * 0.423755 + parrotInput[1] * -0.124227 + parrotInput[2] * -0.203469 + parrotInput[3] * -2.114908 + parrotInput[4] * -0.562813 + parrotInput[5] * -2.074915 + parrotInput[6] * -0.087033 + parrotInput[7] * -0.082069 + parrotInput[8] * -0.057030 + parrotInput[9] * -0.932805 + parrotInput[10] * -1.457265 + parrotInput[11] * -0.257215 + parrotInput[12] * -0.601405 + parrotInput[13] * 0.061286 + parrotInput[14] * -0.054362 + parrotInput[15] * 0.423352 + parrotInput[16] * -0.037011 + parrotInput[17] * -0.548316 + parrotInput[18] * -0.153341 + parrotInput[19] * -0.480100 + parrotInput[20] * -0.076640 + parrotInput[21] * -0.262564 + parrotInput[22] * -0.043164 + parrotInput[23] * -0.013336 + parrotInput[24] * -4.593573 + parrotInput[25] * -0.081963 + parrotInput[26] * -0.192472 + parrotInput[27] * 0.002628 + parrotInput[28] * 0.038806 + parrotInput[29] * -0.772643 + parrotInput[30] * -0.335878 + parrotInput[31] * -0.609503 + parrotInput[32] * -0.210545 + parrotInput[33] * 0.453551 + parrotInput[34] * -0.854270 + parrotInput[35] * 0.051319 + parrotInput[36] * 0.196126 + parrotInput[37] * -0.331115 + parrotInput[38] * -0.324528 + parrotInput[39] * -0.626942 + parrotInput[40] * -0.183984 + parrotInput[41] * 0.297944 + parrotInput[42] * -0.032733 + parrotInput[43] * -0.203119 + parrotInput[44] * 0.114369 + parrotInput[45] * -0.389939 + parrotInput[46] * -0.242133 + parrotInput[47] * -0.174459 + parrotInput[48] * -0.487560 + parrotInput[49] * 1500.000000 + parrotInput[50] * 1500.000000 + parrotInput[51] * 1500.000000 + parrotInput[52] * 1500.000000 + parrotInput[53] * 1500.000000 + parrotInput[54] * 1500.000000 + parrotInput[55] * 1500.000000 + parrotInput[56] * 1500.000000 + parrotInput[57] * 1500.000000 + parrotInput[58] * 1500.000000 + parrotInput[59] * 1500.000000 + parrotInput[60] * 1500.000000 + parrotInput[61] * 1500.000000 + parrotInput[62] * 1500.000000 + parrotInput[63] * 1500.000000 + parrotInput[64] * 1500.000000 + parrotInput[65] * 1500.000000 + parrotInput[66] * 1500.000000 + parrotInput[67] * 1500.000000 + parrotInput[68] * 1500.000000 + parrotInput[69] * 1500.000000 + parrotInput[70] * 1500.000000 + parrotInput[71] * 1500.000000 + parrotInput[72] * 1500.000000 + parrotInput[73] * 1500.000000 + parrotInput[74] * 1500.000000 + parrotInput[75] * 1500.000000 + parrotInput[76] * 1500.000000 + parrotInput[77] * 1500.000000 + parrotInput[78] * 1500.000000 + parrotInput[79] * 1500.000000 + parrotInput[80] * 1500.000000 + 1.0f * 1.827678;

float layer_2_0 = sigmoid(layer_1_0, 0.500000) * 1.408810 + sigmoid(layer_1_1, 0.500000) * 0.740418 + sigmoid(layer_1_2, 0.500000) * 0.205745 + sigmoid(layer_1_3, 0.500000) * -1.294341 + sigmoid(layer_1_4, 0.500000) * 0.834958 + sigmoid(layer_1_5, 0.500000) * -1.111145 + sigmoid(layer_1_6, 0.500000) * 0.220880 + sigmoid(layer_1_7, 0.500000) * -1.802231 + 1.0f * 0.012682;

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

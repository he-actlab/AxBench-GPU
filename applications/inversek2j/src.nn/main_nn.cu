#include "../../../headers/activationFunction.h"

// Designed by: Amir Yazdanbakhsh
// Date: March 26th - 2015
// Alternative Computing Technologies Lab.
// Georgia Institute of Technology


#include "stdlib.h"
#include <fstream>
#include <iostream>
#include <cstddef>

// Cuda Libraries
#include <cuda_runtime_api.h>
#include <cuda.h>

#define MAX_LOOP 1000
#define MAX_DIFF 0.15f
#define NUM_JOINTS 3
#define PI 3.14159265358979f
#define NUM_JOINTS_P1 (NUM_JOINTS + 1)

using namespace std;

__global__ void invkin_kernel(float *xTarget_in, float *yTarget_in, float *angles, int size, float err_thresh)
{

	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	if(idx < size)
	{
		float parrotInput[2];
    	float parrotOutput[3];
    	float angle_out[NUM_JOINTS];

    	for(int i = 0; i < NUM_JOINTS; i++)
    	{
  			angle_out[i] = 0.0;
    	}

    	float max_err 	= err_thresh * (float)(NUM_JOINTS);
    	float err 		= max_err + 1.f; // initialize error to something greater than error threshold

    	parrotInput[0] = xTarget_in[idx];
    	parrotInput[1] = yTarget_in[idx];

float layer_1_0 = parrotInput[0] * -1.798665 + parrotInput[1] * 4.560411 + 1.0f * 1.149290;

float layer_1_1 = parrotInput[0] * -1.262481 + parrotInput[1] * -3.736870 + 1.0f * 0.285704;

float layer_1_2 = parrotInput[0] * -1.223271 + parrotInput[1] * -3.642362 + 1.0f * 0.143369;

float layer_1_3 = parrotInput[0] * -3.280792 + parrotInput[1] * 2.001388 + 1.0f * 1.805166;

float layer_1_4 = parrotInput[0] * -2.898585 + parrotInput[1] * 1.940286 + 1.0f * 1.642777;

float layer_1_5 = parrotInput[0] * -5.762485 + parrotInput[1] * -6.614917 + 1.0f * 0.828460;

float layer_1_6 = parrotInput[0] * -7.034237 + parrotInput[1] * 0.076823 + 1.0f * 0.421022;

float layer_1_7 = parrotInput[0] * -5.059394 + parrotInput[1] * 1.127199 + 1.0f * 0.700742;

float layer_1_8 = parrotInput[0] * -1.756325 + parrotInput[1] * 4.784623 + 1.0f * 0.978976;

float layer_1_9 = parrotInput[0] * -6.691505 + parrotInput[1] * -1.578492 + 1.0f * 0.490338;

float layer_1_10 = parrotInput[0] * -2.959693 + parrotInput[1] * 0.825397 + 1.0f * 1.853845;

float layer_1_11 = parrotInput[0] * -3.711463 + parrotInput[1] * 1.052303 + 1.0f * 1.110465;

float layer_1_12 = parrotInput[0] * -1.804598 + parrotInput[1] * 3.503114 + 1.0f * 0.957473;

float layer_1_13 = parrotInput[0] * -1.629346 + parrotInput[1] * -1.502572 + 1.0f * 2.276568;

float layer_1_14 = parrotInput[0] * -6.138941 + parrotInput[1] * -3.679501 + 1.0f * 0.655275;

float layer_1_15 = parrotInput[0] * -2.587424 + parrotInput[1] * 3.923578 + 1.0f * -0.023949;

float layer_2_0 = sigmoid(layer_1_0, 0.500000) * 0.403742 + sigmoid(layer_1_1, 0.500000) * 1.047462 + sigmoid(layer_1_2, 0.500000) * 1.048148 + sigmoid(layer_1_3, 0.500000) * 0.427120 + sigmoid(layer_1_4, 0.500000) * 0.488172 + sigmoid(layer_1_5, 0.500000) * 1.157142 + sigmoid(layer_1_6, 0.500000) * 0.916205 + sigmoid(layer_1_7, 0.500000) * 0.871136 + sigmoid(layer_1_8, 0.500000) * 0.423161 + sigmoid(layer_1_9, 0.500000) * 1.136492 + sigmoid(layer_1_10, 0.500000) * 0.436868 + sigmoid(layer_1_11, 0.500000) * 0.859039 + sigmoid(layer_1_12, 0.500000) * 0.379688 + sigmoid(layer_1_13, 0.500000) * -1.204959 + sigmoid(layer_1_14, 0.500000) * 1.671170 + sigmoid(layer_1_15, 0.500000) * 1.251833 + 1.0f * 0.308960;

layer_2_0 = linear(layer_2_0, 0.5);

float layer_2_1 = sigmoid(layer_1_0, 0.500000) * 1.686440 + sigmoid(layer_1_1, 0.500000) * 3.135052 + sigmoid(layer_1_2, 0.500000) * 3.180274 + sigmoid(layer_1_3, 0.500000) * 1.642216 + sigmoid(layer_1_4, 0.500000) * 1.486881 + sigmoid(layer_1_5, 0.500000) * 2.441879 + sigmoid(layer_1_6, 0.500000) * 1.347186 + sigmoid(layer_1_7, 0.500000) * 1.998285 + sigmoid(layer_1_8, 0.500000) * 1.717950 + sigmoid(layer_1_9, 0.500000) * 2.358272 + sigmoid(layer_1_10, 0.500000) * 1.416066 + sigmoid(layer_1_11, 0.500000) * 2.602691 + sigmoid(layer_1_12, 0.500000) * 1.568467 + sigmoid(layer_1_13, 0.500000) * 2.274102 + sigmoid(layer_1_14, 0.500000) * 2.580288 + sigmoid(layer_1_15, 0.500000) * 3.413414 + 1.0f * 0.776277;

layer_2_1 = linear(layer_2_1, 0.5);

float layer_2_2 = sigmoid(layer_1_0, 0.500000) * 6.285129 + sigmoid(layer_1_1, 0.500000) * 12.153250 + sigmoid(layer_1_2, 0.500000) * 7.822494 + sigmoid(layer_1_3, 0.500000) * 6.273594 + sigmoid(layer_1_4, 0.500000) * 6.362993 + sigmoid(layer_1_5, 0.500000) * 7.813550 + sigmoid(layer_1_6, 0.500000) * 5.648221 + sigmoid(layer_1_7, 0.500000) * 4.771950 + sigmoid(layer_1_8, 0.500000) * 6.301631 + sigmoid(layer_1_9, 0.500000) * 5.755622 + sigmoid(layer_1_10, 0.500000) * 6.252254 + sigmoid(layer_1_11, 0.500000) * 6.237138 + sigmoid(layer_1_12, 0.500000) * 6.359992 + sigmoid(layer_1_13, 0.500000) * 11.693406 + sigmoid(layer_1_14, 0.500000) * 6.609825 + sigmoid(layer_1_15, 0.500000) * 6.263258 + 1.0f * 6.270045;

layer_2_2 = linear(layer_2_2, 0.5);

parrotOutput[0] = layer_2_0;

parrotOutput[1] = layer_2_1;

parrotOutput[2] = layer_2_2;

// parrotOutput[2] = layer_2_2;
// 
// 		//float max_err = err_thresh * (float)(NUM_JOINTS);
// 		//float err = max_err + 1.f;
// 
// 		// Initialize x and y data
// 		float xData[NUM_JOINTS_P1];
// 		float yData[NUM_JOINTS_P1];
// 
// 		for (int i = 0 ; i < NUM_JOINTS_P1; i++)
// 		{
// 			xData[i] = i;
// 			yData[i] = 0.f;
// 		}
// 
// 		for(int curr_loop = 0; curr_loop < MAX_LOOP; curr_loop++)
// 		{
// 			for (int iter = NUM_JOINTS; iter > 0; iter--)
// 			{
// 				float pe_x = xData[NUM_JOINTS];
// 				float pe_y = yData[NUM_JOINTS];
// 				float pc_x = xData[iter-1];
// 				float pc_y = yData[iter-1];
// 				float diff_pe_pc_x = pe_x - pc_x;
// 				float diff_pe_pc_y = pe_y - pc_y;
// 				float diff_tgt_pc_x = xTarget_in[idx] - pc_x;
// 				float diff_tgt_pc_y = yTarget_in[idx] - pc_y;
// 				float len_diff_pe_pc = sqrt(diff_pe_pc_x * diff_pe_pc_x + diff_pe_pc_y * diff_pe_pc_y);
// 				float len_diff_tgt_pc = sqrt(diff_tgt_pc_x * diff_tgt_pc_x + diff_tgt_pc_y * diff_tgt_pc_y);
// 				float a_x = diff_pe_pc_x / len_diff_pe_pc;
// 				float a_y = diff_pe_pc_y / len_diff_pe_pc;
// 				float b_x = diff_tgt_pc_x / len_diff_tgt_pc;
// 				float b_y = diff_tgt_pc_y / len_diff_tgt_pc;
// 				float a_dot_b = a_x * b_x + a_y * b_y;
// 				if (a_dot_b > 1.f)
// 					a_dot_b = 1.f;
// 				else if (a_dot_b < -1.f)
// 					a_dot_b = -1.f;
// 				float angle = acos(a_dot_b) * (180.f / PI);
// 				// Determine angle direction
// 				float direction = a_x * b_y - a_y * b_x;
// 				if (direction < 0.f)
// 					angle = -angle;
// 				// Make the result look more natural (these checks may be omitted)
// 				// if (angle > 30.f)
// 				// 	angle = 30.f;
// 				// else if (angle < -30.f)
// 				// 	angle = -30.f;
// 				// Save angle
// 				angle_out[iter - 1] = angle;
// 				for (int i = 0; i < NUM_JOINTS; i++)
// 				{
// 					if(i < NUM_JOINTS - 1)
// 					{
// 						angle_out[i+1] += angle_out[i];
// 					}
// 				}
// 			}
// 		}
// 		parrotOutput[0] = angle_out[0] / 30.0;
// 		parrotOutput[1] = angle_out[1] / 30.0;
// 		parrotOutput[2] = angle_out[2] / 30.0;
// 
// #pragma parrot(output, "invkin_kernel", [3]<-1.0; 1.0>parrotOutput)

		angle_out[0] = parrotOutput[0] * 30.0;
		angle_out[1] = parrotOutput[1] * 30.0;
		angle_out[2] = parrotOutput[2] * 30.0;

		angles[idx * NUM_JOINTS + 0] = angle_out[0];
		angles[idx * NUM_JOINTS + 1] = angle_out[1];
		angles[idx * NUM_JOINTS + 2] = angle_out[2];
	}
}
int main(int argc, char* argv[])
{
	if(argc != 4)
	{
		std::cerr << "Usage: ./invkin.out <input file coefficients> <output file> <error threshold>" << std::endl;
		exit(EXIT_FAILURE);
	}

	float* xTarget_in_h;
	float* yTarget_in_h;
	float* angle_out_h;

	cudaError_t cudaStatus;

	int data_size = 0;

	// process the files
	ifstream coordinate_in_file (argv[1]);
	ofstream angle_out_file (argv[2]);
	float err_thresh = atof(argv[3]);


	if(coordinate_in_file.is_open())
	{
		coordinate_in_file >> data_size;
		std::cout << "# Data Size = " << data_size << std::endl;
	}

	// allocate the memory
	xTarget_in_h = new (nothrow) float[data_size];
	if(xTarget_in_h == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	yTarget_in_h = new (nothrow) float[data_size];
	if(yTarget_in_h == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	angle_out_h = new (nothrow) float[data_size*NUM_JOINTS];
	if(angle_out_h == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}


	// Prepare
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	// add data to the arrays
	float xTarget_tmp, yTarget_tmp;
	int coeff_index = 0;
	while(coeff_index < data_size)
	{
		coordinate_in_file >> xTarget_tmp >> yTarget_tmp;

		for(int i = 0; i < NUM_JOINTS ; i++)
		{
			angle_out_h[coeff_index * NUM_JOINTS + i] = 0.0;
		}

		xTarget_in_h[coeff_index] = xTarget_tmp;
		yTarget_in_h[coeff_index++] = yTarget_tmp;
	}


	std::cout << "# Coordinates are read from file..." << std::endl;

	// memory allocations on the host
	float 	*xTarget_in_d,
			*yTarget_in_d;
	float 	*angle_out_d;

	cudaMalloc((void**) &xTarget_in_d, data_size * sizeof(float));
	cudaMalloc((void**) &yTarget_in_d, data_size * sizeof(float));
	cudaMalloc((void**) &angle_out_d,  data_size * NUM_JOINTS * sizeof(float));

	std::cout << "# Memory allocation on GPU is done..." << std::endl;

	cudaMemcpy(xTarget_in_d, xTarget_in_h, data_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(yTarget_in_d, yTarget_in_h, data_size * sizeof(float), cudaMemcpyHostToDevice);

	std::cout << "# Data are transfered to GPU..." << std::endl;

	dim3 dimBlock	( 512, 1 );
	dim3 dimGrid	( data_size / 512, 1 );


	cudaEventRecord(start, 0);

#pragma parrot.start("invkin_kernel")

	invkin_kernel<<<dimGrid, dimBlock>>>(xTarget_in_d, yTarget_in_d, angle_out_d, data_size, err_thresh);

#pragma parrot.end("invkin_kernel")

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
       	std::cout << "Something was wrong! Error code: " << cudaStatus << std::endl;
    }

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	std::cout << "# Elapsed Time in `nrpoly3` kernel = " << elapsedTime << std::endl;
	std::cout << "# GPU computation is done ..." << std::endl;

	cudaMemcpy(angle_out_h, angle_out_d, data_size * NUM_JOINTS * sizeof(float), cudaMemcpyDeviceToHost);

	for(int i = 0; i < data_size; i++)
	{
		angle_out_file << xTarget_in_h[i] << " " << yTarget_in_h[i] << " ";
		for(int j = 0 ; j < NUM_JOINTS; j++)
		{
			angle_out_file << angle_out_h[i * NUM_JOINTS + j] << " ";
		}
		angle_out_file << std::endl;
	}

	// close files
	coordinate_in_file.close();
	angle_out_file.close();

	// de-allocate the memory
	delete[] xTarget_in_h;
	delete[] yTarget_in_h;
	delete[] angle_out_h;

	// de-allocate cuda memory
	cudaFree(xTarget_in_d);
	cudaFree(yTarget_in_d);
	cudaFree(angle_out_d);

	std::cout << "Thank you..." << std::endl;
}

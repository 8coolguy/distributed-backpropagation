#include <cuda.h>
#include <iostream>
#include <assert.h>
#include <time.h>
#include "Layer.h"
#include "Neural_Network.h"
#include "Cost_Function.h"
#include "Activation_Function.h"
#include "wrapper.h"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
	std::cout << "GPUassert: " << cudaGetErrorString(code) << \
		", " << file << ", " << line << std::endl;
        if (abort)
            exit(code);
    }
}
#define gpuErrorCheck(ans) {gpuAssert((ans), __FILE__, __LINE__);}

/*
*
* Inferene Step
*
*
*/
__global__ void parallel_forward(double * inputs, double * bias, int output_dim, int input_dim, double * intermediate, double * weights, double * outputs, Activation_Function *activation_function)
{
	int row = threadIdx.x + (blockDim.x * blockIdx.x);	
	int col = threadIdx.y + (blockDim.y * blockIdx.y);	
	if(row >= output_dim) return;
	if(col >= input_dim) return;

	//int index = row * input_dim + col;
	if(col == 0) {
		intermediate[row] = bias[row];
		for(int i = 0; i < input_dim; i++){
			int w_index = row * input_dim + i;
			double contribution = weights[w_index] *  inputs[i];
			intermediate[row] += contribution;
		}
	}
	//atomicAdd((float*) &intermediate[row], (float)(weights[index] * inputs[col]));
	__syncthreads();
	if(col == 0)
		outputs[row] = 1.0 / (1.0 + exp(-1 * intermediate[row]));
}
void forward_wrapper(double * input, double * bias, int output_dim, int input_dim, double * intermediate, double * weights, double * output, Activation_Function *activation_function){
	double *d_input;
	cudaMalloc((void**)&d_input, input_dim * sizeof(double));
	cudaMemcpy(d_input, input, sizeof(double) * input_dim, cudaMemcpyHostToDevice);

	dim3 block_size(32, 32);
    	dim3 grid_size((output_dim - 1)/32 + 1, (input_dim - 1)/32 + 1);
	parallel_forward<<<grid_size, block_size>>>(d_input, bias, output_dim, input_dim, intermediate, weights, output, activation_function);

	cudaDeviceSynchronize();

}
/*
*
* Backpropagation Step
*
*/

__global__ void parallel_backward(double * activations, double * actual_outputs, double * bias, Cost_Function * f, double learning_rate, int output_dim, int input_dim, double * intermediate, double * weights, double * output, Activation_Function *activation_function, bool final_layer, double * error_term)
{
	int row = threadIdx.x + blockDim.x * blockIdx.x;	
	int col = threadIdx.y + blockDim.y * blockIdx.y;	
	if(row >= output_dim) return;
	if(col >= input_dim) return;

	__shared__ double od[32];
	__shared__ double ig[32];
	
	int index = row * input_dim + col;	

	if (col == 0){
		if(final_layer) od[threadIdx.x] = 2 * (output[row] - actual_outputs[row]);
		else od[threadIdx.x] = actual_outputs[row];
	}
	if(col == 1){
		double sigmoid = 1/(1+ exp(-1 * intermediate[row]));
		ig[threadIdx.x] = sigmoid * (1 - sigmoid);
	}
	if(col == 2) {
		error_term[row] = 0;
		for(int i = 0; i < input_dim; i++){
			int w_index = row * input_dim + i;
			double contribution = weights[w_index];
			error_term[row] += contribution;
		}
	}
	__syncthreads();
	weights[index] -= learning_rate * activations[col] * od[threadIdx.x] * ig[threadIdx.x];
	//printf("%d %d %f \t\n", final_layer, index, weights[index]);
	if (col == 0) error_term[row] = od[threadIdx.x] * ig[threadIdx.x] * error_term[row];
	if (col == 1) bias[row] -= learning_rate * od[threadIdx.x] * ig[threadIdx.x];
	
}
void backward_wrapper(double * activations, double * actual_outputs, double * bias, Cost_Function * f, double learning_rate, int output_dim, int input_dim, double * intermediate, double * weights, double * output, Activation_Function *activation_function, bool final_layer, double * error_term){
	double *d_activations,*d_actual_outputs;

	//inputs
	cudaMalloc((void**)&d_activations, input_dim * sizeof(double));
	cudaMalloc((void**)&d_actual_outputs, output_dim* sizeof(double));

	cudaMemcpy(d_activations, activations, sizeof(double) * input_dim, cudaMemcpyHostToDevice);
	if(final_layer) cudaMemcpy(d_actual_outputs, actual_outputs, sizeof(double) * output_dim, cudaMemcpyHostToDevice);
	else d_actual_outputs = actual_outputs;


	dim3 block_size(32, 32);
    	dim3 grid_size((output_dim - 1)/32 + 1, (input_dim - 1)/32 + 1);
	parallel_backward<<<grid_size, block_size>>>(d_activations, d_actual_outputs, bias, f, learning_rate, output_dim, input_dim, intermediate, weights, output, activation_function, final_layer, error_term);
	cudaDeviceSynchronize();
	gpuErrorCheck(cudaGetLastError());

	// Free device memory
    	cudaFree(d_activations);
    	cudaFree(d_actual_outputs);
}

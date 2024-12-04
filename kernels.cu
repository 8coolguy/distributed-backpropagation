#include <cuda.h>
#include <iostream>
#include <assert.h>
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

//void Layer::forward(double* inputs){
//	for(int row = 0; row < output_dim; row++){
//        _intermediate[row] = _bias[row];
//		for(int col = 0; col < input_dim; col++){
//			int index = row * input_dim + col;
//			_intermediate[row] += _weights[index] * inputs[col];
//		}
//		_outputs[row] = _activation_function->evaluate(_intermediate[row]);
//	}
//}
__global__ void parallel_forward(double * inputs, double * bias, int output_dim, int input_dim, double * intermediate, double * weights, double * outputs, Activation_Function *activation_function)
{
	int row = threadIdx.x + (blockDim.x * blockIdx.x);	
	int col = threadIdx.y + (blockDim.y * blockIdx.y);	
	if(row >= output_dim) return;
	if(col >= input_dim) return;

	int index = row * input_dim + col;
	//printf("Cuda %d %d %f %f %f\n", row, col, intermediate[row], weights[index], inputs[col]);
	atomicAdd((float*) &intermediate[row], (float) (weights[index] * inputs[col]));
	__syncthreads();
	if(col == 0)
		outputs[row] = 1.0 / (1.0 + exp(-1 * intermediate[row]));
}
void forward_wrapper(double * input, double * bias, int output_dim, int input_dim, double * intermediate, double * weights, double * output, Activation_Function *activation_function){

	double *d_input, *d_weights, *d_bias, *d_intermediate, *d_output;
	cudaMalloc((void**)&d_input, input_dim * sizeof(double));
	cudaMalloc((void**)&d_weights, input_dim * output_dim * sizeof(double));
	cudaMalloc((void**)&d_bias, input_dim * sizeof(double)); 
	cudaMalloc((void**)&d_intermediate, output_dim*sizeof(double));
	cudaMalloc((void**)&d_output, output_dim * sizeof(double));

	cudaMemcpy(d_input, input, sizeof(double) * input_dim, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, weights, sizeof(double) * input_dim * output_dim, cudaMemcpyHostToDevice);
	cudaMemcpy(d_intermediate, bias, sizeof(double) * output_dim, cudaMemcpyHostToDevice);

	dim3 block_size(32, 32);
    	dim3 grid_size((output_dim - 1)/32 + 1, (input_dim - 1)/32 + 1);
	parallel_forward<<<grid_size, block_size>>>(d_input, d_bias, output_dim, input_dim, d_intermediate, d_weights, d_output, activation_function);

	cudaDeviceSynchronize();
	cudaMemcpy(output, d_output, sizeof(double)  * output_dim, cudaMemcpyDeviceToHost);
	cudaMemcpy(intermediate, d_intermediate, sizeof(double) * output_dim, cudaMemcpyDeviceToHost);

	// Free device memory
    	cudaFree(d_input);
    	cudaFree(d_weights);
    	cudaFree(d_bias);
    	cudaFree(d_intermediate);
    	cudaFree(d_output);

}

//void Layer::backward(double* actual_outputs, Cost_Function *f, double learning_rate){
//    double output_derivatives[output_dim];
//    double intermediate_gradient[output_dim];
//    
//    for (int i = 0; i < output_dim; i++) {
//        output_derivatives[i] = f->derivative(actual_outputs[i], _outputs[i]);
//        intermediate_gradient[i] = _activation_function->derivative(_intermediate[i]);
//    }
//
//    for (int row = 0; row < output_dim; row++) {
//        for (int col = 0; col < input_dim; col++) {
//            int index = row * input_dim + col;
//            _weights[index] -= learning_rate * _weights[index] * output_derivatives[row] * intermediate_gradient[row];
//        }
//        _bias[row] -= learning_rate * output_derivatives[row];
//    }
//}
/*
__global__ void parallel_backward(double * actual_outputs, double * bias, double * output_derivatives, double * intermediate_gradient, Cost_Function * f, int learning_rate, int input_dim, double * intermediate, double * weights, double * output, Activation_Function *activation_function)
{
	int row = threadIdx.x + blockDim.x * blockIdx.x;	
	int col = threadIdx.y + blockDim.y * blockIdx.y;	
	int index = row * input_dim + col;	

	if (col == 0) output_derivatives[row] = f->derivative(actual_outputs[row], _outputs[row]);
	if (col == 1) intermediate_gradient[row] = _activation_function->derivative(_intermediate[row]);
	__syncthreads();

	weights[index] -= learning_rate * weights[index] * output_derivatives[row] * intermediate_gradient[row];
	if (col == 0) bias[row] -= learning_rate * output_derivatives[row];
	__syncthreads();
	
}


*/

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
	atomicAdd((float*) &intermediate[row], (float)(weights[index] * inputs[col]));
	__syncthreads();
	if(col == 0)
		outputs[row] = 1.0 / (1.0 + exp(-1 * intermediate[row]));
}
void forward_wrapper(double * input, double * bias, int output_dim, int input_dim, double * intermediate, double * weights, double * output, Activation_Function *activation_function){

	double *d_input, *d_weights, *d_bias, *d_intermediate, *d_output;
	cudaMalloc((void**)&d_input, input_dim * sizeof(double));
	cudaMalloc((void**)&d_weights, input_dim * output_dim * sizeof(double));
	cudaMalloc((void**)&d_bias, output_dim * sizeof(double)); 
	cudaMalloc((void**)&d_intermediate, output_dim*sizeof(double));
	gpuErrorCheck(cudaMalloc((void**)&d_output, output_dim * sizeof(double)));

	cudaMemcpy(d_input, input, sizeof(double) * input_dim, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, weights, sizeof(double) * input_dim * output_dim, cudaMemcpyHostToDevice);
	gpuErrorCheck(cudaMemcpy(d_intermediate, bias, sizeof(double) * output_dim, cudaMemcpyHostToDevice));

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
		for(int i = 0; i < input_dim; i++){
			int w_index = row * input_dim + i;
			double contribution = weights[w_index];
			error_term[row] += contribution;
		}
	}
	__syncthreads();
	weights[index] -= learning_rate * activations[col] * od[threadIdx.x] * ig[threadIdx.x];
	if (col == 0) error_term[row] = od[threadIdx.x] * ig[threadIdx.x] * error_term[row];
	if (col == 1) bias[row] -= learning_rate * od[threadIdx.x] * ig[threadIdx.x];
	
}
void backward_wrapper(double * activations, double * actual_outputs, double * bias, Cost_Function * f, double learning_rate, int output_dim, int input_dim, double * intermediate, double * weights, double * output, Activation_Function *activation_function, bool final_layer, double * error_term){

	double *d_activations,*d_actual_outputs, *d_weights, *d_bias, *d_intermediate, *d_output, *d_error_term;

	//inputs
	cudaMalloc((void**)&d_activations, input_dim * sizeof(double));
	cudaMalloc((void**)&d_intermediate, output_dim*sizeof(double));
	cudaMalloc((void**)&d_actual_outputs, output_dim* sizeof(double));
	gpuErrorCheck(cudaMalloc((void**)&d_output, output_dim * sizeof(double)));

	//outputs
	cudaMalloc((void**)&d_weights, input_dim * output_dim * sizeof(double));
	cudaMalloc((void**)&d_bias, output_dim * sizeof(double)); 
	cudaMalloc((void**)&d_error_term, output_dim * sizeof(double));
	

	cudaMemcpy(d_actual_outputs, actual_outputs, sizeof(double) * output_dim, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, weights, sizeof(double) * input_dim * output_dim, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bias, bias, sizeof(double) * output_dim, cudaMemcpyHostToDevice);
	cudaMemcpy(d_activations, activations, input_dim * sizeof(double), cudaMemcpyHostToDevice);
	if(final_layer) cudaMemcpy(d_output, output, output_dim * sizeof(double), cudaMemcpyHostToDevice);
	gpuErrorCheck(cudaMemcpy(d_intermediate, intermediate, sizeof(double) * output_dim, cudaMemcpyHostToDevice));

	dim3 block_size(32, 32);
    	dim3 grid_size((output_dim - 1)/32 + 1, (input_dim - 1)/32 + 1);
	parallel_backward<<<grid_size, block_size>>>(d_activations, d_actual_outputs, d_bias, f, learning_rate, output_dim, input_dim, d_intermediate, d_weights, d_output, activation_function, final_layer, d_error_term);
	gpuErrorCheck(cudaGetLastError());

	cudaDeviceSynchronize();
	cudaMemcpy(weights, d_weights, input_dim * output_dim * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(bias, d_bias, output_dim * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(error_term, d_error_term, output_dim * sizeof(double), cudaMemcpyDeviceToHost);
	gpuErrorCheck(cudaGetLastError());

	// Free device memory
    	cudaFree(d_activations);
    	cudaFree(d_weights);
    	cudaFree(d_bias);
    	cudaFree(d_intermediate);
    	cudaFree(d_output);
    	cudaFree(d_actual_outputs);
    	cudaFree(d_error_term);
	gpuErrorCheck(cudaGetLastError());
}

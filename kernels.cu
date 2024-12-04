#include <cuda.h>
#include <iostream>
#include <assert.h>

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
__global__ void parallel_forward(double * input, double * bias, int output_dim, int input_dim, double * intermediate, double * weights, double * output, Activation_Function *activation_function)
{
	int row = threadIdx.x + blockDim.x * blockIdx.x;	
	int col = threadIdx.y + blockDim.y * blockIdx.y;	
	int index = row * input_dim + col;	

	if (col == 0) intermediate[row] = bias[row];
	__syncthreads();
	intermediate[row] += weights[index] * inputs[col];
	__syncthreads();
	outputs[row] = activation_function->evaluate(intermediate[row]);
}

//void Layer::backward(double* actual_outputs, double* activations, Cost_Function *f, double learning_rate, bool final_layer){
//    double output_derivatives[output_dim];
//    double intermediate_gradient[output_dim];
//    for (int row = 0; row < output_dim; row++) {
//		if (final_layer) output_derivatives[row] = f->derivative(actual_outputs[row], _outputs[row]);
//		else output_derivatives[row] = actual_outputs[row];
//
//		intermediate_gradient[row] = _activation_function->derivative(_intermediate[row]);
//		_error_term[row] = 0;
//
//		for (int col = 0; col < input_dim; col++) {
//			int index = row * input_dim + col;
//			_error_term[row] += _weights[index];
//			_weights[index] -= learning_rate * activations[col] * output_derivatives[row] * intermediate_gradient[row]; 
//		}
//
//		_error_term[row] = output_derivatives[row] * intermediate_gradient[row] * _error_term[row];
//		_bias[row] -= learning_rate * output_derivatives[row] * intermediate_gradient[row]; 
//    }
//}

__global__ void parallel_backward(double * actual_outputs, double * bias, double * output_derivatives, double * intermediate_gradient, Cost_Function * f, int learning_rate, int input_dim, double * intermediate, double * weights, double * output, Activation_Function *activation_function, bool final_layer, double * error_term)
{
	int row = threadIdx.x + blockDim.x * blockIdx.x;	
	int col = threadIdx.y + blockDim.y * blockIdx.y;	
	int index = row * input_dim + col;	

	if (col == 0){
		if(final_layer) output_derivatives[row] = f->derivative(actual_outputs[row], output[row]);
		else output_derivatives[row] = actual_outputs[row];
	}
	if (col == 1) intermediate_gradient[row] = _activation_function->derivative(_intermediate[row]);
	if (col == 2) error_term[row] = 0;
	__syncthreads();


	//This reduction here may lead to non-deterministic behavior
	error_term[row] += weights[index];

	weights[index] -= learning_rate * weights[index] * output_derivatives[row] * intermediate_gradient[row];

	if (col == 0) error_term[row] = output_derivatives[row] * intermediate_gradient[row] * error_term[row];
	if (col == 1) bias[row] -= learning_rate * output_derivatives[row] * intermediate_gradient[row];
	__syncthreads();
	
}

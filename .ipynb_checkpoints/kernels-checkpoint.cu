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

__global__ void foo(void)
{
	std::cout << "skeleton example" << std::endl;
}

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

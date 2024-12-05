/*
 *
 * Layer.cpp
 * Function Defintions for Layer.h.
 *
 */

#include <cmath>
#include <cuda.h>
#include <cstdlib>
#include "Layer.h"
#include "wrapper.h"

Layer::Layer(int input_dim, int output_dim, Activation_Function* activation_function)
	:input_dim(input_dim), output_dim(output_dim), _activation_function(activation_function) {
		cudaMalloc((void**)&_weights, sizeof(double) * input_dim * output_dim);
		cudaMalloc((void**)&_outputs, sizeof(double) * output_dim);
		cudaMalloc((void**)&_error_term, sizeof(double) * output_dim);
		cudaMalloc((void**)&_intermediate, sizeof(double) * output_dim);
        	cudaMalloc((void**)&_bias, sizeof(double) * output_dim);

		double * weights = (double*)malloc(sizeof(double) * input_dim * output_dim);
		double * outputs = (double*)malloc(sizeof(double) * output_dim);
		double * intermediate = (double*)malloc(sizeof(double) * output_dim);
		double * bias = (double*)malloc(sizeof(double) * output_dim);

		for(int i = 0; i < input_dim * output_dim; i++){
			weights[i] = (rand() % 100) / 100.0;
		}
		
		for(int i = 0; i < output_dim; i++){
			outputs[i] = 0.0;
			intermediate[i] = 0.0;
    			bias[i] = 0.0;
		}
		cudaMemcpy(_weights, weights, sizeof(double) * input_dim * output_dim, cudaMemcpyHostToDevice);
		cudaMemcpy(_intermediate, intermediate, sizeof(double) * output_dim, cudaMemcpyHostToDevice);
		cudaMemcpy(_outputs, outputs, sizeof(double) * output_dim, cudaMemcpyHostToDevice);
		cudaMemcpy(_bias, bias, sizeof(double) * output_dim, cudaMemcpyHostToDevice);
		free(intermediate);
		free(bias);
		free(outputs);
		free(weights);
}
void Layer::forward(double* inputs){
	forward_wrapper(inputs, _bias, output_dim, input_dim, _intermediate, _weights, _outputs, _activation_function);
}
void Layer::backward(double* actual_outputs, double* activations, Cost_Function *f, double learning_rate, bool final_layer){
	backward_wrapper(activations, actual_outputs, _bias, f, learning_rate, output_dim, input_dim, _intermediate, _weights, _outputs, _activation_function, final_layer, _error_term);
}
void Layer::info(){
    double * weights = (double*)malloc(sizeof(double) * input_dim * output_dim);
    double * outputs = (double*)malloc(sizeof(double) * output_dim);
    double * bias = (double*)malloc(sizeof(double) * output_dim);
    cudaMemcpy(weights, _weights, sizeof(double) * output_dim * input_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(outputs, _outputs, sizeof(double) * output_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(bias, _bias, sizeof(double) * output_dim, cudaMemcpyDeviceToHost);
    std::cout << "-------" << std::endl;
	std::cout << "Weights" << std::endl;
	for(int row = 0; row < output_dim; row++){
		for(int col = 0; col < input_dim; col++){
			int index = row * input_dim + col;
			std::cout << weights[index] << "\t";
		}
		std::cout << std::endl;
	}
    
    	std::cout << "Bias" << std::endl;
	for(int i = 0; i < output_dim; i++){
		std::cout << bias[i] << std::endl;
	}
    
	std::cout << "Outputs" << std::endl;
	for(int i = 0; i < output_dim; i++){
		std::cout << outputs[i] << std::endl;
	}
	free(weights);
	free(outputs);
	free(bias);
}

double *Layer::getOutput() {
    double * outputs = (double *) malloc(output_dim * sizeof(double));
    cudaMemcpy(outputs, _outputs, sizeof(double) * output_dim, cudaMemcpyDeviceToHost);
    return outputs ;
}
double *Layer::getKernelOutput(){
	return _outputs;
}
double *Layer::get_error_term() {
    return _error_term;
}

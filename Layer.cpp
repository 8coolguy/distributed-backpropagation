/*
 *
 * Layer.cpp
 * Function Defintions for Layer.h.
 *
 */

#include <cmath>
#include <cstdlib>
#include "Layer.h"
#include "wrapper.h"

Layer::Layer(int input_dim, int output_dim, Activation_Function* activation_function)
	:input_dim(input_dim), output_dim(output_dim), _activation_function(activation_function) {
		_weights = (double*)malloc(sizeof(double) * input_dim * output_dim);
		_outputs = (double*)malloc(sizeof(double) * output_dim);
		_error_term = (double*)malloc(sizeof(double) * output_dim);
		_intermediate = (double*)malloc(sizeof(double) * output_dim);
        	_bias = (double*)malloc(sizeof(double) * output_dim);
        
		for(int i = 0; i < input_dim * output_dim; i++){
			_weights[i] = (rand() % 100) / 100.0;
		}
        
		for(int i = 0; i < output_dim; i++){
			_outputs[i] = 0.0;
			_intermediate[i] = 0.0;
    		_bias[i] = 0.0;
        }
}
void Layer::forward(double* inputs){
	forward_wrapper(inputs, _bias, output_dim, input_dim, _intermediate, _weights, _outputs, _activation_function);
}
void Layer::backward(double* actual_outputs, double* activations, Cost_Function *f, double learning_rate, bool final_layer){
    double output_derivatives[output_dim];
    double intermediate_gradient[output_dim];
    for (int row = 0; row < output_dim; row++) {
	if (final_layer) output_derivatives[row] = f->derivative(actual_outputs[row], _outputs[row]);
	else output_derivatives[row] = actual_outputs[row];
	intermediate_gradient[row] = _activation_function->derivative(_intermediate[row]);
	_error_term[row] = 0;
	for (int col = 0; col < input_dim; col++) {
	    int index = row * input_dim + col;
	    _error_term[row] += _weights[index];
	    _weights[index] -= learning_rate * activations[col] * output_derivatives[row] * intermediate_gradient[row]; 
	}
	_error_term[row] = output_derivatives[row] * intermediate_gradient[row] * _error_term[row];
	_bias[row] -= learning_rate * output_derivatives[row]; 
    }
}
void Layer::info(){
    std::cout << "-------" << std::endl;
	std::cout << "Weights" << std::endl;
	for(int row = 0; row < output_dim; row++){
		for(int col = 0; col < input_dim; col++){
			int index = row * input_dim + col;
			std::cout << _weights[index] << "\t";
		}
		std::cout << std::endl;
	}
    
    std::cout << "Bias" << std::endl;
	for(int i = 0; i < output_dim; i++){
		std::cout << _bias[i] << std::endl;
	}
    
	std::cout << "Outputs" << std::endl;
	for(int i = 0; i < output_dim; i++){
		std::cout << _outputs[i] << std::endl;
	}
}

double *Layer::getOutput() {
    return _outputs;
}
double *Layer::get_error_term() {
    return _error_term;
}

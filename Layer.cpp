/*
 *
 * Layer.cpp
 * Function Defintions for Layer.h.
 *
 */

#include <cmath>
#include "Layer.h"

Layer::Layer(int input_dim, int output_dim, double initial_weight, Activation_Function* activation_function, double initial_bias)
	:input_dim(input_dim), output_dim(output_dim), _activation_function(activation_function), _bias(initial_bias) {
		_weights = (double*)malloc(sizeof(double) * input_dim * output_dim);
		_outputs = (double*)malloc(sizeof(double) * output_dim);
		_intermediate = (double*)malloc(sizeof(double) * output_dim);
		for(int i = 0; i < input_dim * output_dim; i++){
			_weights[i] = initial_weight;
		}
		for(int i = 0; i < output_dim; i++){
			_outputs[i] = 0.0;
			_intermediate[i] = 0.0;

		}

}
void Layer::forward(double* inputs){
	for(int row = 0; row < output_dim; row++){
		for(int col = 0; col < input_dim; col++){
			int index = row * input_dim + col;
			_intermediate[row] += _weights[index] * inputs[col];
		}
		_outputs[row] = _activation_function->evaluate(_intermediate[row]);
	}
}
void Layer::backward(double* actual_outputs, Cost_Function *f, double learning_rate){
	for(int i = 0; i < output_dim; i++){
		actual_outputs[i] = f->derivative(actual_outputs[i],_outputs[i]);
		_intermediate[i] = _activation_function->derivative(_intermediate[i]);
	}
	for(int row = 0; row < output_dim; row++){
		for(int col = 0; col < input_dim; col++){
			int index = row * input_dim + col;
			_weights[index] += learning_rate * _weights[index] * actual_outputs[row] * _intermediate[row];
		}
	}
}

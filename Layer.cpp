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
		for(int i = 0; i < input_dim * output_dim; i++){
			_weights[i] = initial_weight;
		}
		for(int i = 0; i < output_dim; i++){
			_outputs[i] = 0.0;
		}

}
void Layer::forward(double* inputs){
	for(int row = 0; row < output_dim; row++){
		for(int col = 0; col < input_dim; col++){
			int index = row * input_dim + col;
			_outputs[row] += _weights[index] * inputs[col];
		}
		_outputs[row] = _activation_function->evaluate(_outputs[row]);
	}
}
void Layer::backward(double* outputs){

}

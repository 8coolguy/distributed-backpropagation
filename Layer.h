/*
 *
 * Layer.h 
 * Header file for the layer class. The layer class will store a matrix of weights to calculate the
 * outputs. The layer can take an acitvation function used for the forward and backward steps. Will
 * be able to set the bias and also initalize the weights.
 *
 */ 
#include "Activation_Function.h"
#include "Cost_Function.h"

# ifndef LAYER_H
# define LAYER_H
class Layer{
private:
	double *_weights;
	double *_intermediate;
	double *_outputs;
	Activation_Function* _activation_function;
	double *_bias;
public:
	int input_dim;
	int output_dim;
	double *outputs;
	Layer(int input_dim, int output_dim, Activation_Function *activation_function);
	void forward(double* inputs);
	void backward(double* outputs, Cost_Function *f, double learning_rate);
	void info();
    double *getOutput();
};
# endif

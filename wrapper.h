/*
 *
 * Wrapper.h 
 * File to wrap the Cuda kernels.
 *
 */ 

#ifndef WRAPPER_H 
#define WRAPPER_H 

void forward_wrapper(double * input, double * bias, int output_dim, int input_dim, double * intermediate, double * weights, double * output, Activation_Function *activation_function);
void backward_wrapper(double * activations, double * actual_outputs, double * bias, Cost_Function * f, double learning_rate, int output_dim, int input_dim, double * intermediate, double * weights, double * output, Activation_Function *activation_function, bool final_layer, double * error_term);
#endif

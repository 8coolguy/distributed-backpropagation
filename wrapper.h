/*
 *
 * Wrapper.h 
 * File to wrap the Cuda kernels.
 *
 */ 

#ifndef WRAPPER_H 
#define WRAPPER_H 

void forward_wrapper(double * input, double * bias, int output_dim, int input_dim, double * intermediate, double * weights, double * output, Activation_Function *activation_function);

#endif

/*
 *
 * Activation_Function.h 
 * Header file for the activation function class. Can create new functions by inheirting the class
 * and defining the evalutate and the derivative function. The sigmoid funciton can be used as an example.
 *
 */ 

#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <cmath>
#include <iostream>

class Activation_Function{
public:
	virtual double evaluate(double x) = 0;

	virtual double derivative(double x) = 0;
};
class Sigmoid:Activation_Function {
public:
	double evaluate(double x){
		double exp = std::exp(-x) + 1;
		return 1/exp;

	}
	double derivative(double x){
		double sigmoid = evaluate(x);
		return sigmoid * (1 - sigmoid); 
	}
};

#endif

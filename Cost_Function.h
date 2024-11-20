/*
 *
 * Cost_Function.h 
 * Header file for the cost function class. Can create new functions by inheirting the class
 * and defining the evalutate and the derivative function. The squared error funciton can be used as an example.
 *
 */ 

#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include <cmath>
#include <iostream>

class Cost_Function{
public:
	virtual double evaluate(double actual, double predicted) = 0;

	virtual double derivative(double actual, double predicted) = 0;
};
class SE: public Cost_Function {
public:
	double evaluate(double actual, double predicted){
		double error = actual - predicted;
		return error * error;
	}
	double derivative(double actual, double predicted){
		return 2 * (predicted - actual);
	}
};

#endif

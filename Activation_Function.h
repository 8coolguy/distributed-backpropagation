/*
 *
 * Activation_Function.h 
 * Header file for the activation function class. Includes the implementation of ReLU.
 *
 */ 

#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <cmath>
#include <iostream>

class Activation_Function {
public:
    virtual double evaluate(double x) = 0;

    virtual double derivative(double x) = 0;
};

class Sigmoid: public Activation_Function {
public:
    double evaluate(double x) {
        double exp = std::exp(-x) + 1;
        return 1 / exp;
    }

    double derivative(double x) {
        double sigmoid = evaluate(x);
        return sigmoid * (1 - sigmoid); 
    }
};

class ReLU: public Activation_Function {
public:
    double evaluate(double x) override {
        return x > 0 ? x : 0.0;
    }

    double derivative(double x) override {
        return x > 0 ? 1.0 : 0.0;
    }
};

#endif

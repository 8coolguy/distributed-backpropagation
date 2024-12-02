/*
 *
 * main.cpp
 * Main used for testing and measuring performance.
 *
 */ 

#include "Layer.h"
#include <iostream>
#include "Activation_Function.h"

using namespace std;

int main(void){
	Sigmoid f;
	SE cost;

    // Test sigmoid evaluation
    cout << "Test sigmoid evaluation: ";
	cout << f.evaluate(0.25) << endl;

    // Initialize Layer
    Layer a(1, 1, &f);
	double in[1] = { 0.25 };
    double out[1] = { 0.625 };
	
    // Run forward and backward operations
    double learning_rate = 0.1;
    cout << "IN: " << in[0] << ", OUT: " << out[0] << endl;
    
    for (int iteration = 1; iteration <= 50; iteration++) {
        a.forward(in);
    	a.backward(out, &cost, learning_rate);
    	a.info();

        double *eval = a.getOutput();
        cout << "LOSS: " << cost.evaluate(out[0], eval[0]) << endl;
    }
    
    // Final
    a.forward(in);
	a.info();
}


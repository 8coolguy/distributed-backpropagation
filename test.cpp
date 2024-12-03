/*
 *
 * main.cpp
 * Main used for testing and measuring performance.
 *
 */ 

#include "Layer.h"
#include <iostream>
#include <chrono>
#include "Activation_Function.h"

using namespace std;

int main(void){
	Sigmoid f;
	SE cost;
	clock_t start, end;

    // Test sigmoid evaluation
    cout << "Test sigmoid evaluation: ";
	cout << f.evaluate(0.25) << endl;

    // Initialize Layer
    Layer a(1, 1, &f, 8);
	double in[1] = { 0.25 };
    double out[1] = { 0.625 };
	
    // Run forward and backward operations
    double learning_rate = 0.1;
    cout << "IN: " << in[0] << ", OUT: " << out[0] << endl;
    
    for (int iteration = 1; iteration <= 50; iteration++) {
        std::cout << "-------" << std::endl;
        std::cout << "Iteration " << iteration << std::endl;
        a.forward(in);
	auto t1 = std::chrono::high_resolution_clock::now();
    	a.backward(out, in, &cost, learning_rate, true);
	auto t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = t2 - t1;
	std::cout << "Time for Backprop: " << duration.count() << "s" << std::endl;
    	a.info();

        double *eval = a.getOutput();
        cout << "LOSS: " << cost.evaluate(out[0], eval[0]) << endl;
    }
    
    // Final
    a.forward(in);
	a.info();
}


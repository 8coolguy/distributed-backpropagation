/*
 *
 * main.cpp
 * Main used for testing and measuring performance with multiple inputs and outputs.
 *
 */

#include "Layer.h"
#include <iostream>
#include "Activation_Function.h"

using namespace std;

int main(void) {
    Sigmoid f;
    SE cost;

    int input_dim = 3;
    int output_dim = 2;
    Layer a(input_dim, output_dim, &f, 8);

    double in[3] = { 0.25, 0.5, 0.75 };
    double out[2] = { 0.6, 0.4 };

    cout << "Initial Inputs: ";
    for (int i = 0; i < input_dim; i++) {
        cout << in[i] << " ";
    }
    cout << endl;

    cout << "Expected Outputs: ";
    for (int i = 0; i < output_dim; i++) {
        cout << out[i] << " ";
    }
    cout << endl;

    double learning_rate = 0.1;
    int iterations = 50;

    for (int iteration = 1; iteration <= iterations; iteration++) {
        std::cout << "-------" << std::endl;
        std::cout << "Iteration " << iteration << std::endl;
        
        a.forward(in);
        
        double* eval = a.getOutput();
        double loss = 0.0;
        for (int i = 0; i < output_dim; i++) {
            loss += cost.evaluate(out[i], eval[i]);
        }

        loss /= output_dim;
        cout << "LOSS: " << loss << endl;

        a.backward(out, in, &cost, learning_rate, true);
        a.info();
    }

    cout << "\nFinal Outputs after training: ";
    a.forward(in);
    double* final_eval = a.getOutput();
    for (int i = 0; i < output_dim; i++) {
        cout << final_eval[i] << " ";
    }
    cout << endl;

    return 0;
}

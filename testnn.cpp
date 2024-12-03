#include <iostream>
#include "Neural_Network.h"
#include "Activation_Function.h"

int main() {
    Sigmoid activation_function;
    SE cost_function;

    double learning_rate = 0.1;
    NeuralNetwork nn(learning_rate);
    
    nn.addLayer(new Layer(3, 5, &activation_function));
    nn.addLayer(new Layer(5, 2, &activation_function));

    int inputs = 3;
    int outputs = 2;
    
    double input[inputs] = {0.25, 0.5, 0.75};
    double expected_output[outputs] = {0.6, 0.4};

    for (int epoch = 1; epoch <= 50; ++epoch) {
        std::cout << "Epoch " << epoch << ":" << std::endl;

        nn.forward(input);

        double* output = nn.getOutput();
        double loss = 0.0;
        for (int i = 0; i < outputs; ++i) {
            loss += cost_function.evaluate(expected_output[i], output[i]);
        }
        loss /= outputs;
        std::cout << "Loss: " << loss << std::endl;

        nn.backward(input, expected_output, &cost_function);
        nn.info();
    }

    return 0;
}

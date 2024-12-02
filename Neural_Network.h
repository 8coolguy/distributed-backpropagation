#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include "Layer.h"
#include "Cost_Function.h"

class NeuralNetwork {
private:
    std::vector<Layer*> layers;
    double learning_rate;

public:
    NeuralNetwork(double lr);
    void addLayer(Layer* layer);
    void forward(double* input);
    void backward(double* actual_output, Cost_Function* cost_function);
    double* getOutput();
    void info();
};

#endif

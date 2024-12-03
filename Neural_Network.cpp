#include "Neural_Network.h"
#include <iostream>

NeuralNetwork::NeuralNetwork(double lr) : learning_rate(lr) {}

void NeuralNetwork::addLayer(Layer* layer) {
    layers.push_back(layer);
}

void NeuralNetwork::forward(double* input) {
    double* current_input = input;
    for (auto& layer : layers) {
        layer->forward(current_input);
        current_input = layer->getOutput();
    }
}

void NeuralNetwork::backward(double* input, double* actual_output, Cost_Function* cost_function) {
    double* current_gradients = actual_output;
    double *activations;

    for (int i = layers.size() - 1; i >= 0; i--) {
	if (i == 0) activations = input;
	else activations = layers[i - 1]->getOutput();
        if (i == layers.size() - 1) layers[i]->backward(current_gradients, activations, cost_function, learning_rate, true);
	else layers[i]->backward(current_gradients, activations, cost_function, learning_rate, false );
        current_gradients = layers[i]->get_error_term();
    }
}

double* NeuralNetwork::getOutput() {
    return layers.back()->getOutput();
}

void NeuralNetwork::info() {
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << "Layer " << i + 1 << ":" << std::endl;
        layers[i]->info();
    }
}

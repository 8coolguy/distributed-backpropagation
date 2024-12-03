/*
 *
 * main.cpp
 * Main used for testing and measuring performance.
 *
 */ 

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

#include "Neural_Network.h"
#include "Activation_Function.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }

    string fileName = argv[1];
    ifstream inputFile(fileName);

    if (!inputFile.is_open()) {
        cerr << "Error: Could not open file " << fileName << " for reading." << endl;
        return 1;
    }

    int numInputs, numOutputs;
    inputFile >> numInputs >> numOutputs;
    cout << "Number of inputs: " << numInputs << ", Number of outputs: " << numOutputs << endl;

    vector<double*> inputs; 
    vector<double*> outputs; 

    double value;
    while (inputFile) {
       double* inputRow = new double[numInputs];
        double* outputRow = new double[numOutputs];

        for (int i = 0; i < numInputs; ++i) {
            if (!(inputFile >> value)) break;
            inputRow[i] = value;
        }

        for (int i = 0; i < numOutputs; ++i) {
            if (!(inputFile >> value)) break;
            outputRow[i] = value;
        }

        if (inputFile) {
            inputs.push_back(inputRow);
            outputs.push_back(outputRow);
        }
    }

    inputFile.close();
    cout << "Finished processing file: " << fileName << endl;

    // Initialize the neural network
    Sigmoid sigmoidActivation;
    ReLU reluActivation; 
    SE costFunction;
    NeuralNetwork nn(0.1);

    // Add layers to the network
    nn.addLayer(new Layer(numInputs, 5, &reluActivation));
    nn.addLayer(new Layer(5, numOutputs, &sigmoidActivation));

    // Train the network
    int epochs = 50;
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double totalLoss = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            nn.forward(inputs[i]);

            double* predicted = nn.getOutput();
            for (int j = 0; j < numOutputs; ++j) {
                totalLoss += costFunction.evaluate(outputs[i][j], predicted[j]);
            }

            nn.backward(inputs[i], outputs[i], &costFunction);
        }

        cout << "Epoch " << epoch << ", Loss: " << totalLoss / inputs.size() << endl;
    }

    // Test the network
    cout << "Final outputs after training:" << endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        nn.forward(inputs[i]);
        double* predicted = nn.getOutput();

        cout << "Input: ";
        for (int j = 0; j < numInputs; ++j) {
            cout << inputs[i][j] << " ";
        }

        cout << "Predicted: ";
        for (int j = 0; j < numOutputs; ++j) {
            cout << predicted[j] << " ";
        }

        cout << "Expected: ";
        for (int j = 0; j < numOutputs; ++j) {
            cout << outputs[i][j] << " ";
        }

        cout << endl;
    }

    return 0;
}

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
#include <chrono>

#include "Neural_Network.h"
#include "Activation_Function.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 5) {
        cout << "Usage: " << argv[0] << " <input_file> <num_layers> <num_nodes_per_layer> <num_threads>" << endl;
        return 1;
    }

    string fileName = argv[1];
    int numLayers = atoi(argv[2]);
    int numNodesPerLayer = atoi(argv[3]);
    int num_threads = atoi(argv[4]);

    if (numLayers < 3) {
        cout << "Note: Number of layers will be at least 3 due to implementation. Extra hidden layers will be added if greater than 3." << endl;
    }
    
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
    Sigmoid activationFunction; 
    SE costFunction;
    NeuralNetwork nn(0.1);

    // Add layers to the network
    nn.addLayer(new Layer(numInputs, numNodesPerLayer, &activationFunction, num_threads));

    for (int i = 0; i < numLayers - 3; ++i) {
        nn.addLayer(new Layer(numNodesPerLayer, numNodesPerLayer, &activationFunction, num_threads));
    }
    
    nn.addLayer(new Layer(numNodesPerLayer, numOutputs, &activationFunction, num_threads));

    // Train the network and log content
    ofstream logStream("log.txt");

    logStream << argv[0] << " " << argv[1] << " " << argv[2] << " " << argv[3] << endl;
    logStream << "Using NN with " << numLayers << " layers (including input and output layers) and " << numNodesPerLayer << " nodes per layer." << endl;
    logStream << "Number of NN inputs: " << numInputs << ", Number of NN outputs: " << numOutputs << endl;
    logStream << "Number of Data Points: " << inputs.size() << endl;
    
    int epochs = 25;
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        // Metrics
        double totalLoss = 0.0;
        double totalForward = 0.0;
        double totalBackward = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            auto start = chrono::high_resolution_clock::now();
            nn.forward(inputs[i]);
            auto end = chrono::high_resolution_clock::now();

            chrono::duration<double> elapsed = end - start;
            totalForward += elapsed.count();
            // logStream << "Epoch " << epoch << ", i=" << i << ", Forward Pass Time: " << elapsed.count() << " seconds" << endl;

            double* predicted = nn.getOutput();
            for (int j = 0; j < numOutputs; ++j) {
                totalLoss += costFunction.evaluate(outputs[i][j], predicted[j]);
            }

            start = chrono::high_resolution_clock::now();
            nn.backward(inputs[i], outputs[i], &costFunction);
            end = chrono::high_resolution_clock::now();

            elapsed = end - start;
            totalBackward += elapsed.count();
            // logStream << "Epoch " << epoch << ", i=" << i << ", Backpropagation Time: " << elapsed.count() << " seconds" << endl;
        }
        logStream << "Epoch " << epoch << ", Loss: " << totalLoss / inputs.size() << endl;
        logStream << "Epoch " << epoch << ", Total Forward Pass Time: " << totalForward << " seconds" << endl;
        logStream << "Epoch " << epoch << ", Total Backpropagation Time: " << totalBackward << " seconds" << endl;
        cout << "Epoch " << epoch << ", Loss: " << totalLoss / inputs.size() << endl;
    }

    // Test the network
    // cout << "Final outputs after training:" << endl;
    // for (size_t i = 0; i < inputs.size(); ++i) {
    //     nn.forward(inputs[i]);
    //     double* predicted = nn.getOutput();

    //     cout << "Input: ";
    //     for (int j = 0; j < numInputs; ++j) {
    //         cout << inputs[i][j] << " ";
    //     }

    //     cout << "Predicted: ";
    //     for (int j = 0; j < numOutputs; ++j) {
    //         cout << predicted[j] << " ";
    //     }

    //     cout << "Expected: ";
    //     for (int j = 0; j < numOutputs; ++j) {
    //         cout << outputs[i][j] << " ";
    //     }

    //     cout << endl;
    // }

    return 0;
}

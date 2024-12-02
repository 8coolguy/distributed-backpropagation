#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>

using namespace std;

double calculateEquation(double x) {
    return x * 0.5 + 0.3;
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        cout << "Usage: " << argv[0] << " <count> <output_file> <num_inputs> <num_outputs>" << endl;
        return 1;
    }

    int count = atoi(argv[1]);
    string fileName = argv[2];
    int numInputs = atoi(argv[3]);
    int numOutputs = atoi(argv[4]);

    ofstream outputFile(fileName);
    if (!outputFile.is_open()) {
        cerr << "Error: Could not open file " << fileName << " for writing." << endl;
        return 1;
    }

    srand(time(0));

    outputFile << numInputs << " " << numOutputs << endl;

    // Generate data points
    for (int i = 0; i < count; ++i) {
        double inputs[numInputs];

        for (int j = 0; j < numInputs; ++j) {
            inputs[j] = (rand() % 100) / 100.0;
            outputFile << inputs[j] << " ";
        }

        for (int j = 0; j < numOutputs; ++j) {
            double x = (j < numInputs) ? inputs[j] : inputs[0];
            double y = calculateEquation(x);
            outputFile << y << (j < numOutputs - 1 ? " " : "");
        }

        outputFile << endl;
    }

    outputFile.close();
    cout << "Data successfully written to " << fileName << endl;

    return 0;
}

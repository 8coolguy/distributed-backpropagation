/*
 *
 * main.cpp
 * Main used for testing and measuring performance.
 *
 */ 

#include <iostream>
#include <fstream>

#include "Activation_Function.h"

using namespace std;

int main(int argc, char* argv[]){
	if (argc < 2) {
        cout << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }

    string fileName = argv[1];
	ifstream inputFile(fileName);

    if (!inputFile.is_open()) {
        cerr << "Error: Could not open file " << fileName << " for writing." << endl;
        return 1;
    }

    double x, y;
    while (inputFile >> x >> y) {
        // Perform processing each input and output (Or store into a vector later)
        cout << "(" << x << ", " << y << ")" << endl;
    }

    inputFile.close();
    cout << "Finished processing file: " << fileName << endl;

	return 0;
}

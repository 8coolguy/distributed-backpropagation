#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>

using namespace std;

double calculateEquation(double x) {
    return x * 5.0 + 5.0;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <count> <output_file>" << endl;
        return 1;
    }

    int count = atoi(argv[1]);
    string fileName = argv[2];

    ofstream outputFile(fileName);
    if (!outputFile.is_open()) {
        cerr << "Error: Could not open file " << fileName << " for writing." << endl;
        return 1;
    }

    srand(time(0));

    for (int i = 0; i < count; ++i) {
        double x = (rand() % 100) / 100.0;
        outputFile << x << " " << calculateEquation(x) << endl;
    }

    outputFile.close();
    cout << "Data successfully written to " << fileName << endl;

    return 0;
}

# Distributed Backpropagation

## Steps   
1. Create serial backpropagtion.

2. Distrubute onto many nodes in the WAVE center.

3. Use CUDA to improve calculation on each node.   

## Activation Functions   
Made a class to create activation functions. They have to inherir the Activation Function class and you have to define evalute and derivative.


## Serial Back Propagation   
Created a Layer class that will store the weights from layer i of size n to layer of j size m. Will do more work to implment the how inital weights are created.

## Usage
1. Run "make"

2. Run "./generate <count> <file_name> <num_inputs> <num_outputs>" to create a sample dataset.

3. Run "./main <file_name>" to train a neural network on the sample dataset or a given input file.

The Makefile also compiles test programs which can be run "./test", "./test2", and "./testnn" to test the layer program and neural network program.

The input file structure starts with the number of inputs and outputs and then the data points, ex:
```
2 1
0.68 0.41 0.64
0.05 0.37 0.325
...
0.9 0.04 0.75
```
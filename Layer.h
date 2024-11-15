/*
 *
 * Layer.h 
 * Header file for the layer class. The layer class will store a matrix of weights to calculate the
 * outputs. The layer can take an acitvation function used for the forward and backward steps. Will
 * be able to set the bias and also initalize the weights.
 *
 */ 

# infdef LAYER_H
# define LAYER_H
class Layer{
private:
	double *_weights;
	double *_outputs;
	Activation_Function _activation_function;
	double _bias;
public:
	int input_dim;
	int output_dim;
	Layer(int input_dim, int output_dim, double intial_weight, Activation_Function activation_function, double intial_bias);
	void forward(double&inputs);
	void backward(double& outputs);
}
# endif

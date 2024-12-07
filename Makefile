#Will have to change this to nvcc when using CUDA
CXX 		= nvcc
CXXFLAGS	= -g -Wall
MAIN        = main
EXPERIMENT  = experiment
PROG_NN     = testnn
PROG_GEN	= generate

all:
	$(CXX) testnn.cu -o $(PROG_NN) Layer.cu Neural_Network.cpp kernels.cu
	$(CXX) generate_data.cpp -o $(PROG_GEN)
	$(CXX) experimentMain.cu -o $(EXPERIMENT) Layer.cu Neural_Network.cpp kernels.cu
clean:
	$(RM) $(PROG_NN) $(PROG_GEN) $(EXPERIMENT)

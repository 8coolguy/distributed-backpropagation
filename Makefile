#Will have to change this to nvcc when using CUDA
CXX 		= nvcc
CXXFLAGS	= -g -Wall
MAIN        = main
EXPERIMENT  = experiment
PROG		= test
PROG_2      = test2
PROG_NN     = testnn
PROG_GEN	= generate

all:
	$(CXX) main.cpp -o $(MAIN) Layer.cpp Neural_Network.cpp kernels.cu
	$(CXX) test.cpp -o $(PROG) Layer.cpp kernels.cu
	$(CXX) test2.cpp -o $(PROG_2) Layer.cpp kernels.cu
	$(CXX) testnn.cpp -o $(PROG_NN) Layer.cpp Neural_Network.cpp kernels.cu
	$(CXX) generate_data.cpp -o $(PROG_GEN)
	$(CXX) experimentMain.cpp -o $(EXPERIMENT) Layer.cpp Neural_Network.cpp kernels.cu
clean:
	$(RM) $(MAIN) $(PROG) $(PROG_2) $(PROG_NN) $(PROG_GEN) $(EXPERIMENT)

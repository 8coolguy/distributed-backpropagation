#Will have to change this to nvcc when using CUDA
CXX 		= g++
CXXFLAGS	= -g -Wall
MAIN        = main
PROG		= test
PROG_2      = test2
PROG_NN     = testnn
PROG_GEN	= generate

all:
	$(CXX) main.cpp -o $(MAIN) Layer.cpp Neural_Network.cpp -fopenmp
	$(CXX) test.cpp -o $(PROG) Layer.cpp -fopenmp
	$(CXX) test2.cpp -o $(PROG_2) Layer.cpp -fopenmp
	$(CXX) testnn.cpp -o $(PROG_NN) Layer.cpp Neural_Network.cpp -fopenmp
	$(CXX) generate_data.cpp -o $(PROG_GEN)
clean:
	$(RM) $(MAIN) $(PROG) $(PROG_2) $(PROG_NN) $(PROG_GEN)

#Will have to change this to nvcc when using CUDA
CXX 		= g++
CXXFLAGS	= -g -Wall
PROG		= test
PROG_2      = test2
PROG_NN     = testnn
PROG_GEN	= generate

all:
	$(CXX) test.cpp -o $(PROG) Layer.cpp
	$(CXX) test2.cpp -o $(PROG_2) Layer.cpp
	$(CXX) testnn.cpp -o $(PROG_NN) Layer.cpp Neural_Network.cpp
	$(CXX) generate_data.cpp -o $(PROG_GEN)
clean:
	$(RM) $(PROG) $(PROG_2) $(PROG_NN) $(PROG_GEN)

#Will have to change this to nvcc when using CUDA
CXX 		= g++
CXXFLAGS	= -g -Wall
PROG		= main
PROG_GEN	= generate

all:
	$(CXX) main.cpp -o $(PROG)
	$(CXX) generate_data.cpp -o $(PROG_GEN)
clean:
	$(RM) $(PROG) $(PROG_GEN)

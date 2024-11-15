#Will have to change this to nvcc when using CUDA
CXX 		= g++
CXXFLAGS	= -g -Wall
PROG		= main

all:
	$(CXX) main.cpp -o $(PROG)
clean:
	$(RM) $(PROG)

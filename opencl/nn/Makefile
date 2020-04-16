include ../../common/make.config

#C compiler
CC = g++

SRC = nearestNeighbor.cpp clutils.cpp utils.cpp

CC_FLAGS = -g -O3 -lm -Wall

EXE = nn

LIBS = -lOpenCL -fopenmp -pthread

release:
	$(CC) -std=c++11 $(CC_FLAGS) $(SRC) -o $(EXE) $(LIBS) -I$(OPENCL_INC) -L$(OPENCL_LIB)
	
clean: 
	rm -f $(EXE) 

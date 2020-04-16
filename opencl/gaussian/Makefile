include ../../common/make.config

#C compiler
CC = g++

SRC = clutils.cpp gaussianElim.cpp utils.cpp

CC_FLAGS = -g -O3

EXE = gaussian

release:
	$(CC) -std=c++11 $(KERNEL_DIM) $(CC_FLAGS) -lOpenCL -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -I$(PROFILER_LOCATION) -L$(OPENCL_LIB) -L$(PROFILER_LOCATION) -lAMDProf -pthread
	
clean: 
	rm -f $(EXE) 

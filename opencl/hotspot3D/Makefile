include ../../common/make.config

CC = g++

OUTPUT = *.out
CCFLAGS= -g -O3 -lm -Wall

LIBS = -lOpenCL -fopenmp -pthread
EXE = 3D
FILES = 3D.cpp CL_helper.cpp CL_helper.h
# OPENCL_DIR = /if10/kw5na/Packages/AMD-APP-SDK-v2.8-RC-lnx64
# OPENCL_INC = $(OPENCL_DIR)/include/ 
# OPENCL_LIB = $(OPENCL_DIR)/lib/x86_64/ -lOpenCL

3D: 
	$(CC) -std=c++11 $(CCFLAGS) -o $(EXE) $(FILES) -I$(OPENCL_INC) -L$(OPENCL_LIB) $(LIBS) -mfpmath=sse

profile:
	$(CC) -std=c++11 $(CCFLAGS) -o $(EXE) $(FILES) -I$(OPENCL_INC) -L$(OPENCL_LIB) $(LIBS) -I$(PROFILER_LOCATION) -lAMDProf -DPROFILER -mfpmath=sse

clean:
	rm -f 3D $(OUTPUT) 

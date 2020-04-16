include ../../common/make.config

# OpenCL installation path      
# OPENCL_DIR = /home/mwb7w/NVIDIA_GPU_Computing_SDK/
# OPENCL_DIR = /home/mwb7w/ati-stream-sdk-v2.2-lnx64/


CC = g++
CC_FLAGS = -g -O3 -Wall 

FILES = hotspot.cpp hotspot.h OpenCL_helper_library.c OpenCL_helper_library.h
EXE = hotspot
OPTIONS = -pthread -lOpenCL -fopenmp
# OPTIONS = -pthread -lOpenCL -fopenmp -lAMDProf 

hotspot: 
	$(CC) -std=c++11 $(KERNEL_DIM) $(CC_FLAGS) -o $(EXE) $(FILES) -I$(OPENCL_INC) -I$(PROFILER_LOCATION) -L$(OPENCL_LIB) $(OPTIONS)

clean:
	rm -f $(EXE)

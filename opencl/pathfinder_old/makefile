include ../../common/make.config
CXX = g++
DEBUGFLAGS = -g -O3
# You will need to modify the following 2 lines
# by inserting the correct path to the OpenCL
# lib and inc directories.
CXXFLAGS = -I$(OPENCL_INC) -I$(PROFILER_LOCATION) -lAMDProf -pthread -lOpenCL  -fopenmp 
LINKFLAGS = -L$(OPENCL_LIB) -L$(PROFILER_LOCATION)
SRCS = main.cpp OpenCL.cpp OpenCL.h

# link objects(binaries) together
# a.out: main.o OpenCL.o
# 	$(CXX) -std=c++11 main.o OpenCL.o $(LINKFLAGS) -o pathfinder

# compile main function file into object (binary)
# main.o: main.cpp OpenCL.cpp kernels.cl
# 	$(CXX) -std=c++11 $(CXXFLAGS) main.cpp OpenCL.cpp

release:
	$(CXX) -std=c++11 $(DEBUGFLAGS) $(SRCS) -o pathfinder $(CXXFLAGS) $(LINKFLAGS)

clean:
	clear
	@rm -f pathfinder *.o *.out *~ *\#

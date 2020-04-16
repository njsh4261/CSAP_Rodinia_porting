# include ../../common/make.config

#C compiler
CC = g++
CC_FLAGS = -g -O3
SRC = bfs.cpp timer.cc
EXE = bfs
PROFILER_LOCATION = /home/jihyo/CoRunBench/tools/memprof/
INCLUDE += -I$(PROFILER_LOCATION)
EXE = bfs
CFILES = bfs.cpp timer.cc
LIB = -lOpenCL -lm -lnuma -fopenmp

release:$(SRC)
	$(CC) -std=c++11 $(CC_FLAGS) -pthread $(INCLUDE) -o $(EXE) $(SRC) $(LIB)

errmsg:$(SRC)
	$(CC) -std=c++11 $(CC_FLAGS) -pthread $(INCLUDE) -o $(EXE) $(SRC) $(LIB) -D ERRMSG

profile:$(SRC)
	$(CC) -std=c++11 $(CC_FLAGS) -pthread $(INCLUDE) -o $(EXE) $(SRC) $(LIB) -D PROFILING

res:$(SRC)
	$(CC) -std=c++11 $(CC_FLAGS) -pthread $(INCLUDE) -o $(EXE) $(SRC) $(LIB) -D RES_MSG

debug: $(SRC)
	$(CC) -std=c++11 $(CC_FLAGS) -pthread $(INCLUDE) -o $(EXE) $(SRC) -lOpenCL -lm -lnuma

clean:
	rm -f $(EXECUTABLE) $(EXE).linkinfo *.so profiling.txt result*

# errmsg:$(SRC)
# 	$(CC) $(CC_FLAGS)  -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -D ERRMSG PTX_MSG

# ptx:$(SRC)
# 	$(CC) $(CC_FLAGS) -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -D PTX_MSG

# profile:$(SRC)
# 	$(CC) $(CC_FLAGS) -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -D PROFILING

# res:$(SRC)
# 	$(CC) $(CC_FLAGS) -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -D RES_MSG

# debug: $(SRC)
# 	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL 

# run:
# 	./$(EXE)

# clean: $(SRC)
# 	rm -f $(EXE) $(EXE).linkinfo result*



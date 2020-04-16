/***********************************************************************
 * PathFinder uses dynamic programming to find a path on a 2-D grid from
 * the bottom row to the top row with the smallest accumulated weights,
 * where each step of the path moves straight ahead or diagonally ahead.
 * It iterates row by row, each node picks a neighboring node in the
 * previous row that has the smallest accumulated weight, and adds its
 * own weight to the sum.
 *
 * This kernel uses the technique of ghost zone optimization
 ***********************************************************************/

// Other header files.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <iostream>
#include "OpenCL.h"

// #include "memprof.h"
#include <thread>
#include <vector>
#include <atomic>
#include <string.h>
#include <mutex>
#include "malloc.h"

using namespace std;

// halo width along one direction when advancing to the next iteration
#define HALO     1
#define STR_SIZE 256
#define DEVICE   0
#define M_SEED   9
#define BENCH_PRINT
#define IN_RANGE(x, min, max)	((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

// Program variables.
int   rows, cols;
int   Ne = rows * cols;
int*  data;
int** wall;
int*  result;
int   pyramid_height;

int compare_result = 0;
int* data_o;
int** wall_o;
int* result_o;

//////////////////////////////
// MBAP parameters
int num_cpu_cores = 1;
int num_gpu_compute_units = 8;
int throttling_mod = 1;
int throttling_alloc = 1;
//////////////////////////////

inline int cpu_first(std::atomic_int* worklist)
{
    int current = worklist->fetch_add(1);
    return current;
}

inline int cpu_next(std::atomic_int* worklist)
{
    int current = worklist->fetch_add(1);
    return current;
}

inline bool cpu_more(int current, int n_tasks)
{
    return current < n_tasks;
}

void dynproc_kernel_MBAP(int k,
						int iteration,
						int* gpuWall,
						int* gpuSrc,
						int* gpuResults,
						int cols,
						int rows,
						int startStep,
						int border,
						int* prev,
						int* result,
						int* outputBuffer,
						size_t* global_size,
						size_t* local_size,
						std::atomic_int* worklist,
						int n_tasks,
						std::mutex &mutx)
{	
	int processed_wgs = 0;
	
	for (int wg_id = cpu_first(worklist); cpu_more(wg_id, n_tasks); wg_id = cpu_next(worklist)){
		processed_wgs++;
		int BLOCK_SIZE = local_size[0];
		int bx = wg_id / (global_size[0] / local_size[0]);

		// tx = local id
		for (int tx = 0; tx < local_size[0]; tx++)
		{
			// calculate the small block size.
			int small_block_cols = BLOCK_SIZE - (iteration*HALO*2);

			// calculate the boundary for the block according to
			// the boundary of its small block
			int blkX = (small_block_cols*bx) - border;
			int blkXmax = blkX+BLOCK_SIZE-1;

			// calculate the global thread coordination
			int xidx = blkX+tx;

			// effective range within this block that falls within
			// the valid range of the input data
			// used to rule out computation outside the boundary.
			int validXmin = (blkX < 0) ? -blkX : 0;
			int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;
			
			int W = tx-1;
			int E = tx+1;

			W = (W < validXmin) ? validXmin : W;
			E = (E > validXmax) ? validXmax : E;

			bool isValid = IN_RANGE(tx, validXmin, validXmax);

			if(IN_RANGE(xidx, 0, cols-1))
			{
				prev[tx] = gpuSrc[xidx];
			}
			
			// barrier(CLK_LOCAL_MEM_FENCE);

			bool computed;
			for (int i = 0; i < iteration; i++)
			{
				computed = false;
				
				if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) && isValid )
				{
					computed = true;
					int left = prev[W];
					int up = prev[tx];
					int right = prev[E];
					int shortest = MIN(left, up);
					shortest = MIN(shortest, right);
					
					int index = cols*(startStep+i)+xidx;
					result[tx] = shortest + gpuWall[index];
					
					// ===================================================================
					// add debugging info to the debug output buffer...
					if (tx==11 && i==0)
					{
						// set bufIndex to what value/range of values you want to know.
						int bufIndex = gpuSrc[xidx];
						// dont touch the line below.
						outputBuffer[bufIndex] = 1;
					}
					// ===================================================================
				}

				// barrier(CLK_LOCAL_MEM_FENCE);

				if(i==iteration-1)
				{
					break;
				}

				if(computed)
				{
					prev[tx] = result[tx];
				}
				// barrier(CLK_LOCAL_MEM_FENCE);
			}
			
			if (computed)
			{
				gpuResults[xidx] = result[tx];
			}
		}
	}
}

void compareResults(int *result_mbap, int *result_original, int size, const char *array_name){
	int fail = 0;
	for(int i=0; i<size; i++){
		if(result_mbap[i] != result_original[i]){
			fail++;
		}
	}
	printf("Array %s: Non-Matching CPU-GPU Outputs Count: %d\n", array_name, fail);
	return;
}


void usage(){
	fprintf(stderr, "Usage: dynproc [num_cpu_cores] [throttling_mod] [throttling_alloc] [row_len] [col_len] [pyramid_height] <[compare_result]>\n");
	fprintf(stderr, "[num_cpu_cores], [throttling_mod], [throttling_alloc] should be non-negative integer.\n");
	fprintf(stderr, "<[compare_result]> is optional.\n");
	
	exit(0);
}

void init(int argc, char** argv)
{
	if(argc < 6 || argc > 7)
		usage();
	
	if( (num_cpu_cores = atoi(argv[1])) < 0 ||
		(throttling_mod = atoi(argv[2])) < 0 ||
		(throttling_alloc = atoi(argv[3])) < 0)
		usage();

	cols = atoi(argv[4]);
	rows = atoi(argv[5]);
	pyramid_height = atoi(argv[6]);

	if(argc == 7)
		compare_result = 1;

	data = new int[rows * cols];
	wall = new int*[rows];
	for (int n = 0; n < rows; n++)
	{
		// wall[n] is set to be the nth row of the data array.
		wall[n] = data + cols * n;
	}
	result = new int[cols];

	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			wall[i][j] = rand() % 10;
		}
	}

	if(compare_result){
		data_o = new int[rows * cols];
		wall_o = new int*[rows];
		for (int n = 0; n < rows; n++)
		{
			wall_o[n] = data_o + cols * n;
		}
		result_o = new int[cols];

		memcpy(data_o, data, sizeof(int)*rows*cols);
	}

#ifdef BENCH_PRINT
	// for (int i = 0; i < rows; i++)
	// {
	// 	for (int j = 0; j < cols; j++)
	// 	{
	// 		printf("%d ", wall[i][j]);
	// 	}
	// 	printf("\n");
	// }
#endif
}

void fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);
}

int main(int argc, char** argv)
{
	init(argc, argv);
	// CreateProfiler();
	
	// Pyramid parameters.
	int borderCols = (pyramid_height) * HALO;
	// int smallBlockCol = ?????? - (pyramid_height) * HALO * 2;
	// int blockCols = cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

	
	/* printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
	   pyramid_height, cols, borderCols, NUMBER_THREADS, blockCols, smallBlockCol); */

	int size = rows * cols;

	// Create and initialize the OpenCL object.
	OpenCL cl(1);  // 1 means to display output (debugging mode).
	cl.init(1);    // 1 means to use GPU. 0 means use CPU.
	cl.gwSize(rows * cols);

	// Create and build the kernel.
	// string kn = "dynproc_kernel";  // the kernel name, for future use.
	string kn = "dynproc_kernel_MBAP";
	size_t localWS = cl.createKernel(kn);

	// Allocate device memory.
	cl_mem d_gpuWall = clCreateBuffer(cl.ctxt(),
	                                  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
	                                  sizeof(cl_int)*(size-cols),
	                                  (data + cols),
	                                  NULL);

	cl_mem d_gpuResult[2];

	d_gpuResult[0] = clCreateBuffer(cl.ctxt(),
	                                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
	                                sizeof(int)*cols,
	                                data,
	                                NULL);

	int *tempData = (int*)memalign(4096, size*sizeof(int));
	for(int i=0; i < size; i++){
		tempData[i] = 0;
	}

	d_gpuResult[1] = clCreateBuffer(cl.ctxt(),
	                                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
	                                sizeof(int)*cols,
	                                tempData,
	                                NULL);

	int* h_outputBuffer = (int*)memalign(4096, 16384*sizeof(int));
	for (int i = 0; i < 16384; i++)
	{
		h_outputBuffer[i] = 0;
	}
	cl_mem d_outputBuffer = clCreateBuffer(cl.ctxt(),
	                                       CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
	                                       sizeof(int)*16384,
	                                       h_outputBuffer,
	                                       NULL);

	int* prevBuffer = (int*)memalign(4096, sizeof(int)*localWS);
	int* resultBuffer = (int*)memalign(4096, sizeof(int)*localWS);
	for (int i = 0; i < localWS; i++)
	{
		prevBuffer[i] = 0;
		resultBuffer[i] = 0;
	}
	cl_mem d_prevBuffer = clCreateBuffer(cl.ctxt(),
										CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
										sizeof(int)*localWS,
										prevBuffer,
										NULL);
	cl_mem d_resultBuffer = clCreateBuffer(cl.ctxt(),
										CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
										sizeof(int)*localWS,
										resultBuffer,
										NULL);

	size_t globalWorkSize[1] = { size };
	size_t localWorkSize[1] = { localWS };

	size_t num_wgs[1];
    num_wgs[0] = globalWorkSize[0] / localWorkSize[0];
    size_t total_num_wgs = num_wgs[0];

    size_t gpu_partitioned_global_work_size[1] = { num_gpu_compute_units * localWorkSize[0] };

	int src = 1, final_ret = 0;
	for (int t = 0; t < rows - 1; t += pyramid_height)
	{
		int temp = src;
		src = final_ret;
		final_ret = temp;

		// Calculate this for the kernel argument...
		int arg0 = MIN(pyramid_height, rows-t-1);
		int theHalo = HALO;

		// Set the kernel arguments.
		clSetKernelArg(cl.kernel(kn), 0,  sizeof(cl_int), (void*) &arg0);
		clSetKernelArg(cl.kernel(kn), 1,  sizeof(cl_mem), (void*) &d_gpuWall);
		clSetKernelArg(cl.kernel(kn), 2,  sizeof(cl_mem), (void*) &d_gpuResult[src]);
		clSetKernelArg(cl.kernel(kn), 3,  sizeof(cl_mem), (void*) &d_gpuResult[final_ret]);
		clSetKernelArg(cl.kernel(kn), 4,  sizeof(cl_int), (void*) &cols);
		clSetKernelArg(cl.kernel(kn), 5,  sizeof(cl_int), (void*) &rows);
		clSetKernelArg(cl.kernel(kn), 6,  sizeof(cl_int), (void*) &t);
		clSetKernelArg(cl.kernel(kn), 7,  sizeof(cl_int), (void*) &borderCols);
		clSetKernelArg(cl.kernel(kn), 8,  sizeof(cl_int), (void*) &theHalo);
		clSetKernelArg(cl.kernel(kn), 9,  sizeof(cl_mem), (void*) &d_prevBuffer);
		clSetKernelArg(cl.kernel(kn), 10, sizeof(cl_mem), (void*) &d_resultBuffer);
		clSetKernelArg(cl.kernel(kn), 11, sizeof(cl_mem), (void*) &d_outputBuffer);
		clSetKernelArg(cl.kernel(kn), 12, sizeof(int), (void *)&throttling_mod);
		clSetKernelArg(cl.kernel(kn), 13, sizeof(int), (void *)&throttling_alloc);    
		
		std::atomic_int* worklist = (std::atomic_int*)clSVMAlloc(cl.ctxt(),
			CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
			sizeof(std::atomic_int), 0);
    	worklist[0].store(0);
    	clSetKernelArgSVMPointer(cl.kernel(kn), 14, worklist);
		clSetKernelArg(cl.kernel(kn), 15, sizeof(int), (void*)&total_num_wgs);

		// StartProfiling();
		cl.launch(kn, gpu_partitioned_global_work_size, localWorkSize);
		// StopProfiling();

		std::vector<std::thread*> threads;
		std::mutex mutx;
		for(int k = 0; k < num_cpu_cores; k++){
			if(src == 0){
				std::thread* thread_NN = new std::thread(
				dynproc_kernel_MBAP, k,
				arg0, (data + cols), data, tempData,
				cols, rows, t, borderCols,
				prevBuffer, resultBuffer, h_outputBuffer,
				globalWorkSize, localWorkSize, worklist, total_num_wgs,
				std::ref(mutx));
				threads.push_back(thread_NN);
			}
			else{
				std::thread* thread_NN = new std::thread(
				dynproc_kernel_MBAP, k,
				arg0, (data + cols), tempData, data,
				cols, rows, t, borderCols,
				prevBuffer, resultBuffer, h_outputBuffer,
				globalWorkSize, localWorkSize, worklist, total_num_wgs,
				std::ref(mutx));
				threads.push_back(thread_NN);
			}
		}

		for(int i = 0; i < threads.size(); i++){
		threads[i]->join();
		}

		clFinish(cl.q());
	}

	// Copy results back to host.
	clEnqueueReadBuffer(cl.q(),                   // The command queue.
	                    d_gpuResult[final_ret],   // The result on the device.
	                    CL_TRUE,                  // Blocking? (ie. Wait at this line until read has finished?)
	                    0,                        // Offset. None in this case.
	                    sizeof(cl_int)*cols,      // Size to copy.
	                    result,                   // The pointer to the memory on the host.
	                    0,                        // Number of events in wait list. Not used.
	                    NULL,                     // Event wait list. Not used.
	                    NULL);                    // Event object for determining status. Not used.


	// Copy string buffer used for debugging from device to host.
	clEnqueueReadBuffer(cl.q(),                   // The command queue.
	                    d_outputBuffer,           // Debug buffer on the device.
	                    CL_TRUE,                  // Blocking? (ie. Wait at this line until read has finished?)
	                    0,                        // Offset. None in this case.
	                    sizeof(cl_char)*16384,    // Size to copy.
	                    h_outputBuffer,           // The pointer to the memory on the host.
	                    0,                        // Number of events in wait list. Not used.
	                    NULL,                     // Event wait list. Not used.
	                    NULL);                    // Event object for determining status. Not used.
	
	// Tack a null terminator at the end of the string.
	h_outputBuffer[16383] = '\0';
	
	if(compare_result){
		OpenCL cl_o(1);  // 1 means to display output (debugging mode).
		cl_o.init(1);    // 1 means to use GPU. 0 means use CPU.
		cl_o.gwSize(rows * cols);

		// Create and build the kernel.
		string kn_o = "dynproc_kernel";  // the kernel name, for future use.
		cl_o.createKernel(kn_o);

		// Allocate device memory.
		cl_mem d_gpuWall_o = clCreateBuffer(cl.ctxt(),
										CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
										sizeof(int)*(size-cols),
										(data_o + cols),
										NULL);

		cl_mem d_gpuResult_o[2];

		d_gpuResult_o[0] = clCreateBuffer(cl.ctxt(),
										CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
										sizeof(int)*cols,
										data_o,
										NULL);

		int *tempData_o = (int*)memalign(4096, cols*sizeof(int));
		for(int i=0; i < cols; i++){
			tempData_o[i] = 0;
		}
		d_gpuResult_o[1] = clCreateBuffer(cl.ctxt(),
										CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
										sizeof(int)*cols,
										tempData_o,
										NULL);

		int* h_outputBuffer_o = (int*)malloc(16384*sizeof(int));
		for (int i = 0; i < 16384; i++)
		{
			h_outputBuffer_o[i] = 0;
		}
		cl_mem d_outputBuffer_o = clCreateBuffer(cl.ctxt(),
											CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
											sizeof(int)*16384,
											h_outputBuffer_o,
											NULL);
		
		int* prevBuffer_o = (int*)memalign(4096, sizeof(int)*localWS);
		int* resultBuffer_o = (int*)memalign(4096, sizeof(int)*localWS);
		for (int i = 0; i < localWS; i++)
		{
			prevBuffer_o[i] = 0;
			resultBuffer_o[i] = 0;
		}
		cl_mem d_prevBuffer_o = clCreateBuffer(cl.ctxt(),
											CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
											sizeof(int)*localWS,
											prevBuffer_o,
											NULL);
		cl_mem d_resultBuffer_o = clCreateBuffer(cl.ctxt(),
											CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
											sizeof(int)*localWS,
											resultBuffer_o,
											NULL);

		size_t globalWorkSize_o[1] = { size };
		size_t localWorkSize_o[1] = { localWS };

		int src = 1, final_ret = 0;
		for (int t = 0; t < rows - 1; t += pyramid_height)
		{
			int temp = src;
			src = final_ret;
			final_ret = temp;

			// Calculate this for the kernel argument...
			int arg0 = MIN(pyramid_height, rows-t-1);
			int theHalo = HALO;

			// Set the kernel arguments.
			clSetKernelArg(cl_o.kernel(kn_o), 0,  sizeof(cl_int), (void*) &arg0);
			clSetKernelArg(cl_o.kernel(kn_o), 1,  sizeof(cl_mem), (void*) &d_gpuWall_o);
			clSetKernelArg(cl_o.kernel(kn_o), 2,  sizeof(cl_mem), (void*) &d_gpuResult_o[src]);
			clSetKernelArg(cl_o.kernel(kn_o), 3,  sizeof(cl_mem), (void*) &d_gpuResult_o[final_ret]);
			clSetKernelArg(cl_o.kernel(kn_o), 4,  sizeof(cl_int), (void*) &cols);
			clSetKernelArg(cl_o.kernel(kn_o), 5,  sizeof(cl_int), (void*) &rows);
			clSetKernelArg(cl_o.kernel(kn_o), 6,  sizeof(cl_int), (void*) &t);
			clSetKernelArg(cl_o.kernel(kn_o), 7,  sizeof(cl_int), (void*) &borderCols);
			clSetKernelArg(cl_o.kernel(kn_o), 8,  sizeof(cl_int), (void*) &theHalo);
			clSetKernelArg(cl.kernel(kn), 9,  sizeof(cl_mem), (void*) &d_prevBuffer_o);
			clSetKernelArg(cl.kernel(kn), 10, sizeof(cl_mem), (void*) &d_resultBuffer_o);
			clSetKernelArg(cl_o.kernel(kn_o), 11, sizeof(cl_mem), (void*) &d_outputBuffer_o);

			cl_o.launch(kn_o, globalWorkSize_o, localWorkSize_o);
		}

		// Copy results back to host.
		clEnqueueReadBuffer(cl_o.q(),                   // The command queue.
							d_gpuResult_o[final_ret],   // The result on the device.
							CL_TRUE,                  // Blocking? (ie. Wait at this line until read has finished?)
							0,                        // Offset. None in this case.
							sizeof(cl_int)*cols,      // Size to copy.
							result_o,                   // The pointer to the memory on the host.
							0,                        // Number of events in wait list. Not used.
							NULL,                     // Event wait list. Not used.
							NULL);                    // Event object for determining status. Not used.


		// Copy string buffer used for debugging from device to host.
		clEnqueueReadBuffer(cl_o.q(),                   // The command queue.
							d_outputBuffer_o,           // Debug buffer on the device.
							CL_TRUE,                  // Blocking? (ie. Wait at this line until read has finished?)
							0,                        // Offset. None in this case.
							sizeof(cl_char)*16384,    // Size to copy.
							h_outputBuffer_o,           // The pointer to the memory on the host.
							0,                        // Number of events in wait list. Not used.
							NULL,                     // Event wait list. Not used.
							NULL);                    // Event object for determining status. Not used.
		
		// Tack a null terminator at the end of the string.
		h_outputBuffer_o[16383] = '\0';

		//compare function
		compareResults(data, data_o, rows*cols, "data");
		compareResults(h_outputBuffer, h_outputBuffer_o, 16384, "h_outputBuffer");

		free(tempData_o);
		free(h_outputBuffer_o);
		free(prevBuffer_o);
		free(resultBuffer_o);
	}

#ifdef BENCH_PRINT
	for (int i = 0; i < cols; i++)
		printf("%d ", data[i]);
	printf("\n");
	for (int i = 0; i < cols; i++)
		printf("%d ", result[i]);
	printf("\n");
#endif

	// Memory cleanup here.
	// ReleaseProfiler();

	free(tempData);
	free(h_outputBuffer);
	free(prevBuffer);
	free(resultBuffer);
	delete[] data;
	delete[] wall;
	delete[] result;
	if(compare_result){
		delete[] data_o;
		delete[] wall_o;
		delete[] result_o;
	}
	
	return EXIT_SUCCESS;
}

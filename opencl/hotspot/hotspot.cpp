#include "hotspot.h"

int num_cpu_cores = 1;
int num_gpu_compute_units = 8;
int throttling_mod = 1;
int throttling_alloc = 1;

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

void hotspot_MBAP(  int id,
					int iteration,  //number of iteration
					global float *power,   //power input
					global float *temp_src,    //temperature input/output
					global float *temp_dst,    //temperature input/output
					int grid_cols,  //Col of grid
					int grid_rows,  //Row of grid
					int border_cols,  // border offset 
					int border_rows,  // border offset
					float Cap,      //Capacitance
					float Rx, 
					float Ry, 
					float Rz, 
					float step,
					size_t* global_size,
					size_t* local_size,
					std::atomic_int* worklist,
					int num_tasks)
{
    int processed_wgs = 0;

    for (int wg_id = cpu_first(worklist);
             cpu_more(wg_id, num_tasks);
             wg_id = cpu_next(worklist)) {
        processed_wgs++;

        for (int global_id = wg_id * local_size[0];
                 global_id < wg_id * local_size[0] + local_size[0];
                 global_id++) {
            //FIXME: fill the kernel content
        }
    }

    printf("num_wgs_thread%d: %d\n", id, processed_wgs);
}

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file) {

	int i,j, index=0;
	FILE *fp;
	char str[STR_SIZE];

	if( (fp = fopen(file, "w" )) == 0 )
          printf( "The file was not opened\n" );


	for (i=0; i < grid_rows; i++) 
	 for (j=0; j < grid_cols; j++)
	 {

		 sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
		 fputs(str,fp);
		 index++;
	 }
		
      fclose(fp);	
}


void readinput(float *vect, int grid_rows, int grid_cols, char *file) {

  	int i,j;
	FILE *fp;
	char str[STR_SIZE];
	float val;

	if( (fp  = fopen(file, "r" )) ==0 )
            fatal( "The file was not opened" );


	for (i=0; i <= grid_rows-1; i++) 
	 for (j=0; j <= grid_cols-1; j++)
	 {
		if (fgets(str, STR_SIZE, fp) == NULL) fatal("Error reading file\n");
		if (feof(fp))
			fatal("not enough lines in file");
		//if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
		if ((sscanf(str, "%f", &val) != 1))
			fatal("invalid file format");
		vect[i*grid_cols+j] = val;
	}

	fclose(fp);	

}


/*
   compute N time steps
*/

int compute_tran_temp(cl_mem MatrixPower, cl_mem MatrixTemp[2], int col, int row, \
		int total_iterations, int num_iterations, int blockCols, int blockRows, int borderCols, int borderRows,
		float *TempCPU, float *PowerCPU, float *MatrixOut) 
{ 
	
	float grid_height = chip_height / row;
	float grid_width = chip_width / col;

	float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	float Rz = t_chip / (K_SI * grid_height * grid_width);

	float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	float step = PRECISION / max_slope;
	int t;

	int src = 0, dst = 1;
	
	cl_int error;
	
	// Determine GPU work group grid
	size_t global_work_size[2];
	global_work_size[0] = BLOCK_SIZE * blockCols;
	global_work_size[1] = BLOCK_SIZE * blockRows;
	size_t local_work_size[2];
	local_work_size[0] = BLOCK_SIZE;
	local_work_size[1] = BLOCK_SIZE;
	
	size_t num_wgs[2];
	num_wgs[0] = global_work_size[0] / local_work_size[0];
	num_wgs[1] = global_work_size[1] / local_work_size[1];
	size_t total_num_wgs = num_wgs[0] * num_wgs[1];

	size_t gpu_partitioned_global_work_size[2];
    gpu_partitioned_global_work_size[0] = num_gpu_compute_units * local_work_size[0];
	gpu_partitioned_global_work_size[1] = global_work_size[1];

	std::atomic_int* worklist;
	int n_tasks = total_num_wgs;

	long long start_time = get_time();	
	
	for (t = 0; t < total_iterations; t += num_iterations) {
		
		// Specify kernel arguments
		int iter = MIN(num_iterations, total_iterations - t);
		clSetKernelArg(kernel, 0, sizeof(int), (void *) &iter);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &MatrixPower);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &MatrixTemp[src]);
		clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &MatrixTemp[dst]);
		clSetKernelArg(kernel, 4, sizeof(int), (void *) &col);
		clSetKernelArg(kernel, 5, sizeof(int), (void *) &row);
		clSetKernelArg(kernel, 6, sizeof(int), (void *) &borderCols);
		clSetKernelArg(kernel, 7, sizeof(int), (void *) &borderRows);
		clSetKernelArg(kernel, 8, sizeof(float), (void *) &Cap);
		clSetKernelArg(kernel, 9, sizeof(float), (void *) &Rx);
		clSetKernelArg(kernel, 10, sizeof(float), (void *) &Ry);
		clSetKernelArg(kernel, 11, sizeof(float), (void *) &Rz);
		clSetKernelArg(kernel, 12, sizeof(float), (void *) &step);
		
		//throttling mod
		clSetKernelArg(kernel, 13, sizeof(int), (void *) &throttling_mod);
		clSetKernelArg(kernel, 14, sizeof(int), (void *) &throttling_alloc);
		
		worklist = (std::atomic_int*)clSVMAlloc(context,
			CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
			sizeof(std::atomic_int), 0);
		worklist[0].store(0);
		error = clSetKernelArgSVMPointer(kernel, 15, worklist);
		if(error != CL_SUCCESS) fatal_CL(error, __LINE__);

		error = clSetKernelArg(kernel, 16, sizeof(int), (void*)&n_tasks);

		// Launch kernel
		// StartProfiling();
		error = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
		// StopProfiling();

		// Flush the queue
		error = clFlush(command_queue);
		if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
		
		std::vector<std::thread*> threads;

		for (int i = 0; i < num_cpu_cores; i++) {
			std::thread* thread_;
			if(src == 0)
				thread_ = new std::thread(hotspot_MBAP, i,
					iter, PowerCPU, TempCPU, MatrixOut,
					col, row, borderCols, borderRows,
					Cap, Rx, Ry, Rz, step,
					global_work_size, local_work_size,
					worklist, n_tasks);
			else
				thread_ = new std::thread(hotspot_MBAP, i,
					iter, PowerCPU, MatrixOut, TempCPU,
					col, row, borderCols, borderRows,
					Cap, Rx, Ry, Rz, step,
					global_work_size, local_work_size,
					worklist, n_tasks);
			threads.push_back(thread_);
		}

		for (int i = 0; i < threads.size(); i++) {
			threads[i]->join();
		}

		clFinish(command_queue);
		
		// Swap input and output GPU matrices
		src = 1 - src;
		dst = 1 - dst;
	}
	
	// Wait for all operations to finish
	error = clFinish(command_queue);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
	long long end_time = get_time();
	long long total_time = (end_time - start_time);	
	printf("\nKernel time: %.3f seconds\n", ((float) total_time) / (1000*1000));
	
	return src;
}

void usage(int argc, char **argv) {
	// fprintf(stderr, "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
	// fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
	// fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
	// fprintf(stderr, "\t<sim_time>   - number of iterations\n");
	// fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
	// fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
	// fprintf(stderr, "\t<output_file> - name of the output file\n");
	// exit(1);

	fprintf(stderr, "Usage: %s <num_cpu_cores> <throttling_mod> <throttling_alloc> <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
	fprintf(stderr, "\t<num_cpu_cores> - the number of cpu cores to use\n");
	fprintf(stderr, "\t<throttling_mod> - throttling mod\n");
	fprintf(stderr, "\t<throttling_alloc> - throttling allocation\n");
	fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
	fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
	fprintf(stderr, "\t<sim_time>   - number of iterations\n");
	fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
	fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
	fprintf(stderr, "\t<output_file> - name of the output file\n");
	exit(1);
}

int main(int argc, char** argv) {

	// CreateProfiler();
    int size;
    int grid_rows,grid_cols = 0;
    // float *FilesavingTemp,*FilesavingPower; //,*MatrixOut;
	float *FilesavingTemp,*FilesavingPower,*MatrixOut;
    char *tfile, *pfile, *ofile;
    
    int total_iterations = 60;
    int pyramid_height = 1; // number of iterations
	
	// if (argc < 7)
	// 	usage(argc, argv);
	// if((grid_rows = atoi(argv[1]))<=0||
	//    (grid_cols = atoi(argv[1]))<=0||
    //    (pyramid_height = atoi(argv[2]))<=0||
    //    (total_iterations = atoi(argv[3]))<=0)
	// 	usage(argc, argv);

	if (argc != 10)
		usage(argc, argv);
	if((num_cpu_cores = atoi(argv[1])) <= 0 ||
	   (throttling_mod = atoi(argv[2])) <= 0 ||
	   (throttling_alloc = atoi(argv[3])) <= 0 ||
	   (grid_rows = atoi(argv[4])) <= 0 ||
	   (grid_cols = atoi(argv[4])) <= 0 ||
       (pyramid_height = atoi(argv[5])) <= 0 ||
       (total_iterations = atoi(argv[6])) <= 0)
		usage(argc, argv);
		
	tfile=argv[7];
    pfile=argv[8];
    ofile=argv[9];
	
    size=grid_rows*grid_cols;

  	printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

	cl_int error;
	cl_uint num_platforms;
	
	// Get the number of platforms
	error = clGetPlatformIDs(0, NULL, &num_platforms);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
	// Get the list of platforms
	cl_platform_id* platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * num_platforms);
	error = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
	// Print the chosen platform (if there are multiple platforms, choose the first one)
	cl_platform_id platform = platforms[0];
	char pbuf[100];
	error = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(pbuf), pbuf, NULL);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	printf("Platform: %s\n", pbuf);
	
	// Create a GPU context
	cl_context_properties context_properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0};
    context = clCreateContextFromType(context_properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
	// Get and print the chosen device (if there are multiple devices, choose the first one)
	size_t devices_size;
	error = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &devices_size);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	cl_device_id *devices = (cl_device_id *) malloc(devices_size);
	error = clGetContextInfo(context, CL_CONTEXT_DEVICES, devices_size, devices, NULL);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	device = devices[0];
	error = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(pbuf), pbuf, NULL);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	printf("Device: %s\n", pbuf);
	
	// Create a command queue
	command_queue = clCreateCommandQueue(context, device, 0, &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

    // --------------- pyramid parameters --------------- 
    int borderCols = (pyramid_height)*EXPAND_RATE/2;
    int borderRows = (pyramid_height)*EXPAND_RATE/2;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
    int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

    FilesavingTemp = (float *) memalign(4096, size*sizeof(float));
    FilesavingPower = (float *) memalign(4096, size*sizeof(float));
    MatrixOut = (float *) memalign(4096, size*sizeof(float));

    if( !FilesavingPower || !FilesavingTemp) // || !MatrixOut)
        fatal("unable to allocate memory");
	
	// Read input data from disk
    readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
    readinput(FilesavingPower, grid_rows, grid_cols, pfile);
	
	// Load kernel source from file
	const char *source = load_kernel_source("hotspot_kernel.cl");
	size_t sourceSize = strlen(source);
	
	// Compile the kernel
    cl_program program = clCreateProgramWithSource(context, 1, &source, &sourceSize, &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
	char clOptions[110];
	//  sprintf(clOptions,"-I../../src"); 
	// sprintf(clOptions," ");
	sprintf(clOptions,"-I. -cl-std=CL2.0");
#ifdef BLOCK_SIZE
	// sprintf(clOptions + strlen(clOptions), " -DBLOCK_SIZE=%d", BLOCK_SIZE);
	sprintf(clOptions + strlen(clOptions), " -DBLOCK_SIZE=%d", BLOCK_SIZE);
#endif

    // Create an executable from the kernel
	error = clBuildProgram(program, 1, &device, clOptions, NULL, NULL);
	// Show compiler warnings/errors
	static char log[65536]; memset(log, 0, sizeof(log));
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
	if (strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	// FIXME: change the following line to decide which kernel will be executed
    // kernel = clCreateKernel(program, "hotspot", &error);
	kernel = clCreateKernel(program, "hotspot_MBAP", &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
		
	long long start_time = get_time();

	///////////////////////////////////////////////////
	// the start of original codes
	///////////////////////////////////////////////////	
	
	// // Create two temperature matrices and copy the temperature input data
	// cl_mem MatrixTemp[2];
	// // Create input memory buffers on device
	// MatrixTemp[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * size, FilesavingTemp, &error);
	// if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
    
    // // Lingjie Zhang modifited at Nov 1, 2015
    // //MatrixTemp[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * size, NULL, &error);
    // MatrixTemp[1] = clCreateBuffer(context, CL_MEM_READ_WRITE , sizeof(float) * size, NULL, &error);
    // // end Lingjie Zhang modification
    
    // if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
	// // Copy the power input data
	// cl_mem MatrixPower = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * size, FilesavingPower, &error);
	// if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
	// // Perform the computation
	// int ret = compute_tran_temp(MatrixPower, MatrixTemp, grid_cols, grid_rows, total_iterations, pyramid_height,
	// 							blockCols, blockRows, borderCols, borderRows, FilesavingTemp, FilesavingPower);
	
	// // Copy final temperature data back
	// cl_float *MatrixOut = (cl_float *) clEnqueueMapBuffer(command_queue, MatrixTemp[ret], CL_TRUE, CL_MAP_READ, 0, sizeof(float) * size, 0, NULL, NULL, &error);
	// if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
	// long long end_time = get_time();	
	// printf("Total time: %.3f seconds\n", ((float) (end_time - start_time)) / (1000*1000));
	
	// // Write final output to output file
    // writeoutput(MatrixOut, grid_rows, grid_cols, ofile);
    
	// error = clEnqueueUnmapMemObject(command_queue, MatrixTemp[ret], (void *) MatrixOut, 0, NULL, NULL);
	// if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

	///////////////////////////////////////////////////
	// end of original codes
	///////////////////////////////////////////////////	

	// Create two temperature matrices and copy the temperature input data
	cl_mem MatrixTemp[2];
	// Create input memory buffers on device
	MatrixTemp[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * size, FilesavingTemp, &error);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
    MatrixTemp[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR , sizeof(float) * size, MatrixOut, &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	// Copy the power input data
	cl_mem MatrixPower = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * size, FilesavingPower, &error);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
	// Perform the computation
	int ret = compute_tran_temp(MatrixPower, MatrixTemp, grid_cols, grid_rows, total_iterations, pyramid_height,
								blockCols, blockRows, borderCols, borderRows, FilesavingTemp, FilesavingPower, MatrixOut);

	long long end_time = get_time();	
	printf("Total time: %.3f seconds\n", ((float) (end_time - start_time)) / (1000*1000));
	
	// Write final output to output file
	if(ret){
		writeoutput(MatrixOut, grid_rows, grid_cols, ofile);
	}
	else{
		writeoutput(FilesavingTemp, grid_rows, grid_cols, ofile);
	}

	clReleaseMemObject(MatrixTemp[0]);
	clReleaseMemObject(MatrixTemp[1]);
	clReleaseMemObject(MatrixPower);
	
    clReleaseContext(context);

	// ReleaseProfiler();
	return 0;
}

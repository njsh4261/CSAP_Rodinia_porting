#ifndef __NEAREST_NEIGHBOR__
#define __NEAREST_NEIGHBOR__

#include "nearestNeighbor.h"

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05
#define SMALL_FLOAT_VAL 0.00000001f

cl_context context=NULL;

// MBAP parameters
int num_cpu_cores = 1;
int num_gpu_compute_units = 8;
int throttling_mod = 1;
int throttling_alloc = 1;

int thread_executed_count = 0;
int cal_count = 0;

#define LOCAL_WORK_GROUP_SIZE 64

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

void NearestNeighbor_MBAP(int id,
                          LatLong *d_locations,
                          float *d_distances,
                          const int numRecords,
                          float lat,
                          float lng,
                          size_t* global_size,
                          size_t* local_size,
                          std::atomic_int* worklist,
                          int num_tasks,
                          std::mutex& mutx)
{
  int processed_wgs = 0;
  thread_executed_count++;

  for (int wg_id = cpu_first(worklist);
        cpu_more(wg_id, num_tasks);
        wg_id = cpu_next(worklist)) {
    processed_wgs++;
    for (int global_id = wg_id * local_size[0];
          global_id < wg_id * local_size[0] + local_size[0];
          global_id++)
    {
      if (global_id < numRecords) {
        LatLong *latLong = d_locations+global_id; 
        float *dist=d_distances+global_id;
        *dist = sqrt((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
      }
    }
  }
  printf("num_wgs_thread%d: %d\n", id, processed_wgs);
}

float absVal(float a)
{
    if(a < 0)
    {
        return (a * -1);
    }
    else
    { 
        return a;
    }
}

float percentDiff(float val1, float val2)
{
    if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01))
    {
        return 0.0f;
    }

    else
    {
        return 100.0f * (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
    }
} 

void compareResults(float *distance_mbap, float *distance_original, int numRecords)
{
	int fail = 0;
	for (int i=0; i<numRecords; i++)
	{
		if (percentDiff(distance_mbap[i], distance_original[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}		
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[]) {
  std::vector<Record> records;
  float *recordDistances;
  std::vector<LatLong> locations;
  int i;
  // args
  char filename[100];
  int resultsCount=10, quiet=0, timing=0, check_result=0, platform=-1, device=-1;
  float lat=0.0, lng=0.0;
  
  // parse command line
  if (parseCommandline(argc, argv, filename, &resultsCount, &lat, &lng,
                     &quiet, &timing, &platform, &device, &check_result)) {
                       //  &quiet, &timing, &platform, &device)) {
    printUsage();
    return 0;
  }
  
  int numRecords = loadData(filename,records,locations);

  if (!quiet) {
    printf("Number of records: %d\n",numRecords);
    printf("Finding the %d closest neighbors.\n",resultsCount);
  }

  if (resultsCount > numRecords) resultsCount = numRecords;

  context = cl_init_context(platform,device,quiet);
  
  recordDistances = OpenClFindNearestNeighbors(
    context, numRecords, locations, lat, lng, timing, check_result
  );

  // find the resultsCount least distances
  findLowest(records,recordDistances,numRecords,resultsCount);

  // print out results
  if (!quiet)
    for(i=0;i<resultsCount;i++) {
      printf("%s --> Distance=%f\n",records[i].recString,records[i].distance);
    }
  free(recordDistances);
  return 0;
}

float *OpenClFindNearestNeighbors(
	cl_context context,
	int numRecords,
	std::vector<LatLong> &locations, float lat, float lng,
  int timing, int check_result)
{
    // 1. set up kernel
    cl_kernel NN_kernel;
    cl_int status;
    cl_program cl_NN_program;
    char compileOption[50];
    sprintf(compileOption, "-I. -cl-std=CL2.0");
    cl_NN_program = cl_compileProgram((char *)"nearestNeighbor_kernel.cl", compileOption);
    
    NN_kernel = clCreateKernel(cl_NN_program, "NearestNeighbor_MBAP", &status);
    status = cl_errChk(status, (char *)"Error Creating Nearest Neighbor kernel",true);
    if(status)exit(1);
    
    // 2. set up memory on device and send ipts data to device
    // copy ipts(1,2) to device
    // also need to alloate memory for the distancePoints
    cl_mem d_locations;
    cl_mem d_distances;

    cl_int error=0;

    d_locations = clCreateBuffer(context,
        // CL_MEM_READ_ONLY,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        sizeof(LatLong) * numRecords,
        // NULL,
        &locations[0],
        &error);

    float *distances = (float *)malloc(sizeof(float) * numRecords);
    d_distances = clCreateBuffer(context,
        // CL_MEM_READ_WRITE,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
        sizeof(float) * numRecords,
        // NULL,
        distances,
        &error);

    cl_command_queue command_queue = cl_getCommandQueue();
    cl_event writeEvent,kernelEvent,readEvent;
    // error = clEnqueueWriteBuffer(command_queue,
    //            d_locations,
    //            1, // change to 0 for nonblocking write
    //            0, // offset
    //            sizeof(LatLong) * numRecords,
    //            &locations[0],
    //            0,
    //            NULL,
    //            &writeEvent);

    // 3. send arguments to device
    cl_int argchk;
    argchk  = clSetKernelArg(NN_kernel, 0, sizeof(cl_mem), (void *)&d_locations);
    argchk |= clSetKernelArg(NN_kernel, 1, sizeof(cl_mem), (void *)&d_distances);
    argchk |= clSetKernelArg(NN_kernel, 2, sizeof(int), (void *)&numRecords);
    argchk |= clSetKernelArg(NN_kernel, 3, sizeof(float), (void *)&lat);
    argchk |= clSetKernelArg(NN_kernel, 4, sizeof(float), (void *)&lng);
    argchk |= clSetKernelArg(NN_kernel, 5, sizeof(int), (void *)&throttling_mod);
    argchk |= clSetKernelArg(NN_kernel, 6, sizeof(int), (void *)&throttling_alloc);
    cl_errChk(argchk,"ERROR in Setting Nearest Neighbor kernel args",true);

    // 4. enqueue kernel
    // size_t globalWorkSize[1];
    size_t globalWorkSize[1], localWorkSize[1];
    globalWorkSize[0] = numRecords;
    localWorkSize[0] = numRecords < LOCAL_WORK_GROUP_SIZE ? numRecords : LOCAL_WORK_GROUP_SIZE;
    if (numRecords % LOCAL_WORK_GROUP_SIZE)
      globalWorkSize[0] += LOCAL_WORK_GROUP_SIZE - (numRecords % LOCAL_WORK_GROUP_SIZE);

    size_t num_wgs[1];
    num_wgs[0] = globalWorkSize[0] / localWorkSize[0];
    size_t total_num_wgs = num_wgs[0];

    size_t gpu_partitioned_global_work_size[1] = { num_gpu_compute_units * localWorkSize[0] };

    std::atomic_int* worklist = (std::atomic_int*)clSVMAlloc(context,
        CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
        sizeof(std::atomic_int), 0);
    worklist[0].store(0);
    error = clSetKernelArgSVMPointer(NN_kernel, 7, worklist);
    cl_errChk(argchk,"ERROR in Setting SVM pointer",true);

    printf("worklist 0: %d\n", worklist[0].load());
    
    int n_tasks = total_num_wgs;
    error = clSetKernelArg(NN_kernel, 8, sizeof(int), (void*)&n_tasks);
    cl_errChk(argchk,"ERROR in Setting Fan1 kernel args",true);

    error = clEnqueueNDRangeKernel(
        command_queue,  NN_kernel, 1, 0,
        gpu_partitioned_global_work_size,
        localWorkSize,
        0, NULL, &kernelEvent);

    cl_errChk(error,"ERROR in Executing Kernel NearestNeighbor",true);

    std::vector<std::thread*> threads;
    std::mutex mutx;
    for(int i = 0; i < num_cpu_cores; i++){
      std::thread* thread_NN = new std::thread(
        NearestNeighbor_MBAP, i,
        &locations[0], distances, numRecords, lat, lng,
        globalWorkSize, localWorkSize, worklist, n_tasks,
        std::ref(mutx));
      threads.push_back(thread_NN);
    }

    for(int i = 0; i < threads.size(); i++){
      threads[i]->join();
    }

    clFinish(command_queue);


    printf("Thread executed count: %d\n", thread_executed_count);
    printf("Number of calculation on CPU: %d\n", cal_count);
    printf("GPU work size: %d\n", gpu_partitioned_global_work_size[0]);

    // 5. transfer data off of device
    
    // create distances std::vector
    // float *distances = (float *)malloc(sizeof(float) * numRecords);

    // error = clEnqueueReadBuffer(command_queue,
    //     d_distances,
    //     1, // change to 0 for nonblocking write
    //     0, // offset
    //     sizeof(float) * numRecords,
    //     distances,
    //     0,
    //     NULL,
    //     &readEvent);

    // cl_errChk(error,"ERROR with clEnqueueReadBuffer",true);

    if(check_result) {
      cl_kernel NN_kernel_original = clCreateKernel(cl_NN_program, "NearestNeighbor", &status);
      status = cl_errChk(status, (char *)"Error Creating Nearest Neighbor kernel",true);
      if(status)exit(1);

      cl_mem d_locations_original, d_distances_original;

      d_locations_original = clCreateBuffer(context,
        CL_MEM_USE_HOST_PTR,
        sizeof(LatLong) * numRecords,
        &locations[0],
        &error);

      float *distances_original = (float *)malloc(sizeof(float) * numRecords);
      d_distances_original = clCreateBuffer(context,
        CL_MEM_USE_HOST_PTR,
        sizeof(float) * numRecords,
        distances_original,
        &error);

      command_queue = cl_getCommandQueue();

      // reset kernel arguments
      argchk  = clSetKernelArg(NN_kernel_original, 0, sizeof(cl_mem), (void *)&d_locations_original);
      argchk |= clSetKernelArg(NN_kernel_original, 1, sizeof(cl_mem), (void *)&d_distances_original);
      argchk |= clSetKernelArg(NN_kernel_original, 2, sizeof(int), (void *)&numRecords);
      argchk |= clSetKernelArg(NN_kernel_original, 3, sizeof(float), (void *)&lat);
      argchk |= clSetKernelArg(NN_kernel_original, 4, sizeof(float), (void *)&lng);
      cl_errChk(argchk,"ERROR in Setting Nearest Neighbor kernel args",true);

      size_t globalWS[1] = { numRecords };

      // execute the original kernel
      error = clEnqueueNDRangeKernel(
        command_queue,  NN_kernel_original, 1, 0,
        globalWS, NULL,
        0, NULL, &kernelEvent);
      cl_errChk(error,"ERROR in Executing Kernel NearestNeighbor",true);
      
      error = clEnqueueReadBuffer(command_queue,
        d_distances_original,
        1, // change to 0 for nonblocking write
        0, // offset
        sizeof(float) * numRecords,
        distances_original,
        0,
        NULL,
        &readEvent);

      compareResults(distances, distances_original, numRecords);
    }

    if (timing) {
        // clFinish(command_queue);
        cl_ulong eventStart,eventEnd,totalTime=0;
        printf("# Records\tWrite(s) [size]\t\tKernel(s)\tRead(s)  [size]\t\tTotal(s)\n");
        printf("%d        \t",numRecords);
        // Write Buffer
        error = clGetEventProfilingInfo(writeEvent,CL_PROFILING_COMMAND_START,
                                        sizeof(cl_ulong),&eventStart,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Write Start)",true); 
        error = clGetEventProfilingInfo(writeEvent,CL_PROFILING_COMMAND_END,
                                        sizeof(cl_ulong),&eventEnd,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Write End)",true);

        printf("%f [%.2fMB]\t",(float)((eventEnd-eventStart)/1e9),(float)((sizeof(LatLong) * numRecords)/1e6));
        totalTime += eventEnd-eventStart;
        // Kernel
        error = clGetEventProfilingInfo(kernelEvent,CL_PROFILING_COMMAND_START,
                                        sizeof(cl_ulong),&eventStart,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Kernel Start)",true); 
        error = clGetEventProfilingInfo(kernelEvent,CL_PROFILING_COMMAND_END,
                                        sizeof(cl_ulong),&eventEnd,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Kernel End)",true);

        printf("%f\t",(float)((eventEnd-eventStart)/1e9));
        totalTime += eventEnd-eventStart;
        // Read Buffer
        error = clGetEventProfilingInfo(readEvent,CL_PROFILING_COMMAND_START,
                                        sizeof(cl_ulong),&eventStart,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Read Start)",true); 
        error = clGetEventProfilingInfo(readEvent,CL_PROFILING_COMMAND_END,
                                        sizeof(cl_ulong),&eventEnd,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Read End)",true);

        printf("%f [%.2fMB]\t",(float)((eventEnd-eventStart)/1e9),(float)((sizeof(float) * numRecords)/1e6));
        totalTime += eventEnd-eventStart;
        
        printf("%f\n\n",(float)(totalTime/1e9));
    }
    // 6. return finalized data and release buffers
    clReleaseMemObject(d_locations);
    clReleaseMemObject(d_distances);
	return distances;
}

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations){
    FILE   *flist,*fp;
	int    i=0;
	char dbname[64];
	int recNum=0;
	
    /**Main processing **/
    
    flist = fopen(filename, "r");
	while(!feof(flist)) {
		/**
		* Read in REC_WINDOW records of length REC_LENGTH
		* If this is the last file in the filelist, then done
		* else open next file to be read next iteration
		*/
		if(fscanf(flist, "%s\n", dbname) != 1) {
            fprintf(stderr, "error reading filelist\n");
            exit(0);
        }
        fp = fopen(dbname, "r");
        if(!fp) {
            printf("error opening a db\n");
            exit(1);
        }
        // read each record
        while(!feof(fp)){
            Record record;
            LatLong latLong;
            fgets(record.recString,49,fp);
            fgetc(fp); // newline
            if (feof(fp)) break;
            
            // parse for lat and long
            char substr[6];
            
            for(i=0;i<5;i++) substr[i] = *(record.recString+i+28);
            substr[5] = '\0';
            latLong.lat = atof(substr);
            
            for(i=0;i<5;i++) substr[i] = *(record.recString+i+33);
            substr[5] = '\0';
            latLong.lng = atof(substr);
            
            locations.push_back(latLong);
            records.push_back(record);
            recNum++;
        }
        fclose(fp);
    }
    fclose(flist);
    return recNum;
}

void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN){
  int i,j;
  float val;
  int minLoc;
  Record *tempRec;
  float tempDist;
  
  for(i=0;i<topN;i++) {
    minLoc = i;
    for(j=i;j<numRecords;j++) {
      val = distances[j];
      if (val < distances[minLoc]) minLoc = j;
    }
    // swap locations and distances
    tempRec = &records[i];
    records[i] = records[minLoc];
    records[minLoc] = *tempRec;
    
    tempDist = distances[i];
    distances[i] = distances[minLoc];
    distances[minLoc] = tempDist;
    
    // add distance to the min we just found
    records[i].distance = distances[i];
  }
}

int parseCommandline(int argc, char *argv[], char* filename,
                     int *r, float *lat, float *lng,
                     int *q, int *t, int *p, int *d, int *c){
                       //  int *q, int *t, int *p, int *d){
    int i;
    if (argc < 5) return 1; // error

    int input_num_cpu_cores = atoi(argv[1]);
    int input_throttling_mod = atoi(argv[2]);
    int input_throttling_alloc = atoi(argv[3]);

    if(input_num_cpu_cores<0 || input_throttling_mod<0 || input_throttling_alloc<0)
      return 1;
    
    num_cpu_cores = input_num_cpu_cores;
    throttling_mod = input_throttling_mod;
    input_throttling_alloc = throttling_alloc;
    
    strncpy(filename,argv[4],100);
    char flag;
    for(i=5;i<argc;i++) {
      if (argv[i][0]=='-') {// flag
        flag = argv[i][1];
          switch (flag) {
            case 'r': // number of results
              i++;
              *r = atoi(argv[i]);
              break;
            case 'l': // lat or lng
              if (argv[i][2]=='a') {//lat
                *lat = atof(argv[i+1]);
              }
              else {//lng
                *lng = atof(argv[i+1]);
              }
              i++;
              break;
            case 'h': // help
              return 1;
              break;
            case 'q': // quiet
              *q = 1;
              break;
            case 't': // timing
              *t = 1;
              break;
            case 'p': // platform
              i++;
              *p = atoi(argv[i]);
              break;
            case 'd': // device
              i++;
              *d = atoi(argv[i]);
              break;
            case 'c':
              i++;
              *c = 1;
              break;
        }
      }
    }
    if ((*d >= 0 && *p<0) || (*p>=0 && *d<0)) // both p and d must be specified if either are specified
      return 1;
    return 0;
}

void printUsage(){
  printf("Nearest Neighbor Usage\n");
  printf("\n");
  printf("nearestNeighbor [num_cpu_cores] [throttling_mod] [throttling_alloc] [filename] -r [int] -lat [float] -lng [float] [-hqt] [-p [int] -d [int]]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./nearestNeighbor 1 1 1 filelist.txt -r 5 -lat 30 -lng 90\n");
  printf("\n");
  printf("num_cpu_cores     the number of cpu cores to use\n");
  printf("throttling_mod    throttling modular number\n");
  printf("throttling_alloc  throttling allocation\n");
  printf("\n");
  printf("filename     the filename that lists the data input files\n");
  printf("-r [int]     the number of records to return (default: 10)\n");
  printf("-lat [float] the latitude for nearest neighbors (default: 0)\n");
  printf("-lng [float] the longitude for nearest neighbors (default: 0)\n");
  printf("\n");
  printf("-h, --help   Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("-c           Run both of original kernel and MBAP kernel and compare the result\n");
  printf("\n");
  printf("-p [int]     Choose the platform (must choose both platform and device)\n");
  printf("-d [int]     Choose the device (must choose both platform and device)\n");
  printf("\n");
  printf("\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
  printf("       2. If you declare either the device or the platform,\n");
  printf("          you must declare both.\n\n");
}

#endif


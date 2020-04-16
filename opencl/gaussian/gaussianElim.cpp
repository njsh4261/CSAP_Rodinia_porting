#ifndef __GAUSSIAN_ELIMINATION__
#define __GAUSSIAN_ELIMINATION__

#include "gaussianElim.h"
#include <math.h>

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE_0 RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE_0 RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE_0 RD_WG_SIZE
#else
        // #define BLOCK_SIZE_0 0
        #define BLOCK_SIZE_0 16
#endif

//2D defines. Go from specific to general                                                
#ifdef RD_WG_SIZE_1_0
        #define BLOCK_SIZE_1_X RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
        #define BLOCK_SIZE_1_X RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE_1_X RD_WG_SIZE
#else
        // #define BLOCK_SIZE_1_X 0
        #define BLOCK_SIZE_1_X 16
#endif

#ifdef RD_WG_SIZE_1_1
        #define BLOCK_SIZE_1_Y RD_WG_SIZE_1_1
#elif defined(RD_WG_SIZE_1)
        #define BLOCK_SIZE_1_Y RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE_1_Y RD_WG_SIZE
#else
        // #define BLOCK_SIZE_1_Y 0
        #define BLOCK_SIZE_1_Y 16
#endif

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05
#define SMALL_FLOAT_VAL 0.00000001f

cl_context context=NULL;

//////////////////////////////
// MBAP parameters
int num_cpu_cores = 1;
int num_gpu_compute_units = 8;
int throttling_mod = 1;
int throttling_alloc = 1;
//////////////////////////////

///////////////////////////////////////////////
// MBAP functions

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

void Fan1_MBAP(int id,
               float *m_dev,
               float *a_dev,
               float *b_dev,
               const int size,
               const int t,
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
          global_id++)
    {
      int globalId = global_id;
      if (globalId < size-1-t) {
        *(m_dev + size * (globalId + t + 1)+t) = *(a_dev + size * (globalId + t + 1) + t) / *(a_dev + size * t + t);
      }
    }
  }
  // printf("num_wgs_thread%d: %d\n", id, processed_wgs);
}

void Fan2_MBAP(int id,
               float *m_dev,
               float *a_dev,
               float *b_dev,
               const int size,
               const int t,
               size_t* global_size,
               size_t* local_size,
               std::atomic_int* worklist,
               int num_tasks,
               std::mutex &mutx)
{
  int processed_wgs = 0;

  int _group_id[2];
  int _global_id[2];

  for (int wg_id = cpu_first(worklist);
             cpu_more(wg_id, num_tasks);
             wg_id = cpu_next(worklist)) {
    processed_wgs++;

    _group_id[0] = wg_id / (global_size[1] / local_size[1]);
    _group_id[1] = wg_id % (global_size[1] / local_size[1]);

    _global_id[0] = _group_id[0] * local_size[0];
    _global_id[1] = _group_id[1] * local_size[1];

    int global_id[2];
    for (global_id[0] = _global_id[0]; global_id[0] < _global_id[0] + local_size[0]; global_id[0]++) {
      for (global_id[1] = _global_id[1]; global_id[1] < _global_id[1] + local_size[1]; global_id[1]++) {
        int globalIdx = global_id[0];
        int globalIdy = global_id[1];
        if (globalIdx < size-1-t && globalIdy < size-t) {
          // mutx.lock();
          a_dev[size*(globalIdx+1+t)+(globalIdy+t)] -= m_dev[size*(globalIdx+1+t)+t] * a_dev[size*t+(globalIdy+t)];
          // mutx.unlock();
          if(globalIdy == 0){
            // mutx.lock();
            b_dev[globalIdx+1+t] -= m_dev[size*(globalIdx+1+t)+(globalIdy+t)] * b_dev[t];
            // mutx.unlock();
          }
        }
      }
    }  
  }
  // printf("num_wgs_thread%d: %d\n", id, processed_wgs);
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

void compareResults(float *result_mbap, float *result_original, int size, const char *array_name)
{
  FILE *fp = fopen("compare.txt", "w");
	int fail = 0;
	for (int i=0; i<size; i++)
	{
		if (percentDiff(result_mbap[i], result_original[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
      fprintf(fp, "%s %d: %f %f\n", array_name, i, result_mbap[i], result_original[i]);
			fail++;
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
  fclose(fp);
  return;
}

//////////////////////////////////////////////

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
void
create_matrix(float *m, int size){
  int i,j;
  float lamda = -0.01;
  float coe[2*size-1];
  float coe_i =0.0;

  for (i=0; i < size; i++)
    {
      coe_i = 10*exp(lamda*i); 
      j=size-1+i;     
      coe[j]=coe_i;
      j=size-1-i;     
      coe[j]=coe_i;
    }

  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
	m[i*size+j]=coe[size-1-i+j];
      }
  }
}


int main(int argc, char *argv[]) {

  printf("WG size of kernel 1 = %d, WG size of kernel 2= %d X %d\n", BLOCK_SIZE_0, BLOCK_SIZE_1_X, BLOCK_SIZE_1_Y);
    float *a=NULL, *b=NULL, *finalVec=NULL;
    float *m=NULL;
    int size = -1;
    
    FILE *fp;
    
    // args
    char filename[200];
    int quiet=1,timing=0,platform=-1,device=-1, compare_result=0;
    
    // parse command line
    if (parseCommandline(argc, argv, filename,
			 &quiet, &timing, &platform, &device, &size, &compare_result)) {
    printUsage();
    return 0;
    }

    // CreateProfiler();
    // StartProfiling();

    context = cl_init_context(platform,device,quiet);
    
    if(size < 1)
    {
      fp = fopen(filename, "r");
      fscanf(fp, "%d", &size);
        
      a = (float *) malloc(size * size * sizeof(float));
      InitMat(fp,size, a, size, size);

      b = (float *) malloc(size * sizeof(float));
      InitAry(fp, b, size);

      fclose(fp);

    }
    else
    {
      printf("create input internally before create, size = %d \n", size);

      a = (float *) malloc(size * size * sizeof(float));
      create_matrix(a, size);

      b = (float *) malloc(size * sizeof(float));
      for (int i =0; i< size; i++)
        b[i]=1.0;

    }

    if (!quiet) {    
      printf("The input matrix a is:\n");
      PrintMat(a, size, size, size);

      printf("The input array b is:\n");
      PrintAry(b, size);
    }
 
    // create the solution matrix
    m = (float *) malloc(size * size * sizeof(float));
	 
    // create a new vector to hold the final answer

    finalVec = (float *) malloc(size * sizeof(float));
    
    InitPerRun(size,m);

    //begin timing	
        // printf("The result of array b is before run: \n");
        // PrintAry(b, size);
    
    // run kernels
	  ForwardSub(context,a,b,m,size,timing,compare_result);
        // printf("The result of array b is after run: \n");
        // PrintAry(b, size);
    
    //end timing
    if (!quiet) {
      printf("The result of matrix m is: \n");
      
      PrintMat(m, size, size, size);
      printf("The result of matrix a is: \n");
      PrintMat(a, size, size, size);
      printf("The result of array b is: \n");
      PrintAry(b, size);
      
      BackSub(a,b,finalVec,size);
      printf("The final solution is: \n");
      PrintAry(finalVec,size);
    }
    
    free(m);
    free(a);
    free(b);
    free(finalVec);
    cl_cleanup();

    // ReleaseProfiler();
  //OpenClGaussianElimination(context,timing);

  return 0;
}

/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
void ForwardSub(cl_context context, float *a, float *b, float *m, int size, int timing, int compare_result){
  float *a_copy, *b_copy, *m_copy;
  if(compare_result){
    // copy inputs to run original kernels
    a_copy = (float*)memalign(4096, sizeof(float)*size*size);
    b_copy = (float*)memalign(4096, sizeof(float)*size);
    m_copy = (float*)memalign(4096, sizeof(float)*size*size);

    memcpy(a_copy, a, sizeof(float)*size*size);
    memcpy(b_copy, b, sizeof(float)*size);
    memcpy(m_copy, m, sizeof(float)*size*size);

    // compareResults(size, m, a, b, m_copy, a_copy, b_copy);
  }

  // 1. set up kernels
  cl_kernel fan1_kernel,fan2_kernel;
  cl_int status=0;
  cl_program gaussianElim_program;
  cl_event writeEvent,kernelEvent,readEvent;
  float writeTime=0,readTime=0,kernelTime=0;
  float writeMB=0,readMB=0;
  
  char build_option[100];
  sprintf(build_option, "-I. -cl-std=CL2.0");
  gaussianElim_program = cl_compileProgram(
      (char *)"gaussianElim_kernels.cl", build_option);
  
  fan1_kernel = clCreateKernel(
      // gaussianElim_program, "Fan1", &status);
      gaussianElim_program, "Fan1_MBAP", &status);
  status = cl_errChk(status, (char *)"Error Creating Fan1_MBAP kernel",true);
  if(status)exit(1);
  
  fan2_kernel = clCreateKernel(
      // gaussianElim_program, "Fan2", &status);
      gaussianElim_program, "Fan2_MBAP", &status);
  status = cl_errChk(status, (char *)"Error Creating Fan2_MBAP kernel",true);
  if(status)exit(1);
  
  // 2. set up memory on device and send ipts data to device

  cl_mem a_dev, b_dev, m_dev;

  cl_int error=0;

  // a_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*size*size, NULL, &error);
  // b_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*size, NULL, &error);
  // m_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size * size, NULL, &error);

  a_dev = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float)*size*size, a, &error);
  b_dev = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float)*size, b, &error);
  m_dev = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float)*size*size, m, &error);

  cl_command_queue command_queue = cl_getCommandQueue();
  
  // error = clEnqueueWriteBuffer(command_queue,
  //            a_dev,
  //            1, // change to 0 for nonblocking write
  //            0, // offset
  //            sizeof(float)*size*size,
  //            a,
  //            0,
  //            NULL,
  //            &writeEvent);
  
  // if (timing) writeTime+=eventTime(writeEvent,command_queue);
  // clReleaseEvent(writeEvent);
  
  // error = clEnqueueWriteBuffer(command_queue,
  //            b_dev,
  //            1, // change to 0 for nonblocking write
  //            0, // offset
  //            sizeof(float)*size,
  //            b,
  //            0,
  //            NULL,
  //            &writeEvent);
  // if (timing) writeTime+=eventTime(writeEvent,command_queue);
  // clReleaseEvent(writeEvent);
              
  // error = clEnqueueWriteBuffer(command_queue,
  //            m_dev,
  //            1, // change to 0 for nonblocking write
  //            0, // offset
  //            sizeof(float)*size*size,
  //            m,
  //            0,
  //            NULL,
  //            &writeEvent);
  // if (timing) writeTime+=eventTime(writeEvent,command_queue);
  // clReleaseEvent(writeEvent);
  writeMB = (float)(sizeof(float) * size * (size + size + 1) / 1e6);

  // 3. Determine block sizes
  size_t globalWorksizeFan1[1];
  size_t globalWorksizeFan2[2];
  size_t localWorksizeFan1Buf[1]={BLOCK_SIZE_0};
  size_t localWorksizeFan2Buf[2]={BLOCK_SIZE_1_X, BLOCK_SIZE_1_Y};
  size_t *localWorksizeFan1=NULL;
  size_t *localWorksizeFan2=NULL;
  // size_t localWorksizeFan1[1];
  // size_t localWorksizeFan2[2];

  globalWorksizeFan1[0] = size;
  globalWorksizeFan2[0] = size;
  globalWorksizeFan2[1] = size;

  if(localWorksizeFan1Buf[0]){
          localWorksizeFan1=localWorksizeFan1Buf;
          // localWorksizeFan1[0] = localWorksizeFan1Buf;
          globalWorksizeFan1[0]=(int)ceil(globalWorksizeFan1[0]/(double)localWorksizeFan1Buf[0])*localWorksizeFan1Buf[0];
  }
  if(localWorksizeFan2Buf[0]){
          localWorksizeFan2=localWorksizeFan2Buf;
          globalWorksizeFan2[0]=(int)ceil(globalWorksizeFan2[0]/(double)localWorksizeFan2Buf[0])*localWorksizeFan2Buf[0];
          globalWorksizeFan2[1]=(int)ceil(globalWorksizeFan2[1]/(double)localWorksizeFan2Buf[1])*localWorksizeFan2Buf[1];
  }

	int t;
  size_t num_wgs_Fan1[1];
  size_t num_wgs_Fan2[2];
  size_t total_num_wgs_Fan1, total_num_wgs_Fan2;
  size_t gpu_partitioned_global_work_size_Fan1[1];
  size_t gpu_partitioned_global_work_size_Fan2[2];
  int n_tasks;

  num_wgs_Fan1[0] = globalWorksizeFan1[0] / localWorksizeFan1Buf[0];
  total_num_wgs_Fan1 = num_wgs_Fan1[0];
  gpu_partitioned_global_work_size_Fan1[0] = num_gpu_compute_units * localWorksizeFan1Buf[0];

  num_wgs_Fan2[0] = globalWorksizeFan2[0] / localWorksizeFan2Buf[0];
  num_wgs_Fan2[1] = globalWorksizeFan2[1] / localWorksizeFan2Buf[1];
  total_num_wgs_Fan2 = num_wgs_Fan2[0] * num_wgs_Fan2[1];

  gpu_partitioned_global_work_size_Fan2[0] = num_gpu_compute_units * localWorksizeFan2Buf[0];
  gpu_partitioned_global_work_size_Fan2[1] = globalWorksizeFan2[1];

	// 4. Setup and Run kernels
  printf("Running Kernels...\n");
	for (t=0; t<(size-1); t++) {
    // kernel args
    cl_int argchk;
    argchk  = clSetKernelArg(fan1_kernel, 0, sizeof(cl_mem), (void *)&m_dev);
    argchk |= clSetKernelArg(fan1_kernel, 1, sizeof(cl_mem), (void *)&a_dev);
    argchk |= clSetKernelArg(fan1_kernel, 2, sizeof(cl_mem), (void *)&b_dev);
    argchk |= clSetKernelArg(fan1_kernel, 3, sizeof(int), (void *)&size);
    argchk |= clSetKernelArg(fan1_kernel, 4, sizeof(int), (void *)&t);
    argchk |= clSetKernelArg(fan1_kernel, 5, sizeof(int), (void *)&throttling_mod);
    argchk |= clSetKernelArg(fan1_kernel, 6, sizeof(int), (void *)&throttling_alloc);
    cl_errChk(argchk,"ERROR in Setting Fan1 kernel args",true);
    
    std::atomic_int* worklist_Fan1 = (std::atomic_int*)clSVMAlloc(context,
        CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
        sizeof(std::atomic_int), 0);
    worklist_Fan1[0].store(0);
    error = clSetKernelArgSVMPointer(fan1_kernel, 7, worklist_Fan1);
    cl_errChk(argchk,"ERROR in Setting Fan1 SVM pointer",true);

    n_tasks = total_num_wgs_Fan1;
    error = clSetKernelArg(fan1_kernel, 8, sizeof(int), (void*)&n_tasks);
    cl_errChk(argchk,"ERROR in Setting Fan1 kernel args",true);

    // StopProfiling("prelude", "profiling.txt");
    // StartProfiling();

    // launch kernel
    error = clEnqueueNDRangeKernel(
              command_queue,  fan1_kernel, 1, 0,
              // globalWorksizeFan1,
              gpu_partitioned_global_work_size_Fan1,
              localWorksizeFan1,
              0, NULL, &kernelEvent);
    cl_errChk(error,"ERROR in Executing Fan1 Kernel",true);
    if (timing) {
        // printf("here1a\n");
        kernelTime+=eventTime(kernelEvent,command_queue);
        // printf("here1b\n");
    }
    clReleaseEvent(kernelEvent);
    //Fan1<<<dimGrid,dimBlock>>>(m_cuda,a_cuda,Size,t);
    //cudaThreadSynchronize();

    clFlush(command_queue);

    std::vector<std::thread*> threads_Fan1;

    for(int i = 0; i < num_cpu_cores; i++){
      std::thread* thread_ = new std::thread(Fan1_MBAP, i,
        m, a, b, size, t,
        globalWorksizeFan1, localWorksizeFan1,
        worklist_Fan1, n_tasks);
      threads_Fan1.push_back(thread_);
    }

    for(int i = 0; i <threads_Fan1.size(); i++){
      threads_Fan1[i]->join();
    }

    clFinish(command_queue);
    
    // kernel args
    argchk  = clSetKernelArg(fan2_kernel, 0, sizeof(cl_mem), (void *)&m_dev);
    argchk |= clSetKernelArg(fan2_kernel, 1, sizeof(cl_mem), (void *)&a_dev);
    argchk |= clSetKernelArg(fan2_kernel, 2, sizeof(cl_mem), (void *)&b_dev);
    argchk |= clSetKernelArg(fan2_kernel, 3, sizeof(int), (void *)&size);
    argchk |= clSetKernelArg(fan2_kernel, 4, sizeof(int), (void *)&t);
    argchk |= clSetKernelArg(fan2_kernel, 5, sizeof(int), (void *)&throttling_mod);
    argchk |= clSetKernelArg(fan2_kernel, 6, sizeof(int), (void *)&throttling_alloc);
    cl_errChk(argchk,"ERROR in Setting Fan2 kernel args",true);

    std::atomic_int* worklist_Fan2 = (std::atomic_int*)clSVMAlloc(context,
        CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
        sizeof(std::atomic_int), 0);
    worklist_Fan2[0].store(0);
    error = clSetKernelArgSVMPointer(fan2_kernel, 7, worklist_Fan2);
    cl_errChk(argchk,"ERROR in Setting Fan2 SVM pointer",true);

    n_tasks = total_num_wgs_Fan2;
    error = clSetKernelArg(fan2_kernel, 8, sizeof(int), (void*)&n_tasks);
    cl_errChk(argchk,"ERROR in Setting Fan2 kernel args",true);

    // StopProfiling("prelude", "profiling.txt");
    // StartProfiling();
    
    // launch kernel
    error = clEnqueueNDRangeKernel(
              command_queue,  fan2_kernel, 2, 0,
              // globalWorksizeFan2,NULL,
              gpu_partitioned_global_work_size_Fan2, localWorksizeFan2,
              0, NULL, &kernelEvent);
    cl_errChk(error,"ERROR in Executing Fan1 Kernel",true);
    if (timing) {
        // printf("here2a\n");
        kernelTime+=eventTime(kernelEvent,command_queue);
        // printf("here2b\n");
    }
    clReleaseEvent(kernelEvent);
    //Fan2<<<dimGridXY,dimBlockXY>>>(m_cuda,a_cuda,b_cuda,Size,Size-t,t);
    //cudaThreadSynchronize();
    clFlush(command_queue);

    std::vector<std::thread*> threads_Fan2;
    std::mutex mutx;
    for(int i = 0; i < num_cpu_cores; i++){
      std::thread* thread_ = new std::thread(Fan2_MBAP, i,
        m, a, b, size, t,
        globalWorksizeFan2, localWorksizeFan2,
        worklist_Fan2, n_tasks,
        std::ref(mutx));
      threads_Fan2.push_back(thread_);
    }

    for(int i = 0; i <threads_Fan2.size(); i++){
      threads_Fan2[i]->join();
    }

    clFinish(command_queue);
    // StopProfiling("gaussianElim", "profiling.txt");
	}
    // 5. transfer data off of device
    error = clEnqueueReadBuffer(command_queue,
        a_dev,
        1, // change to 0 for nonblocking write
        0, // offset
        sizeof(float) * size * size,
        a,
        0,
        NULL,
        &readEvent);

    cl_errChk(error,"ERROR with clEnqueueReadBuffer",true);
    if (timing) readTime+=eventTime(readEvent,command_queue);
    clReleaseEvent(readEvent);
    
    error = clEnqueueReadBuffer(command_queue,
        b_dev,
        1, // change to 0 for nonblocking write
        0, // offset
        sizeof(float) * size,
        b,
        0,
        NULL,
        &readEvent);
    cl_errChk(error,"ERROR with clEnqueueReadBuffer",true);
    if (timing) readTime+=eventTime(readEvent,command_queue);
    clReleaseEvent(readEvent);
    
    error = clEnqueueReadBuffer(command_queue,
        m_dev,
        1, // change to 0 for nonblocking write
        0, // offset
        sizeof(float) * size * size,
        m,
        0,
        NULL,
        &readEvent);

    cl_errChk(error,"ERROR with clEnqueueReadBuffer",true);
    if (timing) readTime+=eventTime(readEvent,command_queue);
    clReleaseEvent(readEvent);

    // run orginal kernels
    if(compare_result){
      cl_kernel fan1_kernel_original, fan2_kernel_original;

      fan1_kernel_original = clCreateKernel(gaussianElim_program, "Fan1", &status);
      status = cl_errChk(status, (char *)"Error Creating Fan1 kernel",true);
      if(status)exit(1);

      fan2_kernel_original = clCreateKernel(gaussianElim_program, "Fan2", &status);
      status = cl_errChk(status, (char *)"Error Creating Fan2 kernel",true);
      if(status)exit(1);

      cl_mem a_dev_o, b_dev_o, m_dev_o;
      a_dev_o = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float)*size*size, a_copy, &error);
      b_dev_o = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float)*size, b_copy, &error);
      m_dev_o = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float)*size*size, m_copy, &error);

      cl_command_queue command_queue_original = cl_getCommandQueue();

      size_t globalWSFan1[1] = { size };
      size_t globalWSFan2[2] = { size, size };
      size_t localWSFan1Buf[1] = { BLOCK_SIZE_0 };
      size_t localWSFan2Buf[2] = { BLOCK_SIZE_1_X, BLOCK_SIZE_1_Y };
      size_t *localWSFan1 = NULL;
      size_t *localWSFan2 = NULL;

      if(localWSFan1Buf[0]){
        localWSFan1=localWSFan1Buf;
        globalWSFan1[0]=(int)ceil(globalWSFan1[0]/(double)localWSFan1Buf[0])*localWSFan1Buf[0];
      }
      if(localWSFan2Buf[0]){
        localWSFan2=localWSFan2Buf;
        globalWSFan2[0]=(int)ceil(globalWSFan2[0]/(double)localWSFan2Buf[0])*localWSFan2Buf[0];
        globalWSFan2[1]=(int)ceil(globalWSFan2[1]/(double)localWSFan2Buf[1])*localWSFan2Buf[1];
      }

      for(int to = 0; to<(size-1); to++){
        // kernel args
        cl_int argchk_o;
        argchk_o  = clSetKernelArg(fan1_kernel_original, 0, sizeof(cl_mem), (void *)&m_dev_o);
        argchk_o |= clSetKernelArg(fan1_kernel_original, 1, sizeof(cl_mem), (void *)&a_dev_o);
        argchk_o |= clSetKernelArg(fan1_kernel_original, 2, sizeof(cl_mem), (void *)&b_dev_o);
        argchk_o |= clSetKernelArg(fan1_kernel_original, 3, sizeof(int), (void *)&size);
        argchk_o |= clSetKernelArg(fan1_kernel_original, 4, sizeof(int), (void *)&to);
    
        cl_errChk(argchk_o,"ERROR in Setting Fan1 kernel args", true);
        
        // launch kernel
        error = clEnqueueNDRangeKernel(
                  command_queue_original,  fan1_kernel_original, 1, 0,
                  globalWSFan1, localWSFan1,
                  0, NULL, &kernelEvent);

        cl_errChk(error,"ERROR in Executing Fan1 Kernel",true);
        clReleaseEvent(kernelEvent);
		
        // kernel args
        argchk_o  = clSetKernelArg(fan2_kernel_original, 0, sizeof(cl_mem), (void *)&m_dev_o);
        argchk_o |= clSetKernelArg(fan2_kernel_original, 1, sizeof(cl_mem), (void *)&a_dev_o);
        argchk_o |= clSetKernelArg(fan2_kernel_original, 2, sizeof(cl_mem), (void *)&b_dev_o);
        argchk_o |= clSetKernelArg(fan2_kernel_original, 3, sizeof(int), (void *)&size);
        argchk_o |= clSetKernelArg(fan2_kernel_original, 4, sizeof(int), (void *)&to);
        cl_errChk(argchk_o,"ERROR in Setting Fan2 kernel args",true);
        
        // launch kernel
        error = clEnqueueNDRangeKernel(
                  command_queue_original,  fan2_kernel_original, 2, 0,
                  globalWSFan2, NULL,
                  0, NULL, &kernelEvent);

        cl_errChk(error,"ERROR in Executing Fan1 Kernel",true);
        clReleaseEvent(kernelEvent);
      }

      error = clEnqueueReadBuffer(command_queue_original,
        a_dev_o,
        1, // change to 0 for nonblocking write
        0, // offset
        sizeof(float) * size * size,
        a_copy,
        0,
        NULL,
        &readEvent);
      
      error = clEnqueueReadBuffer(command_queue,
          b_dev_o,
          1, // change to 0 for nonblocking write
          0, // offset
          sizeof(float) * size,
          b_copy,
          0,
          NULL,
          &readEvent);
      
      error = clEnqueueReadBuffer(command_queue,
          m_dev_o,
          1, // change to 0 for nonblocking write
          0, // offset
          sizeof(float) * size * size,
          m_copy,
          0,
          NULL,
          &readEvent);

      // PrintAry(a_copy, size*size);
      // PrintAry(b_copy, size);
      // PrintAry(m_copy, size*size);

      // compareResults(size, m, a, b, m_copy, a_copy, b_copy);
      printf("\nComparing the result arrays: m, a, b\n");
      printf("Array m: ");
      compareResults(m, m_copy, size, "m");
      printf("Array a: ");
      compareResults(a, a_copy, size, "a");
      printf("Array b: ");
      compareResults(b, b_copy, size, "b");
      printf("\n");

      free(m_copy);
      free(a_copy);
      free(b_copy);
    }

    readMB = (float)(sizeof(float) * size * (size + size + 1) / 1e6);
    if (timing) {
        printf("Matrix Size\tWrite(s) [size]\t\tKernel(s)\tRead(s)  [size]\t\tTotal(s)\n");
        printf("%dx%d      \t",size,size);    
        printf("%f [%.2fMB]\t",writeTime,writeMB);
        printf("%f\t",kernelTime);
        printf("%f [%.2fMB]\t",readTime,readMB);
        printf("%f\n\n",writeTime+kernelTime+readTime);
   }
}

float eventTime(cl_event event,cl_command_queue command_queue){
    cl_int error=0;
    cl_ulong eventStart,eventEnd;
    clFinish(command_queue);
    error = clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,
                                    sizeof(cl_ulong),&eventStart,NULL);
    cl_errChk(error,"ERROR in Event Profiling.",true); 
    error = clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,
                                    sizeof(cl_ulong),&eventEnd,NULL);
    cl_errChk(error,"ERROR in Event Profiling.",true);

    return (float)((eventEnd-eventStart)/1e9);
}

 // Ke Wang add a function to generate input internally
int parseCommandline(int argc, char *argv[], char* filename,
                     int *q, int *t, int *p, int *d, int *size, int* c){
    // if (argc < 2) return 1; // error
    // strncpy(filename,argv[1],100);
    if(argc < 5) return 1;
    
    int input_num_cpu_cores = atoi(argv[1]);
    int input_throttling_mod = atoi(argv[2]);
    int input_throttling_alloc = atoi(argv[3]);

    if(input_num_cpu_cores<0 || input_throttling_mod<0 || input_throttling_alloc<0)
      return 1;
    
    num_cpu_cores = input_num_cpu_cores;
    throttling_mod = input_throttling_mod;
    input_throttling_alloc = throttling_alloc;

    int i;
    char flag;
    
    for(i=4;i<argc;i++) { 
      if (argv[i][0]=='-') {// flag
        flag = argv[i][1];
          switch (flag) {
            case 's': // platform
              i++;
              *size = atoi(argv[i]);
	      printf("Create matrix internally in parse, size = %d \n", *size);
              break;
            case 'f': // platform
              i++;
	      strncpy(filename,argv[i],100);
	      printf("Read file from %s \n", filename);
              break;
            case 'h': // help
              return 1;
              break;
            case 'q': // disable quiet mode
              *q = 0;
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
            case 'c': // comapre the result of MBAP kernels with the one of the orginal kernels
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
  printf("Gaussian Elimination Usage\n");
  printf("\n");
  printf("gaussianElimination [num_cpu_cores] [throttling_mod] [throttling_alloc] [filename] [-hqt] [-p [int] -d [int]]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./gaussianElimination 1 1 1 matrix4.txt\n");
  printf("\n");
  printf("num_cpu_cores     the number of cpu cores to use");
  printf("\n");
  printf("throttling_mod    throttling mod");
  printf("\n");
  printf("throttling_alloc  throttling allocation");
  printf("\n");
  printf("filename          the filename that holds the matrix data\n");
  printf("\n");
  printf("-h                Display the help file\n");
  printf("-q                Disable quiet mode. Show all text output.\n");
  printf("-t                Print timing information.\n");
  printf("-c                Comapre the result of MBAP kernels with the one of the orginal kernels.\n");
  printf("\n");
  printf("-p [int]          Choose the platform (must choose both platform and device)\n");
  printf("-d [int]          Choose the device (must choose both platform and device)\n");
  printf("\n");
  printf("\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
  printf("       2. If you declare either the device or the platform,\n");
  printf("          you must declare both.\n\n");
}


/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun(int size,float *m) 
{
	int i;
	for (i=0; i<size*size; i++)
			*(m+i) = 0.0;
}

void BackSub(float *a, float *b, float *finalVec, int size)
{
	// solve "bottom up"
	int i,j;
	for(i=0;i<size;i++){
		finalVec[size-i-1]=b[size-i-1];
		for(j=0;j<i;j++)
		{
			finalVec[size-i-1]-=*(a+size*(size-i-1)+(size-j-1)) * finalVec[size-j-1];
		}
		finalVec[size-i-1]=finalVec[size-i-1]/ *(a+size*(size-i-1)+(size-i-1));
	}
}

void InitMat(FILE *fp, int size, float *ary, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			fscanf(fp, "%f",  ary+size*i+j);
		}
	}  
}

/*------------------------------------------------------
 ** InitAry() -- Initialize the array (vector) by reading
 ** data from the data file
 **------------------------------------------------------
 */
void InitAry(FILE *fp, float *ary, int ary_size)
{
	int i;
	
	for (i=0; i<ary_size; i++) {
		fscanf(fp, "%f",  &ary[i]);
	}
}

/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(float *ary, int size, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			printf("%8.2e ", *(ary+size*i+j));
		}
		printf("\n");
	}
	printf("\n");
}

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *ary, int ary_size)
{
	int i;
	for (i=0; i<ary_size; i++) {
		printf("%.2e ", ary[i]);
	}
	printf("\n\n");
}

// Comapre the result of MBAP kernels with the one of the orginal kernels
void compareResults(int size, float *m_mbap, float *a_mbap, float *b_mbap,
                    float *m_original, float *a_original, float *b_original){
  printf("Comparing the result of MBAP kernels and the one of origianl kernels...\n");
  int count_m_diff = 0, count_a_diff = 0, count_b_diff = 0;
  FILE *fp = fopen("compareResult.txt", "w");

  int i;
  for(i=0; i<size*size; i++){
    if(*(m_mbap+i) != *(m_original+i)){
      count_m_diff++;
      fprintf(fp, "m %d: %f %f\n", i, *(m_mbap+i), *(m_original+i));
    }
  }
  if(count_m_diff)
    fprintf(fp, "\n");

  for(i=0; i<size*size; i++){
    if(*(a_mbap+i) != *(a_mbap+i)){
      count_a_diff++;
      fprintf(fp, "a %d: %f %f\n", i, *(a_mbap+i), *(a_original+i));
    }
  }
  if(count_a_diff)
    fprintf(fp, "\n");

  for(i=0; i<size; i++){
    if(*(b_mbap+i) != *(b_original+i)){
      count_b_diff++;
      fprintf(fp, "b %d: %f %f\n", i, *(b_mbap+i), *(b_original+i));
    }
  }
  if(count_b_diff)
    fprintf(fp, "\n");

  if(count_m_diff > 0 || count_a_diff > 0 || count_b_diff > 0){
    fclose(fp);
    fprintf(stderr, "Error: Result not match! (Difference - m: %d/%d, a: %d/%d, b: %d/%d)\n", 
      count_m_diff, size*size, count_a_diff, size*size, count_b_diff, size);
    fprintf(stderr, "Exit the program\n");
    exit(1);
  }
  
  printf("Result match!\n");
  fprintf(fp, "Result match!\n");
  fclose(fp);
  return;
}
#endif


#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <CL/cl.h>
#include "CL_helper.h"

////////////////////////////////
#include <thread>
#include <vector>
#include <atomic>
#include <malloc.h>
#include <mutex>

#ifdef PROFILER
#include "profiler_interface.h"
#endif
////////////////////////////////

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#define TOL      (0.001)
#define STR_SIZE (256)
#define MAX_PD   (3.0e6)

/* required precision in degrees	*/
#define PRECISION    0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI         100

/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

#define WG_SIZE_X (64)
#define WG_SIZE_Y (4)
float t_chip      = 0.0005;
float chip_height = 0.016;
float chip_width  = 0.016;
float amb_temp    = 80.0;

///////////////////////////////
/* MBAP parameters */
int num_cpu_cores = 1;
int num_gpu_compute_units = 8;
int throttling_mod = 1;
int throttling_alloc = 1;
//////////////////////////////

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05
#define SMALL_FLOAT_VAL 0.00000001f

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

void hotspotOpt_MBAP(int id,
                  float *p, float* tIn, float *tOut, float sdc,
                  int nx, int ny, int nz,
                  float ce, float cw, float cn, float cs,
                  float ct, float cb, float cc,
                  size_t* global_size,
                  size_t* local_size,
                  std::atomic_int* worklist,
                  int num_tasks,
                  std::mutex& mutx)
{
  int processed_wgs = 0;

  int _group_id[2];
  int _global_id[2];
  int global_id[2];

  float amb_temp = 80.0;
  int i, j, c, xy, W, E, N, S;
  float temp1, temp2, temp3;

  for (int wg_id = cpu_first(worklist);
            cpu_more(wg_id, num_tasks);
            wg_id = cpu_next(worklist)) {
    processed_wgs++;

    _group_id[0] = wg_id / (global_size[1] / local_size[1]);
    _group_id[1] = wg_id % (global_size[1] / local_size[1]);

    _global_id[0] = _group_id[0] * local_size[0];
    _global_id[1] = _group_id[1] * local_size[1];

    for (global_id[0] = _global_id[0]; global_id[0] < _global_id[0] + local_size[0]; global_id[0]++) {
      for (global_id[1] = _global_id[1]; global_id[1] < _global_id[1] + local_size[1]; global_id[1]++) {
          i = global_id[0];
          j = global_id[1];

          c = i + j * nx;
          xy = nx * ny;

          W = (i == 0)        ? c : c - 1;
          E = (i == nx-1)     ? c : c + 1;
          N = (j == 0)        ? c : c - nx;
          S = (j == ny-1)     ? c : c + nx;

          temp1 = temp2 = tIn[c];
          temp3 = tIn[c+xy];

          tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
            + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
          c += xy;
          W += xy;
          E += xy;
          N += xy;
          S += xy;

          for (int k = 1; k < nz-1; ++k) {
            temp1 = temp2;
            temp2 = temp3;
            temp3 = tIn[c+xy];

            tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
              + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
            c += xy;
            W += xy;
            E += xy;
            N += xy;
            S += xy;
          }

          temp1 = temp2;
          temp2 = temp3;

          tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
            + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
      }
    }
  }

  // printf("num_wgs_thread%d: %d\n", id, processed_wgs);
  return;
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

void usage(int argc, char **argv)
{
  fprintf(stderr, "Usage: %s <num_cpu_cores> <throttling_mod> <throttling_alloc> <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile> [<compare_result>]\n", argv[0]);
  fprintf(stderr, "\t<num_cpu_cores> - the number of cpu cores to use\n");
	fprintf(stderr, "\t<throttling_mod> - throttling mod\n");
	fprintf(stderr, "\t<throttling_alloc> - throttling allocation\n");
  fprintf(stderr, "\t<rows/cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t<layers>  - number of layers in the grid (positive integer)\n");
  fprintf(stderr, "\t<iteration> - number of iterations\n");
  fprintf(stderr, "\t<powerFile>  - name of the file containing the initial power values of each cell\n");
  fprintf(stderr, "\t<tempFile>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t<outputFile> - output file\n");
  fprintf(stderr, "\t<compare_result> - Optional. Run both of MBAP kernel and the orginal one, then check if the results are same.\n");
  exit(1);
}

// void compareKernelResults(float* mbap_result, float* original_result, int size){
//   printf("Compare the result of MBAP kernel with the one of original kernel...\n");

//   FILE *fp = fopen("compareDistance.txt", "w");
//   int err = 0;

//   for(int i=0; i<size; i++){
//     if(*(mbap_result+i) != *(original_result+i)){
//         fprintf(fp, "%d: %f %f\n", i, *(mbap_result+i), *(original_result+i));
//         err++;
//     }
//   }
  
//   if(err){ // The results are different
//     fprintf(stderr, "Error: Result not match! (Difference: %d/%d)\n", err, size);
//     fprintf(stderr, "Exit the program\n");
//     fclose(fp);
//     exit(1);
//   }
//   printf("Result match!\n");
//   fprintf(fp, "Result match!\n");
//   fclose(fp);
//   return; // The results are same
// }

int main(int argc, char** argv)
{
  if (argc != 10 && argc != 11)
  {
    usage(argc,argv);
  }
  if( (num_cpu_cores = atoi(argv[1])) < 0 ||
      (throttling_mod = atoi(argv[2])) < 0 ||
      (throttling_alloc = atoi(argv[3])) < 0 ){
    fprintf(stderr, "Invalid arguments\n");
    usage(argc, argv);
  }

  #ifdef PROFILER
  CreateProfiler();
  StartProfiling();
  #endif

  int numCols      = atoi(argv[4]);
  int numRows      = atoi(argv[4]);
  int layers       = atoi(argv[5]);
  int iterations   = atoi(argv[6]);
  char *pfile      = argv[7];
  char *tfile      = argv[8];
  char *ofile      = argv[9];
  int compare = (argc == 11) ? 1 : 0;



  /* calculating parameters*/

  float dx         = chip_height/numRows;
  float dy         = chip_width/numCols;
  float dz         = t_chip/layers;

  float Cap        = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
  float Rx         = dy / (2.0 * K_SI * t_chip * dx);
  float Ry         = dx / (2.0 * K_SI * t_chip * dy);
  float Rz         = dz / (K_SI * dx * dy);

  float max_slope  = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float dt         = PRECISION / max_slope;

  float ce, cw, cn, cs, ct, cb, cc;
  float stepDivCap = dt / Cap;
  ce               = cw                                              = stepDivCap/ Rx;
  cn               = cs                                              = stepDivCap/ Ry;
  ct               = cb                                              = stepDivCap/ Rz;

  cc               = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);


  int          err;           
  int size = numCols * numRows * layers;
  // float*        tIn      = (float*) calloc(size,sizeof(float));
  // float*        pIn      = (float*) calloc(size,sizeof(float));
  // float*        tempCopy = (float*)malloc(size * sizeof(float));
  // float*        tempOut  = (float*) calloc(size,sizeof(float));
  float *tIn = (float*)memalign(4096, sizeof(float)*size);
  float *pIn = (float*)memalign(4096, sizeof(float)*size);
  float *tempCopy = (float*)memalign(4096, sizeof(float)*size);
  float *tempOut = (float*)memalign(4096, sizeof(float)*size);
  // int count = size;
  readinput(tIn,numRows, numCols, layers, tfile);
  readinput(pIn,numRows, numCols, layers, pfile);

  size_t global[2];                   
  size_t local[2];
  memcpy(tempCopy,tIn, size * sizeof(float));

  float *tIn_o, *pIn_o, *tempOut_o;
  if(compare){
    tIn_o = (float*)memalign(4096, sizeof(float)*size);
    pIn_o = (float*)memalign(4096, sizeof(float)*size);
    tempOut_o = (float*)memalign(4096, sizeof(float)*size);

    memcpy(tIn_o, tIn, sizeof(float)*size);
    memcpy(pIn_o, pIn, sizeof(float)*size);
    memcpy(tempOut_o, tempOut, sizeof(float)*size);
  }

  cl_device_id     device_id;     
  cl_context       context;       
  cl_command_queue commands;      
  cl_program       program;       
  cl_kernel        ko_vadd;       

  cl_mem d_a;                     
  cl_mem d_b;                     
  cl_mem d_c;                     
  const char *KernelSource = load_kernel_source("hotspotKernel.cl"); 
  cl_uint numPlatforms;

  err = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (err != CL_SUCCESS || numPlatforms <= 0)
    {
      printf("Error: Failed to find a platform!\n%s\n",err_code(err));
      return EXIT_FAILURE;
    }

  cl_platform_id Platform[numPlatforms];
  err = clGetPlatformIDs(numPlatforms, Platform, NULL);
  if (err != CL_SUCCESS || numPlatforms <= 0)
    {
      printf("Error: Failed to get the platform!\n%s\n",err_code(err));
      return EXIT_FAILURE;
    }

  int i;
  for (i = 0; i < numPlatforms; i++)
    {
      err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
      if (err == CL_SUCCESS)
        {
          break;
        } 
    }

  if (device_id == NULL)
    {
      printf("Error: Failed to create a device group!\n%s\n",err_code(err));
      return EXIT_FAILURE;
    }

  err = output_device_info(device_id);

  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (!context)
    {
      printf("Error: Failed to create a compute context!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  commands = clCreateCommandQueue(context, device_id, 0, &err);
  if (!commands)
    {
      printf("Error: Failed to create a command commands!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
  if (!program)
    {
      printf("Error: Failed to create compute program!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  char clOptions[50];
  sprintf(clOptions, "-I. -cl-std=CL2.0");
  err = clBuildProgram(program, 0, NULL, clOptions, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      size_t len;
      char buffer[2048];

      printf("Error: Failed to build program executable!\n%s\n", err_code(err));
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
      printf("%s\n", buffer);
      return EXIT_FAILURE;
    }

  ko_vadd = clCreateKernel(program, "hotspotOpt1_MBAP", &err);
  if (!ko_vadd || err != CL_SUCCESS)
    {
      printf("Error: Failed to create compute kernel!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  // d_a  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * count, NULL, NULL);
  // d_b  = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(float) * count, NULL, NULL);
  // d_c  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * count, NULL, NULL);
  d_a  = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * size, tIn, NULL);
  d_b  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * size, pIn, NULL);
  d_c  = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * size, tempOut, NULL);

  if (!d_a || !d_b || !d_c) 
    {
      printf("Error: Failed to allocate device memory!\n");
      exit(1);
    }    

  // err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float) * count, tIn, 0, NULL, NULL);
  // if (err != CL_SUCCESS)
  //   {
  //     printf("Error: Failed to write tIn to source array!\n%s\n", err_code(err));
  //     exit(1);
  //   }

  // err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(float) * count, pIn, 0, NULL, NULL);
  // if (err != CL_SUCCESS)
  //   {
  //     printf("Error: Failed to write pIn to source array!\n%s\n", err_code(err));
  //     exit(1);
  //   }

  // err = clEnqueueWriteBuffer(commands, d_c, CL_TRUE, 0, sizeof(float) * count, tempOut, 0, NULL, NULL);
  // if (err != CL_SUCCESS)
  //   {
  //     printf("Error: Failed to write tempOut to source array!\n%s\n", err_code(err));
  //     exit(1);
  //   }

  size_t num_wgs[2];
  size_t total_num_wgs;
  size_t gpu_partitioned_global_work_size[2];
  std::atomic_int* worklist;
  int n_tasks;

  long long start = get_time();
  int j;

  #ifdef PROFILER
  StopProfiling("prelude", "profiling.txt");
  #endif

  for(j = 0; j < iterations; j++)
    {
      #ifdef PROFILER
      StartProfiling();
      #endif

      err  = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_b);
      err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_a);
      err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
      err |= clSetKernelArg(ko_vadd, 3, sizeof(float), &stepDivCap);
      err |= clSetKernelArg(ko_vadd, 4, sizeof(int), &numCols);
      err |= clSetKernelArg(ko_vadd, 5, sizeof(int), &numRows);
      err |= clSetKernelArg(ko_vadd, 6, sizeof(int), &layers);
      err |= clSetKernelArg(ko_vadd, 7, sizeof(float), &ce);
      err |= clSetKernelArg(ko_vadd, 8, sizeof(float), &cw);
      err |= clSetKernelArg(ko_vadd, 9, sizeof(float), &cn);
      err |= clSetKernelArg(ko_vadd, 10, sizeof(float), &cs);
      err |= clSetKernelArg(ko_vadd, 11, sizeof(float), &ct);
      err |= clSetKernelArg(ko_vadd, 12, sizeof(float), &cb);      
      err |= clSetKernelArg(ko_vadd, 13, sizeof(float), &cc);

      err |= clSetKernelArg(ko_vadd, 14, sizeof(int), (void *)&throttling_mod);
      err |= clSetKernelArg(ko_vadd, 15, sizeof(int), (void *)&throttling_alloc);
      
      global[0] = numCols;
      global[1] = numRows;

      local[0] = WG_SIZE_X;
      local[1] = WG_SIZE_Y;

      num_wgs[0] = global[0] /local[0];
      num_wgs[1] = global[1] / local[1];
      total_num_wgs = num_wgs[0] * num_wgs[1];

      gpu_partitioned_global_work_size[0] = num_gpu_compute_units * local[0];
      gpu_partitioned_global_work_size[1] = global[1];

      worklist = (std::atomic_int*)clSVMAlloc(context,
          CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
          sizeof(std::atomic_int), 0);
      worklist[0].store(0);
      err |= clSetKernelArgSVMPointer(ko_vadd, 16, worklist);

      n_tasks = total_num_wgs;
      err |= clSetKernelArg(ko_vadd, 17, sizeof(int), (void*)&n_tasks);

      if (err != CL_SUCCESS)
      {
        printf("Error: Failed to set kernel arguments!\n");
        exit(1);
      }

      #ifdef PROFILER
      StopProfiling("preparing kernel iteration", "porfiling.txt");
      StartProfiling();
      #endif

      err = clEnqueueNDRangeKernel(commands, ko_vadd, 2, NULL, global, local, 0, NULL, NULL);
      if (err)
      {
        printf("Error: Failed to execute kernel!\n%s\n", err_code(err));
        return EXIT_FAILURE;
      }

      clFlush(commands);

      std::vector<std::thread*> threads;
      std::mutex mutx;
      for (int i = 0; i < num_cpu_cores; i++) {
			  std::thread* thread_ = new std::thread(hotspotOpt_MBAP, i,
              pIn, tIn, tempOut, stepDivCap,
              numCols, numRows, layers,
              ce, cw, cn, cs, ct,cb, cc,
              global, local, worklist, n_tasks,
              std::ref(mutx));
        threads.push_back(thread_);
      }

      for (int i = 0; i < threads.size(); i++) {
        threads[i]->join();
      }

      clFinish(commands);

      #ifdef PROFILER
      StopProfiling("hotspot3D", "profiling.txt");
      #endif

      cl_mem temp = d_a;
      d_a         = d_c;
      d_c         = temp;
    }

  clFinish(commands);
  long long stop = get_time();
  // err = clEnqueueReadBuffer( commands, d_c, CL_TRUE, 0, sizeof(float) * count, tempOut, 0, NULL, NULL );  
  // if (err != CL_SUCCESS)
  //   {
  //     printf("Error: Failed to read output array!\n%s\n", err_code(err));
  //     exit(1);
  //   }

  // create orginal kernel and run to compare the result
  if(compare){
    cl_kernel kernel = clCreateKernel(program, "hotspotOpt1", &err);
    cl_command_queue commands_o = clCreateCommandQueue(context, device_id, 0, &err);

    cl_int error_o;
    cl_mem d_a_o  = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float)*size, tIn_o, &error_o);
    cl_mem d_b_o  = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float)*size, pIn_o, &error_o);
    cl_mem d_c_o  = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float)*size, tempOut_o, &error_o);
    if(error_o != CL_SUCCESS){
      fprintf(stderr, "Error: failed to create buffers for original kernel!\n");
      exit(1);
    }

    cl_mem temp_o;

    for(int k = 0; k < iterations; k++)
    {
      error_o  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_b_o);
      error_o |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a_o);
      error_o |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c_o);
      error_o |= clSetKernelArg(kernel, 3, sizeof(float), &stepDivCap);
      error_o |= clSetKernelArg(kernel, 4, sizeof(int), &numCols);
      error_o |= clSetKernelArg(kernel, 5, sizeof(int), &numRows);
      error_o |= clSetKernelArg(kernel, 6, sizeof(int), &layers);
      error_o |= clSetKernelArg(kernel, 7, sizeof(float), &ce);
      error_o |= clSetKernelArg(kernel, 8, sizeof(float), &cw);
      error_o |= clSetKernelArg(kernel, 9, sizeof(float), &cn);
      error_o |= clSetKernelArg(kernel, 10, sizeof(float), &cs);
      error_o |= clSetKernelArg(kernel, 11, sizeof(float), &ct);
      error_o |= clSetKernelArg(kernel, 12, sizeof(float), &cb);      
      error_o |= clSetKernelArg(kernel, 13, sizeof(float), &cc);
      if (error_o != CL_SUCCESS)
      {
        fprintf(stderr, "Error: failed to set kernel arguments for original kernel!\n");
        exit(1);
      }
      
      global[0] = numCols;
      global[1] = numRows;

      local[0] = WG_SIZE_X;
      local[1] = WG_SIZE_Y;

      err = clEnqueueNDRangeKernel(commands_o, kernel, 2, NULL, global, local, 0, NULL, NULL);
      if (err)
        {
          printf("Error: Failed to execute kernel!\n%s\n", err_code(err));
          return EXIT_FAILURE;
        }

      temp_o = d_a_o;
      d_a_o = d_c_o;
      d_c_o = temp_o;
    }

    clFinish(commands_o);

    // err = clEnqueueReadBuffer(commands_o, d_c_o, CL_TRUE, 0, sizeof(float)*size, tempOut_o, 0, NULL, NULL );  

    // compareKernelResults(tempOut, tempOut_o, size);
    compareResults(tempOut, tempOut_o, size, "tempOut");

    clReleaseMemObject(d_a_o);
    clReleaseMemObject(d_b_o);
    clReleaseMemObject(d_c_o);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands_o);
  }

  float* answer = (float*)calloc(size, sizeof(float));
  computeTempCPU(pIn, tempCopy, answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp, iterations);

  float acc = accuracy(tempOut,answer,numRows*numCols*layers);
  float time = (float)((stop - start)/(1000.0 * 1000.0));
  printf("Time: %.3f (s)\n",time);
  printf("Accuracy: %e\n",acc);

  writeoutput(tempOut,numRows,numCols,layers,ofile);
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseProgram(program);
  clReleaseKernel(ko_vadd);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  #ifdef PROFILER
  ReleaseProfiler();
  #endif

  return 0;
}

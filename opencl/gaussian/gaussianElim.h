#ifndef _GAUSSIANELIM
#define _GAUSSIANELIM

#include <iostream>
#include <vector>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include "clutils.h"

////////////////////////////////
#include <thread>
#include <vector>
#include <atomic>
#include <string.h>
#include <malloc.h>
#include <mutex>

// #include "profiler_interface.h"
///////////////////////////////

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif 

float *OpenClGaussianElimination(
	cl_context context,
	int timing);

void printUsage();
int parseCommandline(int argc, char *argv[], char* filename,
                     int *m, int *t, int *p, int *d, int *size, int* c);
                     
void InitPerRun(int size,float *m);
void ForwardSub(cl_context context, float *a, float *b, float *m, int size, int timing, int comapre_result);
void BackSub(float *a, float *b, float *finalVec, int size);
void Fan1(float *m, float *a, int Size, int t);
void Fan2(float *m, float *a, float *b,int Size, int j1, int t);
//void Fan3(float *m, float *b, int Size, int t);
void InitMat(FILE *fp, int size, float *ary, int nrow, int ncol);
void InitAry(FILE *fp, float *ary, int ary_size);
void PrintMat(float *ary, int size, int nrow, int ncolumn);
void PrintAry(float *ary, int ary_size);
float eventTime(cl_event event,cl_command_queue command_queue);

void compareResults(int size, float *m_mbap, float *a_mbap, float *b_mbap,
                    float *m_original, float *a_original, float *b_original);

#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#define IN_RANGE(x, min, max) ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

unsigned long x86_kernel_params[128] = { 0, };

int num_total_cores = 4;
int num_total_params = 16;

void dynproc_kernel (int iteration,
                    int* gpuWall,
                    int* gpuSrc,
                    int* gpuResults,
                    int cols,
                    int rows,
                    int startStep,
                    int border,
                    int HALO,
                    int* prev,
                    int* result,
                    int* outputBuffer,
                    size_t* global_size,
                    size_t* local_size,
                    size_t start_wg_id,
                    size_t num_wgs)
{
	for (size_t wg_id = start_wg_id; wg_id < start_wg_id + num_wgs; wg_id++) {
        for (size_t global_id = wg_id * local_size[0];
                global_id < wg_id * local_size[0] + local_size[0];
                global_id++) {
            int BLOCK_SIZE = get_local_size(0);
            int bx = get_group_id(0);
            int tx = get_local_id(0);

            // Each block finally computes result for a small block
            // after N iterations.
            // it is the non-overlapping small blocks that cover
            // all the input data

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
            
            barrier(CLK_LOCAL_MEM_FENCE);

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

                barrier(CLK_LOCAL_MEM_FENCE);

                if(i==iteration-1)
                {
                    // we are on the last iteration, and thus don't need to 
                    // compute for the next step.
                    break;
                }

                if(computed)
                {
                    //Assign the computation range
                    prev[tx] = result[tx];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            // update the global memory
            // after the last iteration, only threads coordinated within the
            // small block perform the calculation and switch on "computed"
            if (computed)
            {
                gpuResults[xidx] = result[tx];
            }
        }
    }
}

extern "C" {
    int CPU_thread_entry(int id) {
        int iteration = (int)x86_kernel_params[num_total_params * id + 0];
        int* gpuWall = (int*)x86_kernel_params[num_total_params * id + 1];
        int* gpuSrc = (int*)x86_kernel_params[num_total_params * id + 2];
        int* gpuResults = (int*)x86_kernel_params[num_total_params * id + 3];
        int cols = (int)x86_kernel_params[num_total_params * id + 4];
        int rows = (int)x86_kernel_params[num_total_params * id + 5];
        int startStep = (int)x86_kernel_params[num_total_params * id + 6];
        int border = (int)x86_kernel_params[num_total_params * id + 7];
        int HALO = (int)x86_kernel_params[num_total_params * id + 8];
        int* prev = (int*)x86_kernel_params[num_total_params * id + 9];
        int* result = (int*)x86_kernel_params[num_total_params * id + 10];
        int* outputBuffer = (int*)x86_kernel_params[num_total_params * id + 11];
        size_t* global_size = (size_t*)x86_kernel_params[num_total_params * id + 12];
        size_t* local_size = (size_t*)x86_kernel_params[num_total_params * id + 13];
        size_t start_wg_id = (size_t)x86_kernel_params[num_total_params * id + 14];
        size_t num_wgs = (size_t)x86_kernel_params[num_total_params * id + 15];

        dynproc_kernel(iteration, gpuWall, gpuSrc, gpuResults, cols, rows,
            startStep, border, HALO, prev, result, outputBuffer,
            global_size, local_size, start_wg_id, num_wgs);

        return 0;
    }
}
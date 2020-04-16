#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

unsigned long x86_kernel_params[128] = { 0, };

int num_total_cores = 4;
int num_total_params = 18;

typedef float DATA_TYPE;

void hotspotOpt1(DATA_TYPE *p, DATA_TYPE *tIn, DATA_TYPE *tOut, float sdc,
                int nx, int ny, int nz,
                float ce, float cw, 
                float cn, float cs,
                float ct, float cb, 
                float cc,
                size_t* global_size, size_t* local_size,
                size_t start_wg_id, size_t num_wgs)
{
    int _group_id[2];
    int _global_id[2];

    for (size_t wg_id = start_wg_id; wg_id < start_wg_id+num_wgs; wg_id++) {
        _group_id[0] = wg_id / (global_size[1] / local_size[1]);
        _group_id[1] = wg_id % (global_size[1] / local_size[1]);

        _global_id[0] = _group_id[0] * local_size[0];
        _global_id[1] = _group_id[1] * local_size[1];

        int global_id[2];
        for (global_id[0] = _global_id[0]; global_id[0] < _global_id[0] + local_size[0]; global_id[0]++) {
            for (global_id[1] = _global_id[1]; global_id[1] < _global_id[1] + local_size[1]; global_id[1]++) {
                int i = global_id[0]; // get_global_id(0);
                int j = global_id[1]; // get_global_id(0);
                float amb_temp = 80.0;
                int c = i + j * nx;
                int xy = nx * ny;

                int W = (i == 0)        ? c : c - 1;
                int E = (i == nx-1)     ? c : c + 1;
                int N = (j == 0)        ? c : c - nx;
                int S = (j == ny-1)     ? c : c + nx;

                float temp1, temp2, temp3;
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
}

extern "C" {
    int CPU_thread_entry(int id) {
        DATA_TYPE *p = (DATA_TYPE*)x86_kernel_params[num_total_params * id + 0];
        DATA_TYPE *tIn = (DATA_TYPE*)x86_kernel_params[num_total_params * id + 1];
        DATA_TYPE *tOut = (DATA_TYPE*)x86_kernel_params[num_total_params * id + 2];
        float sdc = (float)x86_kernel_params[num_total_params * id + 3];
        int nx = (int)x86_kernel_params[num_total_params * id + 4];
        int ny = (int)x86_kernel_params[num_total_params * id + 5];
        int nz = (int)x86_kernel_params[num_total_params * id + 6];
        float ce = (float)x86_kernel_params[num_total_params * id + 7];
        float cw = (float)x86_kernel_params[num_total_params * id + 8];
        float cn = (float)x86_kernel_params[num_total_params * id + 9];
        float cs = (float)x86_kernel_params[num_total_params * id + 10];
        float ct = (float)x86_kernel_params[num_total_params * id + 11];
        float cb = (float)x86_kernel_params[num_total_params * id + 12];
        float cc = (float)x86_kernel_params[num_total_params * id + 13];
        size_t* global_size = (size_t*)x86_kernel_params[num_total_params * id + 14];
        size_t* local_size = (size_t*)x86_kernel_params[num_total_params * id + 15];
        size_t start_wg_id = (size_t)x86_kernel_params[num_total_params * id + 16];
        size_t num_wgs = (size_t)x86_kernel_params[num_total_params * id + 17];

        hotspotOpt1(p, tIn, tOut, sdc, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc,
                    global_size, local_size, start_wg_id, num_wgs);

        return 0;
    }
}
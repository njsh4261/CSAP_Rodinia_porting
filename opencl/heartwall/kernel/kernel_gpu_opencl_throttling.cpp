#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include <math.h>

unsigned long x86_kernel_params[128] = { 0, };

int num_total_cores = 4;
int num_total_params = 10;

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void heartwall_kernel(DATA_TYPE* a, DATA_TYPE* r, DATA_TYPE* q, int k, int m, int n,
        size_t* global_size, size_t* local_size,
        size_t start_wg_id, size_t num_wgs)
{
    for (size_t wg_id = start_wg_id; wg_id < start_wg_id + num_wgs; wg_id++) {
        for (size_t global_id = wg_id * local_size[0];
                global_id < wg_id * local_size[0] + local_size[0];
                global_id++) {
            int tid = (int)global_id;

            if (tid == 0)
            {
                DATA_TYPE nrm = 0.0;
                int i;
                for (i = 0; i < m; i++)
                {
                    nrm += a[i * n + k] * a[i * n + k];
                }

                r[k * n + k] = sqrt(nrm);
            }
        }
    }
}

extern "C" {
int CPU_thread_entry(int id) {
    DATA_TYPE* a = (DATA_TYPE*)x86_kernel_params[num_total_params * id + 0];
    DATA_TYPE* r = (DATA_TYPE*)x86_kernel_params[num_total_params * id + 1];
    DATA_TYPE* q = (DATA_TYPE*)x86_kernel_params[num_total_params * id + 2];
    int* k = (int*)x86_kernel_params[num_total_params * id + 3];
    int* m = (int*)x86_kernel_params[num_total_params * id + 4];
    int* n = (int*)x86_kernel_params[num_total_params * id + 5];
    size_t* global_size = (size_t*)x86_kernel_params[num_total_params * id + 6];
    size_t* local_size = (size_t*)x86_kernel_params[num_total_params * id + 7];
    size_t start_wg_id = (size_t)x86_kernel_params[num_total_params * id + 8];
    size_t num_wgs = (size_t)x86_kernel_params[num_total_params * id + 9];

    gramschmidt_kernel1(a, r, q, *k, *m, *n, global_size, local_size, start_wg_id, num_wgs);

    return 0;
}
}
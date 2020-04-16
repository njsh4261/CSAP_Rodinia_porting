#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include <atomic>

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable

typedef struct{
	int starting;
	int no_of_edges;
} Node;

unsigned long x86_kernel_params[128] = { 0, };

int num_total_cores = 4;
int num_total_params = 9;

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

void BFS_2( char* g_graph_mask, 
			char* g_updating_graph_mask, 
			char* g_graph_visited, 
			char* g_over,
			int no_of_nodes,
            size_t* global_size,
            size_t* local_size,
            std::atomic_int* worklist,
            size_t num_tasks)
{
    for (size_t wg_id = cpu_first(worklist); cpu_more(wg_id, num_tasks); wg_id = cpu_next(worklist)) {
        for (size_t global_id = wg_id * local_size[0]; global_id < wg_id * local_size[0] + local_size[0]; global_id++) {
            int tid = get_global_id(0);
            if( tid<no_of_nodes && g_updating_graph_mask[tid]){

                g_graph_mask[tid]=true;
                g_graph_visited[tid]=true;
                *g_over=true;
                g_updating_graph_mask[tid]=false;
            }
        }
    }
}


extern "C" {
    int CPU_thread_entry(int id) {
        char* g_graph_mask = (char*)x86_kernel_params[num_total_params * id + 0];
        char* g_updating_graph_mask = (char*)x86_kernel_params[num_total_params * id + 1];
        char* g_graph_visited = (char*)x86_kernel_params[num_total_params * id + 2];
        char* g_over = (char*)x86_kernel_params[num_total_params * id + 3];
        int no_of_nodes = (int)x86_kernel_params[num_total_params * id + 4];
        size_t* global_size = (size_t*)x86_kernel_params[num_total_params * id + 5];
        size_t* local_size = (size_t*)x86_kernel_params[num_total_params * id + 6];
        std::atomic_int* worklist = (std::atomic_int*)x86_kernel_params[num_total_params * id + 7];
        size_t num_tasks = (size_t)x86_kernel_params[num_total_params * id + 8];

        BFS_2(g_graph_mask, g_updating_graph_mask, g_graph_visited, g_over, no_of_nodes,
            global_size, local_size, worklist, num_tasks);

        return 0;
    }
}

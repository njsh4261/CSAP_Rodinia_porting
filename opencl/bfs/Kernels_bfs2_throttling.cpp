#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable

typedef struct{
	int starting;
	int no_of_edges;
} Node;

unsigned long x86_kernel_params[128] = { 0, };

int num_total_cores = 4;
int num_total_params = 9;

void BFS_2( char* g_graph_mask, 
			char* g_updating_graph_mask, 
			char* g_graph_visited, 
			char* g_over,
			int no_of_nodes,
            size_t* global_size,
            size_t* local_size,
            size_t start_wg_id,
            size_t num_wgs )
{
    for (size_t wg_id = start_wg_id; wg_id < start_wg_id + num_wgs; wg_id++) {
        for (size_t global_id = wg_id * local_size[0];
                global_id < wg_id * local_size[0] + local_size[0];
                global_id++) {
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
        size_t start_wg_id = (size_t)x86_kernel_params[num_total_params * id + 7];
        size_t num_wgs = (size_t)x86_kernel_params[num_total_params * id + 8];

        BFS_2(g_graph_mask, g_updating_graph_mask, g_graph_visited, g_over, no_of_nodes,
            global_size, local_size, start_wg_id, num_wgs);

        return 0;
    }
}
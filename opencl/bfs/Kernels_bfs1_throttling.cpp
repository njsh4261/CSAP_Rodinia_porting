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
int num_total_params = 11;

void BFS_1( const Node* g_graph_nodes,
            const int* g_graph_edges, 
            char* g_graph_mask, 
            char* g_updating_graph_mask, 
            char* g_graph_visited, 
            int* g_cost, 
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
            if( tid<no_of_nodes && g_graph_mask[tid]){
                g_graph_mask[tid]=false;
                for(int i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++){
                    int id = g_graph_edges[i];
                    if(!g_graph_visited[id]){
                        g_cost[id]=g_cost[tid]+1;
                        g_updating_graph_mask[id]=true;
                        }
                    }
            }	
        }
    }
}

extern "C" {
    int CPU_thread_entry(int id) {
        Node* g_graph_nodes = (Node*)x86_kernel_params[num_total_params * id + 0];
        int* g_graph_edges = (int*)x86_kernel_params[num_total_params * id + 1];
        char* g_graph_mask = (char*)x86_kernel_params[num_total_params * id + 2];
        char* g_updating_graph_mask = (char*)x86_kernel_params[num_total_params * id + 3];
        char* g_graph_visited = (char*)x86_kernel_params[num_total_params * id + 4];
        int* g_cost = (int*)x86_kernel_params[num_total_params * id + 5];
        int no_of_nodes = (int)x86_kernel_params[num_total_params * id + 6];
        size_t* global_size = (size_t*)x86_kernel_params[num_total_params * id + 7];
        size_t* local_size = (size_t*)x86_kernel_params[num_total_params * id + 8];
        size_t start_wg_id = (size_t)x86_kernel_params[num_total_params * id + 9];
        size_t num_wgs = (size_t)x86_kernel_params[num_total_params * id + 10];

        BFS_1(g_graph_nodes, g_graph_edges, g_graph_mask, g_updating_graph_mask, g_graph_visited,
            g_cost, no_of_nodes, global_size, local_size, start_wg_id, num_wgs);

        return 0;
    }
}
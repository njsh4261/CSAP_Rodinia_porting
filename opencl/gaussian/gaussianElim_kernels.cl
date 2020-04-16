//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

inline int gpu_first(__local int* tmp, __global atomic_int* worklist)
{
    if (get_local_id(2) == 0 && get_local_id(1) == 0 && get_local_id(0) == 0) {
        tmp[0] = atomic_fetch_add(worklist, 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int current = tmp[0];
    return current;
}

inline int gpu_next(__local int* tmp, __global atomic_int* worklist)
{
    if (get_local_id(2) == 0 && get_local_id(1) == 0 && get_local_id(0) == 0) {
        tmp[0] = atomic_fetch_add(worklist, 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int current = tmp[0];
    return current;
}

inline bool gpu_more(int current, int n_tasks)
{
    return (current < n_tasks);
}

__kernel void Fan1(__global float *m_dev,
                  __global float *a_dev,
                  __global float *b_dev,
                  const int size,
                  const int t) {
    int globalId = get_global_id(0);
                              
    if (globalId < size-1-t) {
         *(m_dev + size * (globalId + t + 1)+t) = *(a_dev + size * (globalId + t + 1) + t) / *(a_dev + size * t + t);    
    }
}

__kernel void Fan1_MBAP(__global float *m_dev,
                        __global float *a_dev,
                        __global float *b_dev,
                        const int size,
                        const int t,
                        int throttling_mod,
                        int throttling_alloc,
                        __global atomic_int* worklist,
                        int n_tasks) {
    __local int tmp_[1];
    tmp_[0] = atomic_load(worklist);

    if (get_local_id(0) % throttling_mod < throttling_alloc) {
        for (int wg_id = gpu_first(tmp_, worklist);
                 gpu_more(wg_id, n_tasks);
                 wg_id = gpu_next(tmp_, worklist)) {
            int group_id_0 = wg_id;

            __local int local_worklist[1];
            local_worklist[0] = 0;

            for (int dynamic_work = atomic_add(local_worklist, 1);
                     dynamic_work < get_local_size(0);
                     dynamic_work = atomic_add(local_worklist, 1)) {
                int global_id_0 = group_id_0 * get_local_size(0) + dynamic_work;
                int globalId = global_id_0;
                if (globalId < size-1-t) {
                    *(m_dev + size * (globalId + t + 1)+t) = *(a_dev + size * (globalId + t + 1) + t) / *(a_dev + size * t + t);    
                }
            }
        }
    }
}


__kernel void Fan2(__global float *m_dev,
                  __global float *a_dev,
                  __global float *b_dev,
                  const int size,
                  const int t) {
	 int globalId = get_global_id(0);
	 
	 int globalIdx = get_global_id(0);
	 int globalIdy = get_global_id(1);
      if (globalIdx < size-1-t && globalIdy < size-t) {
         a_dev[size*(globalIdx+1+t)+(globalIdy+t)] -= m_dev[size*(globalIdx+1+t)+t] * a_dev[size*t+(globalIdy+t)];
 	 
 	    if(globalIdy == 0){
 		   b_dev[globalIdx+1+t] -= m_dev[size*(globalIdx+1+t)+(globalIdy+t)] * b_dev[t];
 	    }
 	 }
//   One dimensional
// 	 int globalIdx = globalId % size;
// 	 int globalIdy = globalId / size;
// 	 
// 	 if (globalIdx < size-1-t && globalIdy < size-t) {
//          a_dev[size*(globalIdx+1+t)+(globalIdy+t)] -= m_dev[size*(globalIdx+1+t)+t] * a_dev[size*t+(globalIdy+t)];
// 	 }
// 	 if(globalIdy == 0){
//  		   b_dev[globalIdx+1+t] -= m_dev[size*(globalIdx+1+t)+(globalIdy+t)] * b_dev[t];
//      }
    
}

__kernel void Fan2_MBAP(__global float *m_dev,
                        __global float *a_dev,
                        __global float *b_dev,
                        const int size,
                        const int t,
                        int throttling_mod,
                        int throttling_alloc,
                        __global atomic_int* worklist,
                        int n_tasks) {
    __local int tmp[1];
    tmp[0] = atomic_load(worklist);

    if (get_local_id(0) % throttling_mod < throttling_alloc) {
        for (int wg_id = gpu_first(tmp, worklist);
                 gpu_more(wg_id, n_tasks);
                 wg_id = gpu_next(tmp, worklist)) {
            int group_id_0 = wg_id / (get_global_size(1)/get_local_size(1));
            int group_id_1 = wg_id % (get_global_size(1)/get_local_size(1));

            __local int local_worklist[1];
            local_worklist[0] = 0;

            for (int dynamic_work = atomic_add(local_worklist, 1);
                     dynamic_work < get_local_size(0) * get_local_size(1);
                     dynamic_work = atomic_add(local_worklist, 1)) {
                int global_id_0 = group_id_0 * get_local_size(0) +
                                  dynamic_work / get_local_size(1);
                int global_id_1 = group_id_1 * get_local_size(1) +
                                  dynamic_work % get_local_size(1);
                int globalId = global_id_0;
                int globalIdx = global_id_0;
                int globalIdy = global_id_1;
                if (globalIdx < size-1-t && globalIdy < size-t) {
                    a_dev[size*(globalIdx+1+t)+(globalIdy+t)] -= m_dev[size*(globalIdx+1+t)+t] * a_dev[size*t+(globalIdy+t)];
                    if(globalIdy == 0){
                        b_dev[globalIdx+1+t] -= m_dev[size*(globalIdx+1+t)+(globalIdy+t)] * b_dev[t];
                    }
                }
            }
        }
    }
}
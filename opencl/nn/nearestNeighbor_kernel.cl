//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

typedef struct latLong
    {
        float lat;
        float lng;
    } LatLong;

__kernel void NearestNeighbor(__global LatLong *d_locations,
							  __global float *d_distances,
							  const int numRecords,
							  const float lat,
							  const float lng) {
	 int globalId = get_global_id(0);
							  
     if (globalId < numRecords) {
         __global LatLong *latLong = d_locations+globalId;
    
         __global float *dist=d_distances+globalId;
         *dist = (float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
	 }
}



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

__kernel void NearestNeighbor_MBAP( __global LatLong *d_locations,
                                    __global float *d_distances,
                                    const int numRecords,
                                    const float lat,
                                    const float lng,
                                    int throttling_mod,
                                    int throttling_alloc,
                                    __global atomic_int* worklist,
                                    int n_tasks) {
    __local int tmp_[1];
    __local int local_worklist[1];
    tmp_[0] = atomic_load(worklist);

    if (get_local_id(0) % throttling_mod < throttling_alloc) {
        for (int wg_id = gpu_first(tmp_, worklist);
                 gpu_more(wg_id, n_tasks);
                 wg_id = gpu_next(tmp_, worklist)) {
            int group_id_0 = wg_id;

            local_worklist[0] = 0;

            for (int dynamic_work = atomic_add(local_worklist, 1);
                     dynamic_work < get_local_size(0);
                     dynamic_work = atomic_add(local_worklist, 1)) {
                int global_id_0 = group_id_0 * get_local_size(0) + dynamic_work;
                int globalId = global_id_0;
                if (globalId < numRecords) {
                    __global LatLong *latLong = d_locations+globalId;
                
                    __global float *dist=d_distances+globalId;
                    *dist = (float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
                }
            }
        }
    }
}
// #pragma OPENCL EXTENSION cl_amd_fp64 : enable
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

__kernel void hotspotOpt1(__global float *p, __global float* tIn, __global float *tOut, float sdc,
                            int nx, int ny, int nz,
                            float ce, float cw, 
                            float cn, float cs,
                            float ct, float cb, 
                            float cc) 
{
  float amb_temp = 80.0;

  int i = get_global_id(0);
  int j = get_global_id(1);
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

  // calculate for each layers
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
  return;
}

__kernel void hotspotOpt1_throttling(__global float *p, __global float* tIn, __global float *tOut, float sdc,
                                      int nx, int ny, int nz,
                                      float ce, float cw, 
                                      float cn, float cs,
                                      float ct, float cb, 
                                      float cc,
                                      int throttling_mod, int throttling_alloc) 
{
  float amb_temp = 80.0;

  volatile __local int local_worklist[1];
  if (get_local_id(0) == 0 && get_local_id(1))
    local_worklist[0] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (get_local_id(0) % throttling_mod < throttling_alloc) {
    for (int dynamic_work = atomic_inc(local_worklist);
        dynamic_work < get_local_size(0) * get_local_size(1);
        dynamic_work = atomic_inc(local_worklist)){
      int global_id_0 = get_group_id(0) * get_local_size(0) + get_global_offset(0) +
                        dynamic_work / get_local_size(1);
      int global_id_1 = get_group_id(1) * get_local_size(1) + get_global_offset(1) +
                        dynamic_work % get_local_size(1);

      int i = global_id_0;
      int j = global_id_1;
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
  return;
}

__kernel void hotspotOpt1_MBAP(__global float *p, __global float* tIn, __global float *tOut, float sdc,
                              int nx, int ny, int nz,
                              float ce, float cw, 
                              float cn, float cs,
                              float ct, float cb, 
                              float cc,
                              int throttling_mod,
                              int throttling_alloc,
                              __global atomic_int* worklist,
                              int n_tasks) 
{
  float amb_temp = 80.0;

  __local int tmp[1];
  tmp[0] = atomic_load(worklist);

  __local int local_worklist[1];

  if (get_local_id(0) % throttling_mod < throttling_alloc) {
    for (int wg_id = gpu_first(tmp, worklist);
              gpu_more(wg_id, n_tasks);
              wg_id = gpu_next(tmp, worklist)) {
      int group_id_0 = wg_id / (get_global_size(1)/get_local_size(1));
      int group_id_1 = wg_id % (get_global_size(1)/get_local_size(1));

      local_worklist[0] = 0;

      for (int dynamic_work = atomic_add(local_worklist, 1);
                dynamic_work < get_local_size(0) * get_local_size(1);
                dynamic_work = atomic_add(local_worklist, 1)) {
        int i = group_id_0 * get_local_size(0) +
                          dynamic_work / get_local_size(1);
        int j = group_id_1 * get_local_size(1) +
                          dynamic_work % get_local_size(1);
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
  return;
}

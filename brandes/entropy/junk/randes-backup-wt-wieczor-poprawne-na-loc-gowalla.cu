#include "cuda.h"
#include "common/errors.h"

#include <time.h>
#include <stdio.h>
#include <math.h>

#define MAX_EDGES 4000000
#define MAX_VERTICES 450000
#define THREADS_PER_BLOCK 32
#define BLOCKS 256
#define INF INT_MAX


__global__ void brandes_kernel(int *edges_x, int *edges_y, double *bc, double *delta, int *sigma, int *dist, int no_vertices, int no_edges) {
  int u, v, e, source;
  __shared__ bool cont;
  __shared__ int level;

  int t_idx = threadIdx.x;
  int b_idx = blockIdx.x;
  int offset = no_vertices * b_idx;

  if (b_idx >= no_vertices || (t_idx >= no_vertices && t_idx >= no_edges))
    return;

  for (source = blockIdx.x; source < no_vertices; source += gridDim.x) {
    if (t_idx == 0) {
      level = -1;
      cont = true;
    }
    for (v = t_idx; v < no_vertices; v += blockDim.x) {
      delta[offset + v] = 0.;
      sigma[offset + v] = (v == source);
      dist[offset + v] = (v == source) ? 0 : INF; 
    }

    // Forward phase
    while (cont) {
      if (t_idx == 0) {
        level++;
        cont = false;
      }
      __syncthreads();
      
      for (e = t_idx; e < no_edges; e += blockDim.x) {
        u = edges_x[e];
        if (dist[offset + u] == level) {
          v = edges_y[e];
          if (dist[offset + v] == INF) {
            dist[offset + v] = dist[offset + u] + 1;
            cont = true;
          }
          if (dist[offset + v] == (dist[offset + u] + 1)) atomicAdd(&sigma[offset + v], sigma[offset + u]);
        }
      }
      __syncthreads();
    }

    // Backward phase
    while (level > 0) {
      if (t_idx == 0) level--;
      __syncthreads();

      for (e = t_idx; e < no_edges; e += blockDim.x) {
        u = edges_x[e];
        if (dist[offset + u] == level) {
          v = edges_y[e];
          if ((dist[offset + v] == (dist[offset + u] + 1)) && (sigma[offset + v] != 0))
            atomicAdd(&delta[offset + u], (sigma[offset + u] * 1.0 / sigma[offset + v]) * (1 + delta[offset + v]));
        }
      }
      __syncthreads();
    }

    for (v = t_idx; v < no_vertices; v += blockDim.x)
      if (v != source) atomicAdd(&bc[v], delta[offset + v]);
    //if (v != source) bc[v] += delta[offset + v];
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("usage: ./brandes <input-file> <output-file>\n");
    exit(1);
  }

  FILE *in_file;
  FILE *out_file;
  in_file = fopen(argv[1], "r");
  out_file = fopen(argv[2], "w+");

  // Host data initialization
  int a, b, no_edges = 0, no_vertices = 0;
  int* edges_x = (int*)malloc(MAX_EDGES * sizeof(int));
  int* edges_y = (int*)malloc(MAX_EDGES * sizeof(int));
  int* deg = (int*)malloc(MAX_VERTICES * sizeof(int));
  memset(deg, 0, MAX_VERTICES * sizeof(int));

  fscanf (in_file, "%d %d", &a, &b);
  edges_x[no_edges] = a;
  edges_y[no_edges] = b;
  no_edges += 1;
  edges_x[no_edges] = b;
  edges_y[no_edges] = a;
  no_edges += 1;
  deg[a]++;
  deg[b]++;
  no_vertices = max(max(a, b), no_vertices - 1) + 1;

  while (!feof (in_file)) {
    fscanf(in_file, "%d %d\n", &a, &b);    
    edges_x[no_edges] = a;
    edges_y[no_edges] = b;  
    no_edges += 1;
    edges_x[no_edges] = b;
    edges_y[no_edges] = a;  
    no_edges += 1;
    deg[a]++;
    deg[b]++;
    no_vertices = max(max(a, b), no_vertices - 1) + 1;

    if (no_edges > MAX_EDGES) {
      printf("More edges than allowed (2 mln)\n");
      exit(1);
    }
  }

  printf("no_vertices %d\n", no_vertices);
  printf("no_edges %d\n", no_edges);

  /*
  int no_deg[200] = {0}; 

  for (int i = 0; i < no_vertices; i++) {
    if (deg[i] < 199)
      no_deg[deg[i]]++;
    else
     no_deg[199]++;
  }

  for (int i = 0; i < 200; i++) {
    printf("no_deg[%d] = %d\n", i, no_deg[i]);
  }
*/
  double *bc = (double*)malloc(no_vertices * BLOCKS * sizeof(double));

  // CUDA malloc
  int *dist, *sigma, *dev_edges_x, *dev_edges_y; 
  double *dev_bc, *delta;
  
  HANDLE_ERROR(cudaMalloc((int**)&dev_edges_x, no_edges * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((int**)&dev_edges_y, no_edges * sizeof(int)));

  HANDLE_ERROR(cudaMalloc((int**)&dist, no_vertices * BLOCKS * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((int**)&sigma, no_vertices * BLOCKS * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((double**)&dev_bc, no_vertices * BLOCKS * sizeof(double)));
  HANDLE_ERROR(cudaMalloc((double**)&delta, no_vertices * BLOCKS * sizeof(double)));

  HANDLE_ERROR(cudaMemset(dev_bc, 0, no_vertices * BLOCKS * sizeof(double)));


  // Transfer data to the device
  cudaEvent_t start, start_kernel, stop, stop_kernel;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));
    
  HANDLE_ERROR(cudaMemcpy(dev_edges_x, edges_x, no_edges * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_edges_y, edges_y, no_edges * sizeof(int), cudaMemcpyHostToDevice));


  // Run the kernel
  HANDLE_ERROR(cudaEventCreate(&start_kernel));
  HANDLE_ERROR(cudaEventCreate(&stop_kernel));
  HANDLE_ERROR(cudaEventRecord(start_kernel, 0));

  brandes_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_edges_x, dev_edges_y, dev_bc, delta, sigma, dist, no_vertices, no_edges);

  //HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaEventRecord(stop_kernel, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop_kernel));
  float elapsed_time_kernel;
  HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time_kernel, start_kernel, stop_kernel));
  printf("Kernel running time: %3.1f ms\n", elapsed_time_kernel);
  HANDLE_ERROR(cudaEventDestroy(start_kernel));
  HANDLE_ERROR(cudaEventDestroy(stop_kernel));
  

  // Transfer results from device to the host
  HANDLE_ERROR(cudaMemcpy(bc, dev_bc, no_vertices*sizeof(double), cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  float elapsedTime;
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Total time: %3.1f ms\n", elapsedTime);
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));


  // Save the results to the output file
  for (int i = 0; i < no_vertices; i++) {
    fprintf(out_file, "%f\n", bc[i]);
  }

  // Clean up 
  free(edges_x);
  free(edges_y);
  free(bc);
  cudaFree(dev_edges_x);
  cudaFree(dev_edges_y);
  cudaFree(dev_bc);
  cudaFree(delta);
  cudaFree(sigma);
  cudaFree(dist);

  fclose(in_file);
  fclose(out_file);

  return 0;
}
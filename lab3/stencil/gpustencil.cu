#include <time.h>
#include <stdio.h>
#include <math.h>

#define RADIUS        3000
#define NUM_ELEMENTS  1000000

static void handleError(cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define cudaCheck( err ) (handleError(err, __FILE__, __LINE__ ))

__global__ void stencil_1d(int *in, int *out) {
  //PUT YOUR CODE HERE
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < NUM_ELEMENTS) {
    int total = 0;
    for (int j = max(0, i - RADIUS); j < min(NUM_ELEMENTS, i + RADIUS); j++) {
      total += in[j];
    }
    out[i] = total;
  }
}

void cpu_stencil_1d(int *in, int *out) {
  for (int i = 0; i < NUM_ELEMENTS; i++) {
    int total = 0;
    for (int j = max(0, i - RADIUS); j < min(NUM_ELEMENTS, i + RADIUS); j++) {
      total += in[j];
    }
    out[i] = total;
  }
}

int main() {
  //PUT YOUR CODE HERE - INPUT AND OUTPUT ARRAYS
  int *in, *out, *d_in, *d_out;

  in  = (int*)malloc(sizeof(int) * NUM_ELEMENTS);
  out = (int*)malloc(sizeof(int) * NUM_ELEMENTS);

  for (int i = 0; i < NUM_ELEMENTS; i++) {
    in[i]  = 1;
    out[i] = 0;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );

  //PUT YOUR CODE HERE - DEVICE MEMORY ALLOCATION
  cudaCheck(cudaMalloc((void**)&d_in,  sizeof(int) * NUM_ELEMENTS));
  cudaCheck(cudaMalloc((void**)&d_out, sizeof(int) * NUM_ELEMENTS));

  cudaCheck(cudaMemcpy(d_in,  in, sizeof(int) * NUM_ELEMENTS, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_out, in, sizeof(int) * NUM_ELEMENTS, cudaMemcpyHostToDevice));

  cudaEvent_t start_k, stop_k;
  cudaEventCreate(&start_k);
  cudaEventCreate(&stop_k);
  cudaEventRecord( start_k, 0 );

  //PUT YOUR CODE HERE - KERNEL EXECUTION
  stencil_1d<<<((NUM_ELEMENTS+1024)/1024), 1024>>>(d_in, d_out);

  cudaEventRecord(stop_k, 0);
  cudaEventSynchronize(stop_k);
  float elapsedTime_k;
  cudaEventElapsedTime( &elapsedTime_k, start_k, stop_k);
  printf("GPU kernel execution time:  %3.1f ms\n", elapsedTime_k);
  cudaEventDestroy(start_k);
  cudaEventDestroy(stop_k);

  cudaCheck(cudaPeekAtLastError());

  //PUT YOUR CODE HERE - COPY RESULT FROM DEVICE TO HOST
  cudaCheck(cudaMemcpy(out, d_out, sizeof(int) * NUM_ELEMENTS, cudaMemcpyDeviceToHost));
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, start, stop);
  printf("Total GPU execution time:  %3.1f ms\n", elapsedTime);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  //PUT YOUR CODE HERE - FREE DEVICE MEMORY  
  cudaCheck(cudaFree(d_in));
  cudaCheck(cudaFree(d_out));

  for (int i = 0; i < NUM_ELEMENTS; i++) {
    in[i]  = 1;
    out[i] = 0;
  }

  struct timespec cpu_start, cpu_stop;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);

  cpu_stencil_1d(in, out);

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_stop);
  double result = (cpu_stop.tv_sec - cpu_start.tv_sec) * 1e3 + (cpu_stop.tv_nsec - cpu_start.tv_nsec) / 1e6;
  printf( "CPU execution time:  %3.1f ms\n", result);

  return 0;
}



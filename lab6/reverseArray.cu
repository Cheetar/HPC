#include <stdio.h>
#include <assert.h>

#define THREADS_PER_BLOCK 768
#define ARRAY_SIZE THREADS_PER_BLOCK * 1024

static void HandleError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__ ))

__global__ void reverseArray(int *inArray, int *outArray)
{
    int inOffset = blockDim.x * blockIdx.x;
    int outOffset = blockDim.x * (gridDim.x - 1 - blockIdx.x);
    int inIndex = inOffset + threadIdx.x;
    int outIndex = outOffset + (blockDim.x - 1 - threadIdx.x);
    outArray[outIndex] = inArray[inIndex];
}

int main(void)
{
    int *hostArray;
    int *devInArray, *devOutArray;

    int numBlocks = ARRAY_SIZE / THREADS_PER_BLOCK;

    size_t memSize = ARRAY_SIZE * sizeof(int);
    hostArray = (int *)malloc(memSize);
    HANDLE_ERROR(cudaMalloc((void **)&devInArray, memSize));
    HANDLE_ERROR(cudaMalloc((void **)&devOutArray, memSize));

    for (int i = 0; i < ARRAY_SIZE; i++) {
        hostArray[i] = i;
    }

    HANDLE_ERROR(cudaMemcpy(devInArray, hostArray, memSize, cudaMemcpyHostToDevice));

    dim3 dimGrid(numBlocks);
    dim3 dimBlock(THREADS_PER_BLOCK);
    reverseArray<<<dimGrid, dimBlock>>>  (devInArray, devOutArray);

    cudaThreadSynchronize();

    HANDLE_ERROR(cudaMemcpy(hostArray, devOutArray, memSize, cudaMemcpyDeviceToHost));

    for (int i = 0; i < ARRAY_SIZE; i++) {
        assert(hostArray[i] == ARRAY_SIZE - 1 - i);
    }

    HANDLE_ERROR(cudaFree(devInArray));
    HANDLE_ERROR(cudaFree(devOutArray));

    free(hostArray);

    printf("Correct!\n");
    return 0;
}

